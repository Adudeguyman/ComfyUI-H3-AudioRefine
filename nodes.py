"""ComfyUI-H3-AudioRefine

Audio-only refinement pass for MiniMax H3 packed AV latents.

Takes a fully sampled H3 AV latent (e.g. from a 4-step Turbo pass), freezes the
video stream, and runs additional denoising steps on the audio stream only.
The video stream is protected by a per-stream denoise mask (0 = preserve):
ComfyUI's native masked path injects the clean video at the visual cond
timestep every step -- the same mechanism keyframe conditioning uses -- so the
model denoises audio *in the context of* the finished video, and the final
blend returns the video slice bit-identical to the input.

Compute note: H3 is a single-stream transformer over one packed token
sequence, so each audio refinement step still costs close to a full forward
pass (the frozen video tokens remain in the sequence as attention context).
The saving is step arithmetic (e.g. 4 turbo steps + 4 audio steps vs 20),
not per-step cost.

Two nodes:

- H3 Audio Refine Mask: attaches the freeze-video / generate-audio noise mask
  to a sampled AV latent. Feed the result to a stock sampler
  (SamplerCustomAdvanced + BasicScheduler with denoise < 1.0, or KSampler
  with denoise < 1.0).

- H3 Audio Refine Sampler: all-in-one convenience node that builds the mask
  and runs the refinement pass internally (mirrors the stock KSampler path).
"""

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


def _require_av_nested(samples):
    """Validate the latent is an H3-style packed AV NestedTensor; return (video, audio)."""
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "H3-AudioRefine: latent is not a packed AV latent (expected a nested "
            "video+audio latent from a MiniMax H3 sampling pass, got a plain tensor)."
        )
    streams = samples.unbind()
    if len(streams) < 2:
        raise ValueError(
            "H3-AudioRefine: nested latent has %d stream(s), expected 2 (video, audio)."
            % len(streams)
        )
    video, audio = streams[0], streams[1]
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "H3-AudioRefine: unexpected stream shapes video=%s audio=%s "
            "(expected video [B,C,T,H,W] and audio [B,C,2,T])."
            % (list(video.shape), list(audio.shape))
        )
    return video, audio


def _build_av_noise_mask(video, audio, video_denoise_mask_value):
    """Per-stream denoise masks: video preserved (or partially opened), audio generated.

    Masks are built at full stream spatial/temporal shape with 1 channel;
    core's reshape_mask broadcasts channels. 0.0 = preserve, 1.0 = generate.
    """
    v = torch.full(
        (1, 1, video.shape[2], video.shape[3], video.shape[4]),
        float(video_denoise_mask_value), dtype=torch.float32,
    )
    a = torch.ones((1, 1, audio.shape[2], audio.shape[3]), dtype=torch.float32)
    return comfy.nested_tensor.NestedTensor((v, a))


class H3AudioRefineMask:
    """Attach a freeze-video / generate-audio noise mask to a sampled H3 AV latent."""

    CATEGORY = "latent/minimax"
    FUNCTION = "apply"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = (
        "Marks the video stream of a sampled MiniMax H3 AV latent as preserved and the "
        "audio stream as generated, so a following sampler pass at denoise < 1.0 refines "
        "audio only. Video returns bit-identical (at video_denoise 0.0). Wire the output "
        "to the sampler's latent input and set the scheduler/sampler denoise to the "
        "desired audio re-noise depth (0.3-0.6 typical)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Sampled MiniMax H3 AV latent (video+audio) from a previous pass."}),
            },
            "optional": {
                "video_denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise mask value for the video stream. 0.0 freezes video exactly. "
                               "Values > 0.0 let the refinement pass partially rework video too.",
                }),
            },
        }

    def apply(self, latent, video_denoise=0.0):
        video, audio = _require_av_nested(latent["samples"])
        out = latent.copy()
        out["noise_mask"] = _build_av_noise_mask(video, audio, video_denoise)
        return (out,)


class H3AudioRefineSampler:
    """All-in-one audio refinement pass: mask + partial-denoise sampling, video frozen."""

    CATEGORY = "sampling/minimax"
    FUNCTION = "refine"
    RETURN_TYPES = ("LATENT",)
    DESCRIPTION = (
        "Runs extra denoising steps on the audio stream of a sampled MiniMax H3 AV latent "
        "while the video stream is held frozen as clean context (native masked path, video "
        "returns bit-identical). audio_denoise sets how far the audio is re-noised on the "
        "shared schedule before refinement. Each step still costs near a full model forward; "
        "the saving is total step count."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 model, with the same LoRA/patch stack you intend to refine with."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "Sampled MiniMax H3 AV latent (video+audio) from the first pass."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100,
                                  "tooltip": "Refinement steps. These run at denoise depth audio_denoise (KSampler-style: the full schedule is steps/audio_denoise long and only the tail runs)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Keep at the value the first pass used (1.0 for Turbo LoRA passes)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "audio_denoise": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                            "tooltip": "How far the audio is re-noised on the shared (video) schedule before refining. "
                                                       "0.3-0.6 keeps pass-1 audio content; 1.0 regenerates audio from scratch against the frozen video."}),
            },
            "optional": {
                "video_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "0.0 freezes video exactly (default). > 0.0 partially opens video to the refinement pass."}),
            },
        }

    def refine(self, model, positive, negative, latent, seed, steps, cfg,
               sampler_name, scheduler, audio_denoise, video_denoise=0.0):
        video, audio = _require_av_nested(latent["samples"])
        latent_image = latent["samples"]
        noise_mask = _build_av_noise_mask(video, audio, video_denoise)

        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise=audio_denoise, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        out = latent.copy()
        out.pop("noise_mask", None)
        out["samples"] = samples
        return (out,)


class H3FrozenVideoCache:
    """Patch the model with the frozen-video KV cache accelerator for refinement passes."""

    CATEGORY = "model/minimax"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = (
        "Accelerates the H3 audio refinement pass: caches each block's K/V for the frozen "
        "video/text/cond rows on the first refinement step and computes only the audio rows "
        "afterwards. Approximation: cached rows stop reacting to the audio between rebuilds "
        "(refresh_interval > 0 rebuilds every N steps at full cost). Only activates when the "
        "model runs with video fully frozen and audio fully generated -- all other calls, "
        "including normal pass-1 sampling, pass through exactly. Place after the LoRA/patch "
        "stack, feed the patched MODEL to the refine sampler. Cache size and backend choice "
        "are printed to the console on every build."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 model, after the LoRA/patch stack."}),
                "cache_contents": (["hidden", "kv"], {
                    "default": "hidden",
                    "tooltip": "What to cache per block for the frozen rows. hidden: post-norm hidden "
                               "states (~2.7x smaller: ~5.3 GB int4 at full canvas), K/V rebuilt on the "
                               "fly each step (~30% of video compute remains; ~3x faster cached steps). "
                               "kv: post-rope K/V directly (~14 GB int4), cached steps nearly free -- "
                               "needs the RAM/VRAM to hold it."}),
                "backend": (["auto", "vram", "ram", "disk"], {
                    "default": "auto",
                    "tooltip": "Where the cache lives. auto picks the first of vram/ram/disk that fits "
                               "(with margin) and reports the choice. disk works on any machine."}),
                "precision": (["int4", "fp8", "bf16"], {
                    "default": "int4",
                    "tooltip": "Cache storage precision. int4 (group-128) is smallest and fastest to "
                               "stream (~14 GB at full canvas); fp8 ~27 GB; bf16 ~55 GB, exact."}),
                "refresh_interval": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "Rebuild the cache every N refinement steps (each rebuild costs one "
                               "full-price step) so the frozen rows periodically see the current "
                               "audio. 0 = build once, never refresh."}),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log per-step diagnostics to the console: how many blocks took the "
                               "cached / build / stock path, time spent inside the patched blocks, "
                               "and time spent in the rest of the model call. Use this to find out "
                               "where a refinement step is actually spending its time."}),
                "allow_disk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow the cache to be written to disk. OFF by default: a disk "
                               "cache writes the whole thing (several GB) to your drive on every "
                               "build, which is significant SSD wear over repeated runs. With this "
                               "off, the node raises a clear error instead of falling back to disk "
                               "when the cache does not fit in VRAM or RAM."}),
            },
        }

    def patch(self, model, cache_contents, backend, precision, refresh_interval, verbose=False,
              allow_disk=False):
        from . import frozen_cache
        return (frozen_cache.patch_model(model, backend, precision, refresh_interval,
                                         cache_contents=cache_contents, verbose=verbose,
                                         allow_disk=allow_disk),)


NODE_CLASS_MAPPINGS = {
    "H3AudioRefineMask": H3AudioRefineMask,
    "H3AudioRefineSampler": H3AudioRefineSampler,
    "H3FrozenVideoCache": H3FrozenVideoCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AudioRefineMask": "H3 Audio Refine Mask",
    "H3AudioRefineSampler": "H3 Audio Refine Sampler",
    "H3FrozenVideoCache": "H3 Frozen Video Cache",
}
