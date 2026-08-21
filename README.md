# ComfyUI-H3-AudioRefine

Audio-only refinement pass for MiniMax H3 packed AV latents. v1.0.0.

Freeze the video stream of a sampled H3 latent (e.g. from a 4-step Turbo pass)
and run additional denoising steps on the audio stream only. The intended use
is Turbo-LoRA workflows where 4-step video is acceptable but 4-step audio is
not: pay for a few extra audio steps instead of raising the whole joint pass
to 8-20 steps.

## How it works

ComfyUI's native MiniMax H3 support already contains a masked-inpaint path for
the packed AV latent (`MiniMaxH3.scale_latent_inpaint` /
`_denoise_mask_conds` in `comfy/model_base.py`), and the sampler accepts a
per-stream `noise_mask` as a NestedTensor. This pack builds that mask
(video = 0.0 preserve, audio = 1.0 generate) and runs a partial-denoise pass:

- Each step, the frozen video slice is injected at the visual cond timestep
  (0.999) -- the same treatment keyframe conditioning gets -- so the model
  denoises the audio *in the context of* the finished video.
- The sampler's final masked blend returns the video slice bit-identical to
  the input (at `video_denoise` 0.0).
- The audio stream rides ComfyUI's ModelSamplingAV carry, so its effective
  sigma stays on the trained dual-schedule relationship. No custom sampler.

**Compute expectation:** H3 is a single-stream transformer over one packed
token sequence. Frozen video tokens remain in the sequence as attention
context, so each audio refinement step still costs close to a full forward
pass. The saving is step arithmetic (4 turbo + 4-6 audio ~= 8-10 full-cost
steps vs 20), not per-step cost.

## Nodes

### H3 Audio Refine Sampler (all-in-one)

Inputs: `model`, `positive`, `negative`, `latent` (the sampled AV latent),
`seed`, `steps`, `cfg`, `sampler_name` (default euler), `scheduler` (default
simple), `audio_denoise` (default 0.5), optional `video_denoise` (default 0.0).

Wire the first pass's sampler LATENT output straight into `latent`, reuse the
same model stack (including the Turbo LoRA) and conditioning, and take the
refined LATENT to your existing VAE decode nodes. `audio_denoise` controls how
far the audio is re-noised before refinement:

- 0.3-0.6: keep pass-1 audio content, clean up noise floor / artifacts.
- 1.0: regenerate the audio from scratch against the frozen video.

`steps` runs KSampler-style at that denoise depth (the schedule is
`steps / audio_denoise` long and only the tail executes).

### H3 Audio Refine Mask (composable)

Takes the sampled LATENT, attaches the freeze-video / generate-audio noise
mask, outputs LATENT. Feed it to a stock `SamplerCustomAdvanced` (with
`BasicScheduler` at `denoise` = desired audio re-noise depth) or a stock
`KSampler` with `denoise` < 1.0. Use this variant to control the refinement
schedule/guider yourself or to A/B against the all-in-one node.

Optional `video_denoise` > 0.0 partially opens the video stream to the
refinement pass instead of freezing it exactly.

## Suggested starting point (Turbo 4-step)

1. Pass 1: your existing 4-step Turbo graph, unchanged.
2. H3 Audio Refine Sampler: steps 4-6, audio_denoise 0.5, cfg 1.0,
   euler / simple, same model+LoRA and conditioning as pass 1.
3. Decode as usual.

A/B against a straight 6-8 step Turbo run at matched total step count -- that
is the honest baseline this approach has to beat.

## Requirements

- ComfyUI with native MiniMax H3 support including the AV masked path
  (0.33.x verified against master as of 2026-08-21; the per-stream
  denoise-mask handling in `CFGGuider.sample` and
  `MiniMaxH3.scale_latent_inpaint` must be present).
- No extra Python dependencies. No monkeypatching: only public
  `comfy.sample` / `noise_mask` APIs are used.

## Notes / limits

- Refining with a different model stack than pass 1 (e.g. dropping the Turbo
  LoRA for the refinement pass) is allowed by the graph and worth testing,
  but is untested territory: the base model at 4-6 tail steps may fight the
  distilled trajectory the audio was started on.
- `latent` must be a sampled H3 AV latent (nested video+audio). Plain latents
  are rejected with a clear error.
- Conditioning with keyframes/refs is passed through untouched; the frozen
  video mask stacks with them via the native pooled-mask path.

## Changelog

- 1.0.0: Initial release. H3AudioRefineMask, H3AudioRefineSampler.
