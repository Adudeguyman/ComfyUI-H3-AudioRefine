# Technical rundown

How the frozen video cache works, why it is built this way, and what to be careful about
if you modify it. Written for someone who has not seen this code or MiniMax H3 before.

The user-facing documentation is in [README.md](README.md) — what the knobs do and how to
diagnose a slow pass. This document is about the mechanism.

---

## 1. The problem

MiniMax H3 generates video and audio jointly. Both live in a single latent — a
`NestedTensor` of a video tensor `[B, 24, T, H/16, W/16]` and an audio tensor
`[B, 32, 2, T*40]`. The model flattens both into one sequence of rows:

```
[ text | cond/ref rows | audio | video ]
```

and runs 50 transformer blocks over it with **full bidirectional attention** — no causal
mask, no cross-attention split, `mask=None`. Every row attends to every other row at
every block.

The proportions are lopsided. On a 1344×768, 124-frame clip the video occupies roughly
37,000 rows and the audio roughly 400 — about **1%** of the sequence.

Turbo LoRAs make this awkward. Four-step video is often acceptable while four-step audio
is not, so you want more denoising steps on the audio without paying for more video. The
refinement pass does exactly that: freeze the video, run extra steps on the audio only.

ComfyUI supports the freezing natively. A per-stream `denoise_mask` (video 0 = preserve,
audio 1 = generate) causes the sampler to pin the video rows to the clean latent every
step:

```python
# comfy/samplers.py
x = x * denoise_mask + inner_model.scale_latent_inpaint(...) * (1. - denoise_mask)
...
out = out * denoise_mask + self.latent_image * latent_mask
```

For H3, `scale_latent_inpaint` returns the latent **unmodified** (`comfy/model_base.py`)
rather than re-noising it, and the frozen rows are fed at a fixed conditioning timestep
(`VISUAL_COND_TIMESTEP = 0.999`) instead of the evolving one — the same treatment
keyframe conditioning gets. The final blend then discards whatever the model produced for
those rows, so the returned video is bit-identical to the input.

**But the cost does not go away.** Because attention is a single fused sequence, all
37,000 frozen video rows still get their qkv projection, attention, and MLP at every one
of the 50 blocks — full price — and the result is thrown away. Measured on an RTX 5090: a
refinement step costs 20.7s against 23.0s for a full joint generation step. Freezing the
video saves almost nothing by itself.

That video compute is not *pointless* — the audio rows attend to the video rows' K/V at
every block, which is what keeps refined audio matched to the picture. It is only
**redundant**, because the frozen rows' inputs never change between steps, so the same
values are recomputed identically every time.

---

## 2. The idea

Compute the frozen rows once, store each block's attention input, and on subsequent steps
compute only the audio rows — attending against the stored values.

What has to be stored is the **input to each block's attention**, for the whole packed
sequence. Two choices, and this is the `cache_contents` dial:

| mode | what is stored | width per row | on a cached step |
|---|---|---|---|
| `hidden` (default) | post-norm, modulation-applied hidden state `h`, the tensor immediately before `qkv_proj` | 5376 | K/V rebuilt on the fly via `qkv_proj` + rope |
| `kv` | post-rope K and V | 14336 (7168 each) | used directly |

`hidden` is 2.7× smaller but keeps the qkv projection in the per-step cost. That
projection is about 30% of a block's per-row matmul work:

```
per block, per row (params, from the H3 config: hidden 5376, ffn 14336, kv dim 7168)
  qkv       5376 × 21504 = 115.6M   <-- rebuilt in hidden mode
  out_proj  7168 ×  5376 =  38.5M
  fc1       5376 × 28672 = 154.1M
  fc2      14336 ×  5376 =  77.1M
                    total = 385.4M
```

Two things are deliberately **not** cached:

- **Block outputs.** The evolving `x` for the frozen rows is never stored. On cached steps
  those rows are left stale and produce garbage, which the sampler's final mask blend
  discards. Only the attention *input* is needed.
- **A meaningful copy of the audio rows.** Audio is stored along with everything else on
  the build step purely so the sequence stays contiguous and the attention call stays a
  single op. On every cached step the audio slice is overwritten in place with freshly
  computed values (`dq_h[aa:ab] = live`), so the stored audio is never used.

### The approximation

The frozen rows' cached inputs were computed against the audio as it existed at build
time. They do not see how the audio evolves afterwards. In other words, the **video→audio
attention edge is severed** between rebuilds — audio still attends to video, but video no
longer reacts to audio.

`refresh_interval N` rebuilds every N steps (each rebuild costs one full-price step) to
periodically reopen that path. `0` means build once.

Measured deviation on the CPU test model, cached step vs stock: bf16 ~0, fp8 ~4e-4
relative, int4 ~2e-3 relative — small next to the approximation itself.

---

## 3. Where it hooks into ComfyUI

Two hooks, both stock extension points. No monkeypatching.

**Block replacement.** H3's block loop checks a patch table:

```python
# comfy/ldm/minimax/model.py
blocks_replace = patches_replace.get("dit", {})
for i, block in enumerate(self.blocks):
    if ("double_block", i) in blocks_replace:
        h = blocks_replace[("double_block", i)](
            {"img": h, "t_emb": t_emb, "mod_segments": mod_segments,
             "rope_freqs": rope_freqs, "transformer_options": transformer_options},
            {"original_block": block_wrap})["img"]
    else:
        h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
```

`patch_model()` registers a replacement for all 50 blocks via
`ModelPatcher.set_model_patch_replace(..., "dit", "double_block", i)`.

**Model wrapper.** A `WrappersMP.DIFFUSION_MODEL` wrapper runs around the whole forward.
It decides, once per model call, whether this call is a frozen-video refinement step and
what mode the blocks should run in. The block replacement itself is nearly stateless — it
reads `state.mode` and dispatches.

This split matters: the gating decision needs the denoise mask and payload, which are only
visible at the model-call level, while the work happens per block.

**Numerical fidelity.** The build path is a mirror of the stock block, calling core's own
kernels — `comfy.quant_ops.ck.rms_rope_split_half_` (fused per-head RMSNorm + partial
split-half rope), `optimized_attention`, and the block's own `mlp`, `adaln_proj`,
`norm1/2`, and `_mod_scale_shift` / `_mod_gate` helpers from
`comfy.ldm.minimax.model`. Nothing is reimplemented. Verified bit-identical (0.00e+00) to
the stock forward on CPU.

`check_core_compat()` verifies the structural assumptions at patch time and again per
step, and falls back loudly to the exact path rather than producing wrong output.

---

## 4. Activation gating

The patch must be inert unless the call really is a frozen-video refinement step, so the
same patched MODEL can be wired anywhere — including into pass-1 sampling — without
changing results.

```python
activate = (
    layout is not None                                   # H3 packed payload present
    and denoise_mask is not None
    and isinstance(x, (list, tuple)) and len(x) == 2      # video + audio streams
    and float(denoise_mask.max()) < 1e-3                  # video fully frozen
    and (audio_mask is None or float(audio_mask.max()) > 1e-3)   # audio still generating
)
```

The subtlety is in the last line. Core splits a packed two-stream mask into **two**
kwargs:

```python
# comfy/model_base.py, MiniMaxH3.extra_conds
denoise_mask = utils.unpack_latents(denoise_mask, kwargs["latent_shapes"])
if len(denoise_mask) > 1:
    audio_denoise_mask = denoise_mask[1]
denoise_mask = denoise_mask[0]
```

So on a real refinement call `denoise_mask` is video-only and `audio_denoise_mask` is
always present. An earlier version of this gate required `audio_denoise_mask` to be
absent, which meant it never fired — correct output, zero speedup, and no error. See
[§8](#8-bugs-worth-knowing-about).

If audio is *also* frozen there is nothing to refine, and the stock path is the honest
answer.

---

## 5. Cache identity and invalidation

Caches live in `_State.slots`, keyed and invalidated separately.

**Key** — `(layout signature, context shape)`. Up to 2 slots, LRU.

**Invalidation** is by content fingerprint (cheap `sum` and `abs().sum()` of the tensor),
not by object identity:

| reason | trigger |
|---|---|
| `no cache yet` | first call for this key |
| `layout changed` | packed layout signature differs |
| `video latent changed` | new seed, or a different pass-1 result |
| `conditioning changed` | different prompt embedding |
| `refresh_interval reached` | N cached steps since the last build |

Every build logs its reason. A build line on *every* step means the cache is never being
reused, and the reason field says what is invalidating it.

**Do not key on `data_ptr()`.** Core allocates a fresh context tensor for every model call
(`samplers.cond_cat` → `CONDRegular.concat` → `torch.cat`), so pointer identity is
unstable across steps. An earlier version did this and rebuilt every single step.

**Interrupted builds.** Validity fingerprints are stamped only *after* the build completes
every block, and the store is freed if the executor raises. A half-written cache is never
marked valid — otherwise an interrupted first step would leave zeroed K/V for the
un-built blocks and produce silently wrong audio on the next run.

---

## 6. Storage

### Codecs

Storage precision only. **No 4-bit arithmetic is ever executed** — values are dequantized
to the working dtype before any matmul or attention, so `int4` needs no particular GPU
generation and runs fine on CPU.

| codec | scheme | notes |
|---|---|---|
| `bf16` | none | exact, largest |
| `fp8` | per-row absmax scaled e4m3, `scale = absmax / 448` | needs `torch.float8_e4m3fn` |
| `int4` | group-128 symmetric, `scale = absmax / 7`, values `[-7, 7]` offset to `[0, 14]`, nibble-packed two per byte, fp16 scales | default |

### Backends

All implement `put` / `get` / `begin_step` / `end_step` / `free`.

- **`vram`** — tensors held on device. Fastest, but not tracked by ComfyUI's memory
  manager, so it competes with resident weights.
- **`ram`** — pinned host memory for DMA without a staging copy. Falls back to pageable
  memory with a warning if pinning fails, so RAM pressure degrades to swapping instead of
  an OOM kill.
- **`disk`** — `np.lib.format.open_memmap` per tensor kind, with a double-buffered
  read-ahead thread staying one block ahead of the consumer. **Off unless `allow_disk` is
  set**, because it writes the entire cache on every build — gigabytes per run, and real
  SSD wear.

`auto` picks the first that fits with a 4 GB margin — free VRAM via
`comfy.model_management.get_free_memory`, then `MemAvailable` from `/proc/meminfo` — and
logs the choice and the reason. It will not fall through to disk unless `allow_disk` is
on; it raises an error naming what did not fit instead.

### Memory reporting

`MemAvailable` is a kernel *estimate* of what could be reclaimed under pressure, not a
measurement, and it counts reclaimable page cache and slab. It is the right input for a
pre-allocation decision and the wrong thing to compare against afterwards. Builds
therefore log the estimate alongside the measured process RSS delta.

Note that **pinned memory is not returned to the OS on free** — PyTorch's
`CachingHostAllocator` retains freed pinned blocks for reuse (which is why
`torch._C._host_emptyCache()` exists). RSS stays elevated after a free; the next build
reuses those blocks. The free log says so explicitly.

Sizes at 1344×768 / 124 frames (~38k rows, 50 blocks):

| | `bf16` | `fp8` | `int4` |
|---|---|---|---|
| `kv` | ~55 GB | ~27 GB | ~14 GB |
| `hidden` | ~21 GB | ~10.4 GB | ~5.3 GB |

Plus working VRAM during cached steps, which is not in that table: `kv` holds two
full-sequence dequant buffers (~1.1 GB at that size), `hidden` holds one (~0.4 GB) plus a
transient full-sequence qkv activation (~1.6 GB peak).

---

## 7. Cost model vs measurement

Predicted ratio of a cached step to a full step, in `hidden` mode, for `n` rows:

```
cached / full = 231.2M·n / (770M·n + 28672·n²)
              ≈ 12%  at n ≈ 38k
```

The `n²` term is attention, which nearly vanishes on cached steps because only the audio
rows issue queries.

Measured on an RTX 5090, 75,216 rows, 9.7 GB cache in RAM at `hidden`/`int4`:

| | steps | refinement pass | whole prompt |
|---|---|---|---|
| Turbo only, no refinement | 4 | — | 137.3s |
| + refinement, frozen cache | 4 + 6 | 44.6s (26.5s build, then 3.6s/step) | 184.6s |
| + refinement, cache bypassed | 4 + 6 | 124.0s (20.7s/step) | 263.4s |

3.6s / 20.7s ≈ **17.6%**, against a predicted 12%. The gap is cache streaming: ~9.7 GB
moved from RAM to VRAM per step, plus dequantization. The model accounts for compute only.

The build costs about **28% more** than a normal step (26.5s vs 20.7s) — capturing and
quantizing 50 blocks of state. Break-even is therefore around **1.3 refinement steps**, so
the cache wins from 2 steps upward; the deciding question is available memory, not step
count.

**The cache only removes compute.** On a machine where weight streaming or another
non-compute term dominates the step, it will save proportionally less. The `verbose` log
exists to show which regime you are in: `block loop` versus `outside blocks` per model
call.

---

## 8. Bugs worth knowing about

Both of these shipped, both passed a green test harness, and both are instructive about
*how* this code fails — silently, without errors, producing correct output.

**The gate never fired.** It required `audio_denoise_mask` to be absent, but core always
sets it on a two-stream mask ([§4](#4-activation-gating)). Every call fell through to the
stock path. Symptom: correct output, no speedup, no error.

The harness missed it because it passed a video-only mask with no `latent_shapes`, so
core's split never ran — it validated a call signature the sampler never produces. Worse,
the tell was visible and misread: every cached-step error reading was a clean `0.0000`,
which looks like excellent numerics and is actually the signature of a total no-op.

Now `T4` asserts the cached step *diverges* from stock, and `T8` asserts the gate fires on
a real two-stream mask.

**The cache rebuilt every step.** The slot key included `context.data_ptr()`, and core
reallocates the context tensor every model call ([§5](#5-cache-identity-and-invalidation)).
Every step got a fresh empty slot and a full rebuild — so every step paid full compute
*plus* quantization overhead, and the cache was never once read.

`T10` now reproduces the reallocation with `torch.cat([context])` and asserts one build
followed by cached steps.

The general lesson: this component's failure mode is being *inert*, not being wrong. Tests
that only check correctness will pass while it does nothing. Assert that work is being
skipped, not just that output is right.

---

## 9. Test harness

`test_frozen_cache.py` (not shipped in the pack) runs against real ComfyUI source on CPU
with a tiny randomly-initialised H3 model. Notes for anyone re-running it:

- `attention_head_dim` must be 128 — the rope table is a fixed 96 wide (3 axes × 16 inv
  freqs, doubled), so smaller head dims break.
- Init std must be ~0.12. At 0.02 the cross-modal coupling is near zero, and errors the
  tests are supposed to catch vanish into the noise.
- Needs `torch.no_grad()` and `model.requires_grad_(False)` — `ck`'s in-place rope is
  inference-only.

| test | what it pins down |
|---|---|
| T1 | codec round-trips within tolerance |
| T2 | build step is bit-identical to stock, all precisions, both contents modes |
| T3 | cached step with unchanged audio matches stock within cache precision |
| T4 | cached step with changed audio **diverges** — i.e. the cached path actually ran |
| T5 | `ram` and `disk` reproduce `vram` exactly |
| T6 | no mask / partial video mask pass through stock bit-for-bit |
| T7 | interrupted build leaves no poisoned cache; next run rebuilds cleanly |
| T8 | gate fires on a real two-stream mask; passes through when audio is frozen too |
| T9 | verbose accounting reports every block on both paths |
| T10 | reallocated context does not rebuild; changed conditioning does |
| T11 | disk is refused unless `allow_disk` is set, explicit and auto paths |
| T12 | `enabled=False` attaches nothing at all |

One unresolved item: a single run once failed with `nan` in the build-step comparison and
has not reproduced across 20 full-harness runs or a 40-init stress of the build path. It
may have been an artifact of editing files mid-run. If a refinement pass ever produces
garbage that a re-run fixes, that is the lead.

---

## 10. If you modify this

- **Widget order is positional in saved graphs.** Appending a new widget is safe; inserting
  one silently shifts every saved value after it. `enabled` was added first deliberately,
  accepting that users must re-add the node.
- **Read core before theorising.** Both shipped bugs were in assumptions about what core
  passes, and both were settled in minutes by reading `model_base.py` and `samplers.py`.
- **Assume the failure mode is silent.** Correct output proves nothing here.
- **Keep the build path a mirror of stock.** If you need different numerics, change the
  cached path — the build must stay reproducible against the unpatched model, because that
  is the only thing anchoring correctness.
