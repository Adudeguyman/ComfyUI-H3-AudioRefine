"""Frozen-video KV cache for the MiniMax H3 audio refinement pass.

When the refinement pass runs with the video stream fully frozen (denoise mask
0 everywhere on video, audio fully generated), every non-audio row of the
packed sequence -- text, cond/ref rows, video -- receives constant inputs and a
constant timestep on every step. Their per-block attention K/V are therefore
identical across steps *except* for their attention back to the changing audio
tokens. This cache severs that video->audio edge: it computes the full sequence
once (build step), stores each block's post-rope K and V, and on subsequent
steps computes only the audio rows, attending against the cached K/V.

This is an approximation: cached context rows stop reacting to the audio
between rebuilds. It only activates when the model is called with a fully
frozen video mask and fully generated audio; all other calls (including normal
pass-1 sampling through the same patched model) pass through untouched.

Numerics reuse core's own kernels (comfy.quant_ops.ck.rms_rope_split_half_,
optimized_attention, comfy.ops.linear_input_act, _mod_scale_shift/_mod_gate
from comfy.ldm.minimax.model), so the build step reproduces the stock forward
exactly; only the orchestration lives here. Structural compatibility with core
is checked at patch time and per call, with loud fallback to the stock path.
"""

import logging
import os
import shutil
import tempfile
import threading
import time
import weakref

import numpy as np
import torch

import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy.ldm.minimax.model as mm_h3
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.patcher_extension import WrappersMP

log = logging.getLogger("H3-AudioRefine")

WRAPPER_KEY = "h3_frozen_video_cache"
_FP8 = getattr(torch, "float8_e4m3fn", None)
INT4_GROUP = 128


# ---------------------------------------------------------------------------
# codecs

class _CodecBF16:
    name = "bf16"

    def nbytes(self, s, d):
        return s * d * 2

    def quantize(self, t):
        return t.to(torch.bfloat16).contiguous(), None

    def dequantize(self, payload, scales, out, dtype):
        out.copy_(payload)
        return out

    def payload_shape(self, s, d):
        return (s, d), torch.bfloat16

    def scales_shape(self, s, d):
        return None, None


class _CodecFP8:
    name = "fp8"

    def nbytes(self, s, d):
        return s * d * 1 + s * 4

    def quantize(self, t):
        scales = t.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8) / 448.0
        return (t / scales).clamp(-448.0, 448.0).to(_FP8).contiguous(), scales.contiguous()

    def dequantize(self, payload, scales, out, dtype):
        tmp = payload.to(torch.float32)
        tmp.mul_(scales)
        out.copy_(tmp)
        return out

    def payload_shape(self, s, d):
        return (s, d), _FP8

    def scales_shape(self, s, d):
        return (s, 1), torch.float32


class _CodecINT4:
    name = "int4"

    def nbytes(self, s, d):
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        return s * d_pad // 2 + s * (d_pad // INT4_GROUP) * 2

    def quantize(self, t):
        s, d = t.shape
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        if d_pad != d:
            t = torch.nn.functional.pad(t, (0, d_pad - d))
        g = t.view(s, d_pad // INT4_GROUP, INT4_GROUP).float()
        scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.round(g / scales).clamp(-7, 7).to(torch.int8) + 7  # [0, 14]
        q = q.view(s, d_pad).to(torch.uint8)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
        return packed, scales.view(s, d_pad // INT4_GROUP).to(torch.float16).contiguous()

    def dequantize(self, payload, scales, out, dtype):
        s, half = payload.shape
        d_pad = half * 2
        q = torch.empty((s, d_pad), dtype=torch.uint8, device=payload.device)
        q[:, 0::2] = payload & 0xF
        q[:, 1::2] = payload >> 4
        v = (q.to(torch.float32) - 7.0).view(s, d_pad // INT4_GROUP, INT4_GROUP)
        v = v * scales.to(torch.float32).unsqueeze(-1)
        out.copy_(v.view(s, d_pad)[:, : out.shape[1]])
        return out

    def payload_shape(self, s, d):
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        return (s, d_pad // 2), torch.uint8

    def scales_shape(self, s, d):
        d_pad = -(-d // INT4_GROUP) * INT4_GROUP
        return (s, d_pad // INT4_GROUP), torch.float16


CODECS = {"bf16": _CodecBF16(), "fp8": _CodecFP8(), "int4": _CodecINT4()}


# ---------------------------------------------------------------------------
# storage backends

def _rss_bytes():
    """Resident set size of this process -- what the cache actually cost, as opposed to
    what MemAvailable predicted it would."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _gb(n):
    return "n/a" if n is None else "%.1f GB" % (n / 2**30)


def _meminfo_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


class _StoreVRAM:
    name = "vram"

    def __init__(self, n_blocks, codec, s, d, device, pair=True):
        self.k = [None] * n_blocks
        self.v = [None] * n_blocks

    def put(self, i, k_q, k_s, v_q, v_s):
        self.k[i] = (k_q, k_s)
        self.v[i] = (v_q, v_s)

    def get(self, i, device):
        v = self.v[i] if self.v[i] is not None else (None, None)
        return self.k[i] + v

    def begin_step(self, order):
        pass

    def end_step(self):
        pass

    def free(self):
        self.k = self.v = None


class _StoreRAM:
    name = "ram"

    def __init__(self, n_blocks, codec, s, d, device, pair=True):
        self.k = [None] * n_blocks
        self.v = [None] * n_blocks

    _pin_failed = False

    @classmethod
    def _pin(cls, t):
        if t is None:
            return None
        pin = torch.cuda.is_available() and not cls._pin_failed
        try:
            c = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin)
        except RuntimeError:
            if not cls._pin_failed:
                cls._pin_failed = True
                log.warning("H3 Frozen Video Cache: could not pin cache memory (RAM pressure); "
                            "falling back to pageable memory. Cached steps will be slightly "
                            "slower but the cache can be swapped instead of killing the process.")
            c = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=False)
        c.copy_(t)
        return c

    def put(self, i, k_q, k_s, v_q, v_s):
        self.k[i] = (self._pin(k_q), self._pin(k_s))
        self.v[i] = (self._pin(v_q), self._pin(v_s))

    def get(self, i, device):
        k_q, k_s = self.k[i]
        v_q, v_s = self.v[i] if self.v[i] is not None else (None, None)
        to = lambda t: None if t is None else t.to(device, non_blocking=True)
        return to(k_q), to(k_s), to(v_q), to(v_s)

    def begin_step(self, order):
        pass

    def end_step(self):
        pass

    def free(self):
        self.k = self.v = None


class _StoreDisk:
    """Memory-mapped on-disk cache with a one-block-ahead reader thread."""

    name = "disk"

    def __init__(self, n_blocks, codec, s, d, device, pair=True):
        self.n_blocks = n_blocks
        base = _disk_base_dir()
        _sweep_stale_dirs(base)
        self.dir = tempfile.mkdtemp(prefix="h3_frozen_cache_", dir=base)
        _LIVE_DISK_DIRS.add(self.dir)
        self._fin = weakref.finalize(self, _cleanup_disk_dir, self.dir)
        p_shape, p_dtype = codec.payload_shape(s, d)
        s_shape, s_dtype = codec.scales_shape(s, d)
        self.shapes = (p_shape, p_dtype, s_shape, s_dtype)
        self.mm = {}
        v_shape, vs_shape = (p_shape, s_shape) if pair else (None, None)
        for kind, shape, dtype in (("k", p_shape, p_dtype), ("v", v_shape, p_dtype),
                                   ("ks", s_shape, s_dtype), ("vs", vs_shape, s_dtype)):
            if shape is None:
                self.mm[kind] = None
                continue
            npdt = np.uint8 if dtype in (torch.uint8, _FP8) else (np.float16 if dtype == torch.float16 else
                                                                  np.float32 if dtype == torch.float32 else np.uint16)
            path = os.path.join(self.dir, kind + ".bin")
            self.mm[kind] = np.lib.format.open_memmap(path, mode="w+", dtype=npdt,
                                                      shape=(n_blocks,) + tuple(shape))
        self._buffers = None
        self._thread = None
        self._stop = False

    def _torch_view(self, kind, i):
        arr = self.mm[kind][i]
        t = torch.from_numpy(np.ascontiguousarray(arr))
        _, p_dtype, _, s_dtype = self.shapes
        want = p_dtype if kind in ("k", "v") else s_dtype
        if want == _FP8:
            t = t.view(_FP8)
        elif want == torch.bfloat16:
            t = t.view(torch.bfloat16)
        elif want == torch.float16 and t.dtype != torch.float16:
            t = t.view(torch.float16)
        return t

    def put(self, i, k_q, k_s, v_q, v_s):
        def w(kind, t):
            if t is None:
                return
            a = t.detach().cpu()
            if a.dtype == _FP8:
                a = a.view(torch.uint8)
            elif a.dtype == torch.bfloat16:
                a = a.view(torch.uint16)
            self.mm[kind][i] = a.numpy()
        w("k", k_q); w("ks", k_s); w("v", v_q); w("vs", v_s)

    def begin_step(self, order):
        # double-buffered pinned staging + reader thread one block ahead
        pin = torch.cuda.is_available()
        if self._buffers is None:
            mk = lambda kind: (None if self.mm[kind] is None else
                               [torch.empty(self.mm[kind].shape[1:],
                                            dtype=self._torch_view(kind, 0).dtype,
                                            device="cpu", pin_memory=pin) for _ in range(2)])
            self._buffers = {kind: mk(kind) for kind in ("k", "ks", "v", "vs")}
            self._ready = [threading.Event(), threading.Event()]
            self._consumed = [threading.Event(), threading.Event()]
        for e in self._ready:
            e.clear()
        for e in self._consumed:
            e.set()
        self._stop = False
        self._order = list(order)

        def reader():
            for pos, i in enumerate(self._order):
                slot = pos % 2
                self._consumed[slot].wait()
                if self._stop:
                    return
                self._consumed[slot].clear()
                for kind in ("k", "ks", "v", "vs"):
                    if self.mm[kind] is not None:
                        self._buffers[kind][slot].copy_(self._torch_view(kind, i))
                self._ready[slot].set()

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()
        self._pos = 0

    def get(self, i, device):
        slot = self._pos % 2
        self._ready[slot].wait()
        self._ready[slot].clear()
        out = []
        for kind in ("k", "ks", "v", "vs"):
            b = self._buffers[kind]
            if b is None:
                out.append(None)
                continue
            t = b[slot].to(device, non_blocking=False)
            if t.data_ptr() == b[slot].data_ptr():
                t = t.clone()  # .to() aliased the staging buffer; reader will overwrite it
            out.append(t)
        self._consumed[slot].set()
        self._pos += 1
        return out[0], out[1], out[2], out[3]

    def end_step(self):
        self._stop = True
        for e in self._consumed:
            e.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def free(self):
        self.end_step()
        self.mm = {}
        self._fin()


_LIVE_DISK_DIRS = set()


def _cleanup_disk_dir(d):
    _LIVE_DISK_DIRS.discard(d)
    shutil.rmtree(d, ignore_errors=True)


def _sweep_stale_dirs(base):
    """Remove cache dirs orphaned by previous processes or dropped states."""
    try:
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if name.startswith("h3_frozen_cache_") and p not in _LIVE_DISK_DIRS:
                shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _disk_base_dir():
    try:
        import folder_paths
        d = folder_paths.get_temp_directory()
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return tempfile.gettempdir()


STORES = {"vram": _StoreVRAM, "ram": _StoreRAM, "disk": _StoreDisk}


# ---------------------------------------------------------------------------
# cache state

class _Slot:
    def __init__(self):
        self.store = None
        self.codec = None
        self.layout_sig = None
        self.video_fp = None
        self.context_fp = None
        self.last_sigma = None
        self.steps_since_build = 0
        self.dq_k = None
        self.dq_v = None
        self.dq_h = None
        self.seq_len = 0

    def free(self):
        if self.store is not None:
            backend = getattr(self.store, "name", "?")
            rss0 = _rss_bytes()
            self.store.free()
            self.dq_k = self.dq_v = self.dq_h = None
            rss1 = _rss_bytes()
            if rss0 is not None and rss1 is not None:
                returned = rss0 - rss1
                note = ""
                if backend == "ram" and returned < 0.5 * (rss0 - rss1 + 1):
                    note = (" -- pinned host memory is held by PyTorch's caching host "
                            "allocator and is not returned to the OS; it is reused by the "
                            "next build rather than released")
                log.info("H3 Frozen Video Cache: cache freed (%s) | process RSS %s -> %s "
                         "(returned %s)%s", backend, _gb(rss0), _gb(rss1), _gb(max(returned, 0)), note)
        self.store = None
        self.dq_k = self.dq_v = self.dq_h = None


class _State:
    def __init__(self, backend, precision, refresh_interval, contents="kv", verbose=False,
                 allow_disk=False):
        self.verbose = verbose
        self.allow_disk = allow_disk
        self.backend = backend
        self.precision = precision
        self.refresh_interval = refresh_interval
        self.contents = contents
        self.slots = {}       # key -> _Slot (max 2, insertion-ordered LRU)
        # per-call, set by the wrapper before the block loop runs:
        self.mode = "off"     # off | build | cached
        self.slot = None
        self.aa = self.ab = 0
        self.step_ctx = {}
        self.warned = set()
        self.rss_before = None
        self.avail_before = None
        self.est_bytes = 0
        self.n_cached = 0     # blocks that took the cached path this call
        self.n_built = 0      # blocks that took the (full) build path this call
        self.n_stock = 0      # blocks that fell through to the stock path this call
        self.t_blocks = 0.0   # wall time spent inside patched blocks this call

    def warn_once(self, tag, msg):
        if tag not in self.warned:
            self.warned.add(tag)
            log.warning("H3 Frozen Video Cache: %s", msg)

    def get_slot(self, key):
        if key in self.slots:
            slot = self.slots.pop(key)
            self.slots[key] = slot
            return slot
        slot = _Slot()
        self.slots[key] = slot
        while len(self.slots) > 2:
            _, old = next(iter(self.slots.items()))
            oldk = next(iter(self.slots))
            self.slots.pop(oldk)
            old.free()
        return slot


def _tensor_fingerprint(t):
    """Cheap content fingerprint. Used instead of data_ptr(): core reallocates conds
    every model call (samplers.cond_cat -> torch.cat), so pointer identity is unstable
    and keying on it forces a rebuild on every single step."""
    v = t.detach().float()
    return (float(v.sum()), float(v.abs().sum()))


def _fp_matches(a, b):
    if a is None or b is None:
        return False
    for x, y in zip(a, b):
        if abs(x - y) > 1e-3 * (abs(x) + abs(y) + 1e-6):
            return False
    return True


_DISK_HELP = (
    "The disk backend is off by default because it writes the whole cache to your drive "
    "on every build -- gigabytes per run, which is real SSD wear. Enable the 'allow_disk' "
    "toggle on the H3 Frozen Video Cache node if you accept that, or reduce the cache size "
    "(cache_contents=hidden, precision=int4) so it fits in RAM or VRAM, or bypass the node "
    "to run the refinement pass without a cache at all."
)


def _resolve_backend(requested, total_bytes, device, allow_disk=False):
    reasons = []
    if requested != "auto":
        if requested == "disk" and not allow_disk:
            raise RuntimeError("H3 Frozen Video Cache: backend is set to 'disk' but disk "
                               "writing is not enabled. " + _DISK_HELP)
        return requested, "requested"
    try:
        free_vram = comfy.model_management.get_free_memory(device)
    except Exception:
        free_vram = 0
    if total_bytes + 4 * (1 << 30) < free_vram:
        return "vram", "fits in free VRAM (%.1f GB free)" % (free_vram / 2**30)
    reasons.append("VRAM: need %.1f GB + 4 GB margin, %.1f GB free" % (total_bytes / 2**30, free_vram / 2**30))
    ram = _meminfo_available_bytes()
    if ram is not None and total_bytes + 4 * (1 << 30) < ram:
        return "ram", "fits in available system RAM (%.1f GB available)" % (ram / 2**30)
    reasons.append("RAM: %.1f GB available" % ((ram or 0) / 2**30))
    if not allow_disk:
        raise RuntimeError(
            "H3 Frozen Video Cache: the cache needs %.1f GB and does not fit in VRAM or "
            "RAM (%s), and disk writing is not enabled. %s"
            % (total_bytes / 2**30, "; ".join(reasons), _DISK_HELP))
    return "disk", "; ".join(reasons)


# ---------------------------------------------------------------------------
# patched block math (uses core's own kernels; orchestration only)

def _attn_qkv_rope(attn, h, rope_slice):
    """QKV + fused per-head RMSNorm + partial split-half rope, exactly as core's Attention.forward."""
    s = h.shape[0]
    q, k, v = attn.qkv_proj(h).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    q = q.view(1, s, attn.heads, attn.head_dim)
    k = k.view(1, s, attn.heads, attn.head_dim)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=h.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=h.device)
    rot = rope_slice.shape[-3] * 2
    if comfy.model_management.in_training:
        q, k = comfy.quant_ops.ck.rms_rope_split_half(
            q, k, rope_slice, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    else:
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_slice, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    return q[0], k[0], v  # [s, heads, head_dim] each


def _block_build(state, blk, args):
    """Full-sequence block forward, numerically identical to core, capturing post-rope K and V."""
    x = args["img"]
    t_emb = args["t_emb"]
    mod_segments = args["mod_segments"]
    rope_freqs = args["rope_freqs"]
    topt = args["transformer_options"]
    i = state.step_ctx["block_index"]

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(t_emb)
    h = mm_h3._mod_scale_shift(blk.norm1(x), shift_msa, scale_msa, mod_segments)

    codec = state.slot.codec
    if state.contents == "hidden":
        h_q, h_s = codec.quantize(h)
        state.slot.store.put(i, h_q, h_s, None, None)
    q, k, v = _attn_qkv_rope(blk.attn, h, rope_freqs)
    s = h.shape[0]
    if state.contents == "kv":
        k_q, k_s = codec.quantize(k.reshape(s, -1))
        v_q, v_s = codec.quantize(v.reshape(s, -1))
        state.slot.store.put(i, k_q, k_s, v_q, v_s)

    vq = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    vk = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    vv = AttentionTensorContainer(v.clone().transpose(0, 1).unsqueeze(0))
    out = optimized_attention(vq, vk, vv, blk.attn.heads, mask=None, skip_reshape=True,
                              transformer_options=topt)
    x = mm_h3._mod_gate(x, gate_msa, blk.attn.out_proj(out.squeeze(0)), mod_segments)
    h = mm_h3._mod_scale_shift(blk.norm2(x), shift_mlp, scale_mlp, mod_segments)
    x = mm_h3._mod_gate(x, gate_mlp, blk.mlp(h), mod_segments)

    state.step_ctx["block_index"] = i + 1
    return {"img": x}


def _block_cached(state, blk, args):
    """Audio-rows-only block forward against cached K/V. Non-audio rows of x are left untouched."""
    x = args["img"]
    t_emb = args["t_emb"]
    rope_freqs = args["rope_freqs"]
    topt = args["transformer_options"]
    aa, ab = state.aa, state.ab
    row = state.step_ctx["audio_mod_row"]
    seg = [(0, ab - aa, row)]
    i = state.step_ctx["block_index"]
    slot = state.slot

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(t_emb)

    xa = x[aa:ab]
    heads, hd = blk.attn.heads, blk.attn.head_dim
    codec = slot.codec
    if state.contents == "hidden":
        # rebuild K/V on the fly from the stored post-norm hidden states
        h_q, h_s, _, _ = slot.store.get(i, x.device)
        codec.dequantize(h_q, h_s, slot.dq_h, x.dtype)
        h_a = mm_h3._mod_scale_shift(blk.norm1(xa), shift_msa, scale_msa, seg)
        slot.dq_h[aa:ab] = h_a.to(slot.dq_h.dtype)
        q_full, kf, vf = _attn_qkv_rope(blk.attn, slot.dq_h, rope_freqs)
        q = q_full[aa:ab]
        vf = vf.clone()
    else:
        h = mm_h3._mod_scale_shift(blk.norm1(xa), shift_msa, scale_msa, seg)
        q, k_live, v_live = _attn_qkv_rope(blk.attn, h, rope_freqs[:, aa:ab])

        # cached K/V for the whole sequence, live audio slice patched in
        k_q, k_s, v_q, v_s = slot.store.get(i, x.device)
        codec.dequantize(k_q, k_s, slot.dq_k, x.dtype)
        codec.dequantize(v_q, v_s, slot.dq_v, x.dtype)
        kf = slot.dq_k.view(slot.seq_len, heads, hd)
        vf = slot.dq_v.view(slot.seq_len, heads, hd)
        kf[aa:ab] = k_live.to(kf.dtype)
        vf[aa:ab] = v_live.to(vf.dtype)

    cq = AttentionTensorContainer(q.to(kf.dtype).transpose(0, 1).unsqueeze(0))
    ck_ = AttentionTensorContainer(kf.transpose(0, 1).unsqueeze(0))
    cv = AttentionTensorContainer(vf.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(cq, ck_, cv, heads, mask=None, skip_reshape=True,
                              transformer_options=topt)
    mm_h3._mod_gate(xa, gate_msa, blk.attn.out_proj(out.squeeze(0)).to(xa.dtype), seg)
    h = mm_h3._mod_scale_shift(blk.norm2(xa), shift_mlp, scale_mlp, seg)
    mm_h3._mod_gate(xa, gate_mlp, blk.mlp(h).to(xa.dtype), seg)

    state.step_ctx["block_index"] = i + 1
    return {"img": x}


def _sync_if_cuda(t):
    """CUDA is async; without a sync the block timings are meaningless."""
    if t.device.type == "cuda":
        torch.cuda.synchronize(t.device)


def _find_audio_mod_row(mod_segments, aa, ab):
    for a, b, row in mod_segments:
        if a == aa and b == ab and isinstance(row, int):
            return row
    return None


def make_block_replace(state, get_block, n_blocks):
    def block_replace(args, extra):
        if state.mode == "off":
            state.n_stock += 1
            return extra["original_block"](args)
        i = state.step_ctx.get("block_index", 0)
        if i == 0:
            # first block of the step: final validation against live per-step data
            row = _find_audio_mod_row(args["mod_segments"], state.aa, state.ab)
            seq_len = args["img"].shape[0]
            if row is None or (state.mode == "cached" and seq_len != state.slot.seq_len):
                state.warn_once("segments", "unexpected packed layout at runtime; "
                                            "falling back to the exact (uncached) path for this pass.")
                if state.mode == "build":
                    state.slot.free()  # never leave a half-built slot behind
                state.mode = "off"
                return extra["original_block"](args)
            state.step_ctx["audio_mod_row"] = row
            state.step_ctx["block_index"] = 0
            if state.mode == "cached":
                if state.contents == "hidden":
                    if state.slot.dq_h is None:
                        state.slot.dq_h = torch.empty((state.slot.seq_len, state.slot.store_d),
                                                      dtype=args["img"].dtype, device=args["img"].device)
                elif state.slot.dq_k is None:
                    d = state.slot.store_d
                    state.slot.dq_k = torch.empty((state.slot.seq_len, d), dtype=args["img"].dtype,
                                                  device=args["img"].device)
                    state.slot.dq_v = torch.empty_like(state.slot.dq_k)
                state.slot.store.begin_step(range(n_blocks))
        blk = get_block(i)
        if not state.verbose:
            if state.mode == "build":
                return _block_build(state, blk, args)
            return _block_cached(state, blk, args)
        t0 = time.perf_counter()
        try:
            if state.mode == "build":
                state.n_built += 1
                return _block_build(state, blk, args)
            state.n_cached += 1
            return _block_cached(state, blk, args)
        finally:
            _sync_if_cuda(args["img"])
            state.t_blocks += time.perf_counter() - t0

    return block_replace


def make_wrapper(state, n_blocks, d_kv, d_hidden):
    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        state.mode = "off"
        state.step_ctx = {}
        state.n_cached = state.n_built = state.n_stock = 0
        state.t_blocks = 0.0
        t_call = time.perf_counter()
        try:
            payload = kwargs.get("minimax_payload") or {}
            layout = payload.get("layout")
            denoise_mask = kwargs.get("denoise_mask")
            audio_mask = kwargs.get("audio_denoise_mask")
            activate = (
                layout is not None
                and denoise_mask is not None
                and isinstance(x, (list, tuple)) and len(x) == 2
                and float(denoise_mask.max()) < 1e-3
                # audio must actually be generating; if it is also frozen there is
                # nothing to refine and the stock path is the honest answer
                and (audio_mask is None or float(audio_mask.max()) > 1e-3)
            )
            if activate:
                seg = {s[2]: (s[0], s[1]) for s in layout.segments}
                if "audio" not in seg or "video" not in seg:
                    activate = False
            if not activate:
                return executor(x, timestep, context, transformer_options, **kwargs)

            state.aa, state.ab = seg["audio"]
            sig = getattr(layout, "signature", None)
            # NB: no data_ptr() here -- core allocates a fresh context tensor per call.
            key = (sig, tuple(context.shape))
            slot = state.get_slot(key)
            video_fp = _tensor_fingerprint(x[0])
            context_fp = _tensor_fingerprint(context)
            sigma = float(timestep.flatten()[0])

            new_step = slot.last_sigma is None or abs(sigma - slot.last_sigma) > 1e-9
            slot.last_sigma = sigma

            if slot.store is None:
                reason = "no cache yet"
            elif slot.layout_sig != sig:
                reason = "layout changed"
            elif not _fp_matches(slot.video_fp, video_fp):
                reason = "video latent changed (new seed or new pass-1 result)"
            elif not _fp_matches(slot.context_fp, context_fp):
                reason = "conditioning changed"
            elif state.refresh_interval > 0 and slot.steps_since_build >= state.refresh_interval:
                reason = "refresh_interval reached"
            else:
                reason = None
            need_build = reason is not None
            seq_len = layout.seq_len

            if need_build:
                if slot.store is not None:
                    slot.free()
                codec = CODECS[state.precision]
                pair = state.contents == "kv"
                d_store = d_kv if pair else d_hidden
                total = codec.nbytes(seq_len, d_store) * (2 if pair else 1) * n_blocks
                backend, why = _resolve_backend(state.backend, total, x[0].device,
                                                allow_disk=state.allow_disk)
                log.info("H3 Frozen Video Cache: building cache (%s) | rows=%d contents=%s "
                         "dim=%d blocks=%d precision=%s size=%.1f GB backend=%s (%s)",
                         reason, seq_len, state.contents, d_store, n_blocks, state.precision,
                         total / 2**30, backend, why)
                if backend == "disk":
                    log.warning("H3 Frozen Video Cache: writing %.1f GB to disk for this cache "
                                "build. Repeated runs cause real SSD wear -- see the disk wear "
                                "note in the README.", total / 2**30)
                slot.codec = codec
                state.rss_before = _rss_bytes()
                state.avail_before = _meminfo_available_bytes()
                state.est_bytes = total
                slot.store = STORES[backend](n_blocks, codec, seq_len, d_store, x[0].device, pair=pair)
                slot.seq_len = seq_len
                slot.store_d = d_store
                slot.steps_since_build = 0
                state.mode = "build"
            else:
                if new_step:
                    slot.steps_since_build += 1
                state.mode = "cached"

            state.slot = slot
            state.step_ctx = {"block_index": 0}
            try:
                ret = executor(x, timestep, context, transformer_options, **kwargs)
            except BaseException:
                if state.mode == "build":
                    slot.free()  # interrupted build: never leave a half-written cache marked valid
                raise
            finally:
                if state.mode == "cached" and slot.store is not None:
                    slot.store.end_step()
            if state.mode == "build":
                # stamp validity only after the build completed every block
                slot.layout_sig = sig
                slot.video_fp = video_fp
                slot.context_fp = context_fp
                rss_after = _rss_bytes()
                avail_after = _meminfo_available_bytes()
                d_rss = None if (rss_after is None or state.rss_before is None) else rss_after - state.rss_before
                d_av = None if (avail_after is None or state.avail_before is None) else state.avail_before - avail_after
                log.info("H3 Frozen Video Cache: cache built | estimated %s | process RSS %s -> %s "
                         "(+%s) | MemAvailable %s -> %s (-%s)",
                         _gb(state.est_bytes), _gb(state.rss_before), _gb(rss_after), _gb(d_rss),
                         _gb(state.avail_before), _gb(avail_after), _gb(d_av))
            if state.verbose:
                _sync_if_cuda(x[0])
                total = time.perf_counter() - t_call
                if state.n_cached + state.n_built + state.n_stock == 0:
                    log.warning("H3 Frozen Video Cache: the block replacement never ran this "
                                "call (0 of %d blocks) -- this build of ComfyUI is not "
                                "dispatching patches_replace['dit'][('double_block', i)], so "
                                "the cache cannot take effect. Model call took %.2fs.",
                                n_blocks, total)
                log.info("H3 Frozen Video Cache: %s step | blocks: %d cached, %d built, "
                         "%d stock (of %d) | block loop %.2fs | whole model call %.2fs | "
                         "outside blocks %.2fs",
                         state.mode, state.n_cached, state.n_built, state.n_stock, n_blocks,
                         state.t_blocks, total, total - state.t_blocks)
            return ret
        finally:
            state.mode = "off"

    return wrapper


def check_core_compat(dm):
    problems = []
    for name in ("_mod_scale_shift", "_mod_gate"):
        if not hasattr(mm_h3, name):
            problems.append("comfy.ldm.minimax.model.%s missing" % name)
    if not hasattr(comfy.quant_ops, "ck") or not hasattr(comfy.quant_ops.ck, "rms_rope_split_half_"):
        problems.append("comfy.quant_ops.ck.rms_rope_split_half_ missing")
    blocks = getattr(dm, "blocks", None)
    if blocks is None or len(blocks) == 0:
        problems.append("diffusion_model.blocks missing")
    else:
        blk = blocks[0]
        for name in ("adaln_proj", "norm1", "norm2", "attn", "mlp"):
            if not hasattr(blk, name):
                problems.append("block.%s missing" % name)
        attn = getattr(blk, "attn", None)
        if attn is not None:
            for name in ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads", "head_dim"):
                if not hasattr(attn, name):
                    problems.append("block.attn.%s missing" % name)
    if problems:
        raise RuntimeError(
            "H3 Frozen Video Cache: this ComfyUI build's MiniMax H3 internals do not match "
            "what the cache was written against (%s). Core has probably changed; the cache "
            "node needs an update. Bypass the node to keep working." % "; ".join(problems))


def patch_model(model, backend, precision, refresh_interval, cache_contents="kv", verbose=False,
                allow_disk=False):
    if precision == "fp8" and _FP8 is None:
        raise RuntimeError("H3 Frozen Video Cache: this PyTorch build has no float8_e4m3fn; "
                           "use bf16 or int4 precision instead.")
    dm = model.get_model_object("diffusion_model")
    check_core_compat(dm)
    n_blocks = len(dm.blocks)
    d_kv = dm.blocks[0].attn.heads * dm.blocks[0].attn.head_dim
    d_hidden = getattr(dm, "hidden_size", None)
    if d_hidden is None:
        d_hidden = dm.blocks[0].norm1.weight.shape[0]

    m = model.clone()
    state = _State(backend, precision, refresh_interval, cache_contents, verbose=verbose,
                   allow_disk=allow_disk)
    replace = make_block_replace(state, lambda i: dm.blocks[i], n_blocks)
    for i in range(n_blocks):
        m.set_model_patch_replace(replace, "dit", "double_block", i)
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                           make_wrapper(state, n_blocks, d_kv, d_hidden))
    return m
