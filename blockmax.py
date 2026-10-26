"""BlockMax: a drop-in replacement for torch.optim.AdamW that keeps about half the optimizer state.

    from blockmax import BlockMax
    opt = BlockMax(params, lr=4e-3, weight_decay=0.1)

State per parameter tensor: one momentum tensor plus one scale per `block` coordinates (AdamW keeps a
full-size second-moment tensor instead). On CUDA with float32 parameters and Triton installed, each
tensor is updated by a single fused GPU pass; anywhere else it falls back to plain PyTorch with
identical results.

Learning rates are NOT interchangeable with AdamW's: the scale is a decaying block maximum, which is
larger than Adam's RMS, so BlockMax wants roughly 3-5x AdamW's learning rate. Tune it.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

try:  # optional fused path
    import triton
    import triton.language as tl

    @triton.jit
    def _blockmax_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps,
                         K: tl.constexpr, BB: tl.constexpr):
        blk = tl.program_id(0) * BB + tl.arange(0, BB)
        off = blk[:, None] * K + tl.arange(0, K)[None, :]
        mask = off < n
        bmask = blk < nb
        g = tl.load(g_ptr + off, mask=mask, other=0.0)
        m = tl.load(m_ptr + off, mask=mask, other=0.0)
        p = tl.load(p_ptr + off, mask=mask, other=0.0)
        v = tl.load(v_ptr + blk, mask=bmask, other=0.0)
        v = tl.maximum(v * b2, tl.max(tl.abs(g), axis=1))
        m = m + (1.0 - b1) * (g - m)
        p = p * wd_mult - lr * (c_g * g + c_m * m) / (v[:, None] + eps)
        tl.store(m_ptr + off, m, mask=mask)
        tl.store(p_ptr + off, p, mask=mask)
        tl.store(v_ptr + blk, v, mask=bmask)

    _HAS_TRITON = True
except Exception:  # pragma: no cover - Triton missing or unusable on this platform
    _HAS_TRITON = False


class BlockMax(Optimizer):
    """Nesterov-corrected momentum divided by a decaying block maximum of |grad|, with decoupled weight decay."""

    def __init__(self, params, lr=4e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, block=16, fused=None):
        if block < 1 or (block & (block - 1)):
            raise ValueError("block must be a power of two (the fused kernel requires it)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block=block, fused=fused))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd, K = group["lr"], group["betas"], group["eps"], group["weight_decay"], group["block"]
            want_fused = _HAS_TRITON if group["fused"] is None else group["fused"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("BlockMax does not support sparse gradients")
                st = self.state[p]
                n = p.numel()
                nb = (n + K - 1) // K
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros(nb, device=p.device, dtype=p.dtype)
                st["step"] += 1
                t = st["step"]
                c_g = (1 - b1) / (1 - b1 ** t)
                c_m = b1 / (1 - b1 ** (t + 1))
                m, v = st["m"], st["v"]
                if (want_fused and p.is_cuda and p.dtype == torch.float32 and g.dtype == torch.float32
                        and p.is_contiguous() and g.is_contiguous()):
                    BB = 64
                    _blockmax_kernel[((nb + BB - 1) // BB,)](p.data, g, m, v, n, nb, lr, 1.0 - lr * wd, b1, b2, c_g, c_m, eps, K=K, BB=BB)
                    continue
                gf = g.reshape(-1)
                pad = nb * K - n
                if pad:
                    gf = torch.nn.functional.pad(gf, (0, pad))
                torch.maximum(v.mul_(b2), gf.abs().view(nb, K).amax(1), out=v)
                m.lerp_(g.to(m.dtype), 1 - b1)
                num = g * c_g + m * c_m
                scale = v.repeat_interleave(K)[:n].view_as(p) + eps
                p.mul_(1 - lr * wd).sub_(num / scale, alpha=lr)
        return loss

    def state_bytes(self) -> int:
        """Bytes of optimizer state currently held."""
        return sum(t.numel() * t.element_size() for s in self.state.values() for t in s.values() if torch.is_tensor(t))
