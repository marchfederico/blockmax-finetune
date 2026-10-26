# IDEA: FLASH-BLOCKMAX. FlashOptim's weight storage with Packed BlockMax's update rule. Each weight is a bf16 value plus an int8
# correction (FlashOptim's format: the rounding error as a fraction of half a bf16 ulp, times 127), so it carries about 24 bits;
# the model computes in bf16 and its gradients are bf16. The update is BlockMax: one decaying max-|g| scale per block (block size
# scheduled, as in packedsched_triton.py) and an int8 momentum with the packed kernel's sign + quarter-octave code.
# Memory per weight: 2 (bf16) + 1 (correction) + 1 (momentum) bytes + 4/K for the scales; gradients bf16.
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "flashbm_triton"
DEFAULT_LR = 1e-5
SCHEDULE = [(0, 16)]  # (first step, K), as in packedsched_triton.py
INHERIT = True
BB = 64


@triton.jit
def _half_ulp_log2(x_bf):
    """log2 of half the gap between x (a bf16 value, given as fp32) and the next bf16: 2^(exponent - 7 - 1)."""
    bits = x_bf.to(tl.int32, bitcast=True)
    e = (bits >> 23) & 255
    e = tl.where(e == 0, 1, e)                                       # zero and subnormals share the smallest exponent
    return (e - 127 - 8).to(tl.float32)


@triton.jit
def _flashbm_kernel(p_ptr, c_ptr, g_ptr, m_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps,
                    K: tl.constexpr, BB: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
    pb = tl.load(p_ptr + off, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + off, mask=mask, other=0).to(tl.float32)
    q = tl.load(m_ptr + off, mask=mask, other=0).to(tl.int32)
    v = tl.load(v_ptr + blk, mask=bmask, other=0.0)

    # 24-bit weight = bf16 + correction (FlashOptim's reconstruction; the exponent is split in two to stay finite)
    ls = _half_ulp_log2(pb)
    h = tl.floor(ls / 2.0)
    w = pb + (c / 127.0) * tl.exp2(h) * tl.exp2(ls - h)

    v = tl.maximum(v * b2, tl.max(tl.abs(g), axis=1))
    qf = q.to(tl.float32)
    mag = tl.exp2((tl.abs(qf) - 101.0) * 0.25)
    m = tl.where(q > 0, mag, tl.where(q < 0, -mag, 0.0))            # code 0 means exactly zero
    m = m + (1.0 - b1) * (g - m)
    idx = tl.floor(tl.log2(tl.maximum(tl.abs(m), 1e-30)) * 4.0 + 0.5) + 101.0
    idx = tl.minimum(tl.maximum(idx, 1.0), 127.0)
    sgn = tl.where(m > 0, 1.0, tl.where(m < 0, -1.0, 0.0))
    code = (idx * sgn).to(tl.int8)

    w = w * wd_mult - lr * (c_g * g + c_m * m) / (v[:, None] + eps)

    # split back: bf16 rounded to nearest, and the rounding error in units of half a bf16 ulp, times 127 (FlashOptim's encoding)
    nb16 = w.to(tl.bfloat16)
    nf = nb16.to(tl.float32)
    ls2 = _half_ulp_log2(nf)
    h2 = tl.floor(-ls2 / 2.0)
    en = (w - nf) * tl.exp2(h2) * tl.exp2(-ls2 - h2)
    en = tl.minimum(tl.maximum(en, -1.0), 1.0) * 127.0
    cn = tl.where(en >= 0, tl.floor(en + 0.5), -tl.floor(-en + 0.5)).to(tl.int8)

    tl.store(p_ptr + off, nb16, mask=mask)
    tl.store(c_ptr + off, cn, mask=mask)
    tl.store(m_ptr + off, code, mask=mask)
    tl.store(v_ptr + blk, v, mask=bmask)


def _k_at(t):
    k = SCHEDULE[0][1]
    for start, kk in SCHEDULE:
        if t > start:
            k = kk
    return k


def _rescale(v, n, k_old, k_new):
    nb_new = (n + k_new - 1) // k_new
    if k_new < k_old:                      # split: every child inherits its parent's max
        return v.repeat_interleave(k_old // k_new)[:nb_new].contiguous()
    f = k_new // k_old                     # merge: max over the children
    pad = nb_new * f - v.numel()
    return torch.nn.functional.pad(v, (0, pad)).view(nb_new, f).amax(1).contiguous()


def full_precision(p, state):
    """The ~24-bit weight (bf16 + correction) as fp32, e.g. to save a checkpoint at full precision."""
    c = state.get("c") if state else None
    x = p.detach().float()
    if c is None:
        return x
    e = ((x.view(torch.int32) >> 23) & 255).clamp_min(1) - 127 - 8
    return x + c.float() / 127.0 * torch.exp2(e.float())


class CandidateOptimizer(Optimizer):
    def __init__(self, params, lr=DEFAULT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, **kwargs):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, group)

    @torch.no_grad()
    def step_param(self, p):
        """Update one parameter now (streamed updates: called from a post-accumulate-grad hook, then p.grad is freed)."""
        for group in self.param_groups:
            if any(q is p for q in group["params"]):
                self._update(p, group)
                return

    def _update(self, p, group):
        lr = group["lr"]
        b1, b2 = group["betas"]
        assert p.dtype == torch.bfloat16 and p.is_contiguous(), "flashbm needs contiguous bf16 weights"
        st = self.state[p]
        n = p.numel()
        if "v" not in st:
            st["step"] = 0
            st["K"] = SCHEDULE[0][1]
            st["v"] = torch.zeros((n + st["K"] - 1) // st["K"], device=p.device, dtype=torch.float32)
            st["m"] = torch.zeros(p.shape, device=p.device, dtype=torch.int8)
            st["c"] = torch.zeros(p.shape, device=p.device, dtype=torch.int8)   # bf16 weights start exact
        st["step"] += 1
        t = st["step"]
        k = _k_at(t)
        if k != st["K"]:
            st["v"] = _rescale(st["v"], n, st["K"], k) if INHERIT else torch.zeros((n + k - 1) // k, device=p.device, dtype=torch.float32)
            st["K"] = k
        nb = (n + k - 1) // k
        bb = max(1, min(BB, 4096 // k))    # at most 4096 weights per program (larger tiles spill registers)
        _flashbm_kernel[((nb + bb - 1) // bb,)](
            p.data, st["c"], p.grad, st["m"], st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b1, b2,
            (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), group["eps"], K=k, BB=bb,
            num_warps=16 if k >= 4096 else 4)
