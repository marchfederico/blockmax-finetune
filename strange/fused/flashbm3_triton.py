# IDEA: FLASH-BLOCKMAX IN 3 BYTES PER WEIGHT, optimizer state included. Each weight is a bf16 value plus ONE companion byte:
#   low 4 bits  = weight correction: the bf16 rounding error in units of 1/7 of half a bf16 ulp (-7..7), so ~20-bit weights
#                 (FlashOptim's correction idea at half the width); written with STOCHASTIC rounding, so steps finer than the
#                 grid still accumulate on average
#   high 4 bits = momentum, stored RELATIVE TO ITS BLOCK'S SCALE v: r = m / v, sign + a 3-bit octave level (0, 1/32 ... 2),
#                 also stochastically rounded (unbiased); |m| <= ~1.95 v always holds for betas (0.9, 0.95)
# The update is BlockMax's: one decaying max-|g| scale per block of 16 (4 bytes / 16 weights). Gradients are bf16.
# step_param(p) updates one tensor at once, for streamed updates from a post-accumulate-grad hook.
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "flashbm3_triton"
DEFAULT_LR = 1e-5
SCHEDULE = [(0, 16)]
INHERIT = True
BB = 64


@triton.jit
def _half_ulp_log2(x_bf):
    bits = x_bf.to(tl.int32, bitcast=True)
    e = (bits >> 23) & 255
    e = tl.where(e == 0, 1, e)
    return (e - 127 - 8).to(tl.float32)


@triton.jit
def _fbm3_kernel(p_ptr, s_ptr, g_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps, seed,
                 K: tl.constexpr, BB: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
    pb = tl.load(p_ptr + off, mask=mask, other=0.0).to(tl.float32)
    byte = tl.load(s_ptr + off, mask=mask, other=0).to(tl.int32) & 255
    v_old = tl.load(v_ptr + blk, mask=bmask, other=0.0)

    qc = byte & 15
    qc = tl.where(qc > 7, qc - 16, qc)
    qm = (byte >> 4) & 15
    qm = tl.where(qm > 7, qm - 16, qm)

    # ~20-bit weight = bf16 + correction
    ls = _half_ulp_log2(pb)
    h = tl.floor(ls / 2.0)
    w = pb + (qc.to(tl.float32) / 7.0) * tl.exp2(h) * tl.exp2(ls - h)

    # momentum = level(|qm|) * sign * v_old, levels 2^(k-6) for k = 1..7 (1/32 .. 2), 0 for k = 0
    km = tl.abs(qm).to(tl.float32)
    lev = tl.where(km > 0, tl.exp2(km - 6.0), 0.0)
    m = tl.where(qm < 0, -lev, lev) * v_old[:, None]
    m = m + (1.0 - b1) * (g - m)
    v = tl.maximum(v_old * b2, tl.max(tl.abs(g), axis=1))

    w = w * wd_mult - lr * (c_g * g + c_m * m) / (v[:, None] + eps)

    # weight split: bf16 to nearest, correction stochastically rounded to 1/7 of half an ulp
    nb16 = w.to(tl.bfloat16)
    nf = nb16.to(tl.float32)
    ls2 = _half_ulp_log2(nf)
    h2 = tl.floor(-ls2 / 2.0)
    en = (w - nf) * tl.exp2(h2) * tl.exp2(-ls2 - h2)
    en = tl.minimum(tl.maximum(en, -1.0), 1.0) * 7.0
    u1 = tl.rand(seed, off)
    cn = tl.floor(en + u1)
    cn = tl.minimum(tl.maximum(cn, -7.0), 7.0).to(tl.int32)

    # momentum code: r = m / v, unbiased stochastic rounding between neighbouring octave levels
    r = m / (v[:, None] + 1e-30)
    a = tl.minimum(tl.abs(r), 2.0)
    u2 = tl.rand(seed + 7919, off)
    e = tl.floor(tl.log2(tl.maximum(a, 1e-30)))
    lo = tl.exp2(e)
    k_hi = tl.where(u2 < (a - lo) / lo, e + 7.0, e + 6.0)           # between lo = 2^e (level e+6) and 2^(e+1)
    k_small = tl.where(u2 < a * 32.0, 1.0, 0.0)                       # below 1/32: between 0 and 1/32
    k = tl.where(a < 1.0 / 32.0, k_small, tl.minimum(k_hi, 7.0))
    k = tl.where(a >= 2.0, 7.0, k)
    qn = tl.where(r < 0, -k, k).to(tl.int32)

    out = (((qn & 15) << 4) | (cn & 15)).to(tl.int8)
    tl.store(p_ptr + off, nb16, mask=mask)
    tl.store(s_ptr + off, out, mask=mask)
    tl.store(v_ptr + blk, v, mask=bmask)


def full_precision(p, state):
    """The ~20-bit weight (bf16 + 4-bit correction) as fp32."""
    s = state.get("s") if state else None
    x = p.detach().float()
    if s is None:
        return x
    qc = (s.to(torch.int32) & 15); qc = torch.where(qc > 7, qc - 16, qc)
    e = ((x.view(torch.int32) >> 23) & 255).clamp_min(1) - 127 - 8
    return x + qc.float() / 7.0 * torch.exp2(e.float())


def momentum(p, state):
    """The momentum as fp32 (decoded from the high 4 bits and the block scale), for checks."""
    s, v, K = state["s"], state["v"], state["K"]
    qm = (s.to(torch.int32) >> 4) & 15; qm = torch.where(qm > 7, qm - 16, qm)
    lev = torch.where(qm != 0, torch.exp2(qm.abs().float() - 6), torch.zeros_like(qm, dtype=torch.float32))
    return torch.sign(qm.float()) * lev * v.repeat_interleave(K)[: p.numel()].view(p.shape)


class CandidateOptimizer(Optimizer):
    def __init__(self, params, lr=DEFAULT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, seed=0, **kwargs):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._seed = seed
        assert SCHEDULE == [(0, 16)] or all(k <= SCHEDULE[0][1] for _, k in SCHEDULE), "only shrinking block schedules"

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, group)

    @torch.no_grad()
    def step_param(self, p):
        for group in self.param_groups:
            if any(q is p for q in group["params"]):
                self._update(p, group)
                return

    def _update(self, p, group):
        lr = group["lr"]
        b1, b2 = group["betas"]
        assert p.dtype == torch.bfloat16 and p.is_contiguous(), "flashbm3 needs contiguous bf16 weights"
        st = self.state[p]
        n = p.numel()
        if "v" not in st:
            st["step"] = 0
            st["K"] = SCHEDULE[0][1]
            st["v"] = torch.zeros((n + st["K"] - 1) // st["K"], device=p.device, dtype=torch.float32)
            st["s"] = torch.zeros(p.shape, device=p.device, dtype=torch.int8)   # correction 0, momentum 0
        st["step"] += 1
        t = st["step"]
        k = SCHEDULE[0][1]
        for start, kk in SCHEDULE:
            if t > start:
                k = kk
        if k != st["K"]:                   # split only: children inherit the parent's scale, so r = m / v is unchanged
            st["v"] = st["v"].repeat_interleave(st["K"] // k)[: (n + k - 1) // k].contiguous()
            st["K"] = k
        nb = (n + k - 1) // k
        bb = max(1, min(BB, 4096 // k))
        self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF
        _fbm3_kernel[((nb + bb - 1) // bb,)](
            p.data, st["s"], p.grad, st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b1, b2,
            (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), group["eps"], self._seed, K=k, BB=bb,
            num_warps=16 if k >= 4096 else 4)
