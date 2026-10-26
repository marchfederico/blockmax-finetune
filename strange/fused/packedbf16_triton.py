# IDEA: the 16-bit version of Packed BlockMax. A bf16 weight has only 7 mantissa bits, so the 8-bit momentum cannot live inside it
# as it does in fp32. Here the weight stays bf16 (updated with STOCHASTIC ROUNDING, so steps far below one bf16 ulp still move it on
# average) and the momentum is a separate int8 tensor holding the same sign + quarter-octave code as the packed kernel. One scale
# per block, block size scheduled as in packedsched_triton.py. Memory per weight: 2 (weight) + 1 (momentum) bytes + 4/K for the scales.
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "packedbf16_triton"
DEFAULT_LR = 1e-5
SCHEDULE = [(0, 16)]  # (first step, K), as in packedsched_triton.py
INHERIT = True
BB = 64


@triton.jit
def _bf16_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps, seed,
                 K: tl.constexpr, BB: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + off, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(m_ptr + off, mask=mask, other=0).to(tl.int32)
    v = tl.load(v_ptr + blk, mask=bmask, other=0.0)
    v = tl.maximum(v * b2, tl.max(tl.abs(g), axis=1))

    qf = q.to(tl.float32)
    mag = tl.exp2((tl.abs(qf) - 101.0) * 0.25)
    m = tl.where(q > 0, mag, tl.where(q < 0, -mag, 0.0))          # code 0 means exactly zero
    m = m + (1.0 - b1) * (g - m)
    idx = tl.floor(tl.log2(tl.maximum(tl.abs(m), 1e-30)) * 4.0 + 0.5) + 101.0
    idx = tl.minimum(tl.maximum(idx, 1.0), 127.0)
    sgn = tl.where(m > 0, 1.0, tl.where(m < 0, -1.0, 0.0))
    code = (idx * sgn).to(tl.int8)

    w = p * wd_mult - lr * (c_g * g + c_m * m) / (v[:, None] + eps)
    # stochastic rounding fp32 -> bf16: add 16 random bits below the bf16 mantissa, then truncate
    r = tl.randint(seed, off).to(tl.int32) & 0xFFFF
    bits = (w.to(tl.int32, bitcast=True) + r) & -65536
    out = bits.to(tl.float32, bitcast=True).to(tl.bfloat16)
    tl.store(p_ptr + off, out, mask=mask)
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


class CandidateOptimizer(Optimizer):
    def __init__(self, params, lr=DEFAULT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, seed=0, **kwargs):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._seed = seed

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.dtype == torch.bfloat16 and p.is_contiguous(), "packedbf16 needs contiguous bf16 weights"
                st = self.state[p]
                n = p.numel()
                if "v" not in st:
                    st["step"] = 0
                    st["K"] = SCHEDULE[0][1]
                    st["v"] = torch.zeros((n + st["K"] - 1) // st["K"], device=p.device, dtype=torch.float32)
                    st["m"] = torch.zeros(p.shape, device=p.device, dtype=torch.int8)
                st["step"] += 1
                t = st["step"]
                k = _k_at(t)
                if k != st["K"]:
                    st["v"] = _rescale(st["v"], n, st["K"], k) if INHERIT else torch.zeros((n + k - 1) // k, device=p.device, dtype=torch.float32)
                    st["K"] = k
                nb = (n + k - 1) // k
                bb = max(1, min(BB, 4096 // k))    # at most 4096 weights per program (larger tiles spill registers: 30-40x slower)
                self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF   # a fresh random stream per tensor per step
                _bf16_kernel[((nb + bb - 1) // bb,)](
                    p.data, p.grad, st["m"], st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b1, b2,
                    (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), group["eps"], self._seed, K=k, BB=bb, num_warps=16 if k >= 4096 else 4)
