# IDEA: ABLATION VARIANTS of Packed BlockMax with the block-size schedule (fp32 weights). Two switches, as module globals:
#   MOMENTUM = "packed" (8-bit log code in the weight's low 8 bits, the deliverable) or "fp32" (a separate fp32 tensor, no packing,
#              no quantisation, weights keep all 32 bits)
#   STAT     = "max" (decaying max |g| per block, the deliverable) or "meansq" (Adam-mini style: an EMA of the block's mean g^2,
#              bias-corrected, denominator sqrt)
# With MOMENTUM="packed", STAT="max" this is bit-identical to packedsched_triton.py (checked in remote/test_blockvar.py).
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "blockvar_triton"
DEFAULT_LR = 1e-5
SCHEDULE = [(0, 16)]
INHERIT = True
MOMENTUM = "packed"
STAT = "max"
BB = 64


@triton.jit
def _var_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps, bc2,
                K: tl.constexpr, BB: tl.constexpr, PACKED: tl.constexpr, MEANSQ: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0)
    p = tl.load(p_ptr + off, mask=mask, other=0.0)
    v = tl.load(v_ptr + blk, mask=bmask, other=0.0)
    if MEANSQ:
        cnt = tl.sum(mask.to(tl.float32), axis=1)
        v = b2 * v + (1.0 - b2) * tl.sum(g * g, axis=1) / tl.maximum(cnt, 1.0)
        den = tl.sqrt(v / bc2) + eps
    else:
        v = tl.maximum(v * b2, tl.max(tl.abs(g), axis=1))
        den = v + eps

    if PACKED:
        bits = p.to(tl.int32, bitcast=True)
        q = bits & 255
        q = tl.where(q > 127, q - 256, q)
        qf = q.to(tl.float32)
        mag = tl.exp2((tl.abs(qf) - 101.0) * 0.25)
        m = tl.where(q > 0, mag, tl.where(q < 0, -mag, 0.0))
    else:
        m = tl.load(m_ptr + off, mask=mask, other=0.0)
    m = m + (1.0 - b1) * (g - m)

    w = p * wd_mult - lr * (c_g * g + c_m * m) / den[:, None]
    if PACKED:
        idx = tl.floor(tl.log2(tl.maximum(tl.abs(m), 1e-30)) * 4.0 + 0.5) + 101.0
        idx = tl.minimum(tl.maximum(idx, 1.0), 127.0)
        sgn = tl.where(m > 0, 1.0, tl.where(m < 0, -1.0, 0.0))
        code = (idx * sgn).to(tl.int32) & 255
        w = ((w.to(tl.int32, bitcast=True) & -256) | code).to(tl.float32, bitcast=True)
    else:
        tl.store(m_ptr + off, m, mask=mask)
    tl.store(p_ptr + off, w, mask=mask)
    tl.store(v_ptr + blk, v, mask=bmask)


def _k_at(t):
    k = SCHEDULE[0][1]
    for start, kk in SCHEDULE:
        if t > start:
            k = kk
    return k


def _rescale(v, n, k_old, k_new):
    nb_new = (n + k_new - 1) // k_new
    if k_new < k_old:                      # split: every child inherits its parent's statistic
        return v.repeat_interleave(k_old // k_new)[:nb_new].contiguous()
    f = k_new // k_old                     # merge: max (or mean) over the children
    pad = nb_new * f - v.numel()
    w = torch.nn.functional.pad(v, (0, pad)).view(nb_new, f)
    return (w.mean(1) if STAT == "meansq" else w.amax(1)).contiguous()


class CandidateOptimizer(Optimizer):
    def __init__(self, params, lr=DEFAULT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, **kwargs):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        assert MOMENTUM in ("packed", "fp32") and STAT in ("max", "meansq")

    @torch.no_grad()
    def step(self, closure=None):
        packed, meansq = MOMENTUM == "packed", STAT == "meansq"
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                n = p.numel()
                if "v" not in st:
                    st["step"] = 0
                    st["K"] = SCHEDULE[0][1]
                    st["v"] = torch.zeros((n + st["K"] - 1) // st["K"], device=p.device, dtype=p.dtype)
                    if packed:
                        p.data.view(torch.int32).bitwise_and_(-256)
                    else:
                        st["m"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                k = _k_at(t)
                if k != st["K"]:
                    st["v"] = _rescale(st["v"], n, st["K"], k) if INHERIT else torch.zeros((n + k - 1) // k, device=p.device, dtype=p.dtype)
                    st["K"] = k
                nb = (n + k - 1) // k
                bb = max(1, min(BB, 4096 // k))
                _var_kernel[((nb + bb - 1) // bb,)](
                    p.data, p.grad, st["m"] if not packed else p.data, st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b1, b2,
                    (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), group["eps"], 1 - b2 ** t,
                    K=k, BB=bb, PACKED=packed, MEANSQ=meansq)
