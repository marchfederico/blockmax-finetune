# IDEA: PACKED BLOCKMAX with a block-size SCHEDULE: K changes during training (e.g. 64 -> 32 -> 16). At a switch each new block's
# scale v starts from its parent block (split: copy; merge: max of the children). The packed momentum lives in the weights, so it carries over.
import math
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "packedsched_triton"
DEFAULT_LR = 4e-3
SCHEDULE = [(0, 64), (1000, 32), (2000, 16)]  # (first step, K); rewritten by remote/blocksched.py
INHERIT = True  # False: at a switch every block's scale v restarts from zero instead of its parent's
BB = 64


@triton.jit
def _packed_kernel(p_ptr, g_ptr, v_ptr, n, nb, lr, wd_mult, b1, b2, c_g, c_m, eps,
                   K: tl.constexpr, BB: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0)
    p = tl.load(p_ptr + off, mask=mask, other=0.0)
    v = tl.load(v_ptr + blk, mask=bmask, other=0.0)
    v = tl.maximum(v * b2, tl.max(tl.abs(g), axis=1))

    bits = p.to(tl.int32, bitcast=True)
    q = bits & 255
    q = tl.where(q > 127, q - 256, q)
    qf = q.to(tl.float32)
    mag = tl.exp2((tl.abs(qf) - 101.0) * 0.25)
    m = tl.where(q > 0, mag, tl.where(q < 0, -mag, 0.0))          # code 0 means exactly zero
    m = m + (1.0 - b1) * (g - m)

    idx = tl.floor(tl.log2(tl.maximum(tl.abs(m), 1e-30)) * 4.0 + 0.5) + 101.0
    idx = tl.minimum(tl.maximum(idx, 1.0), 127.0)
    sgn = tl.where(m > 0, 1.0, tl.where(m < 0, -1.0, 0.0))
    code = (idx * sgn).to(tl.int32) & 255

    w = p * wd_mult - lr * (c_g * g + c_m * m) / (v[:, None] + eps)
    out = ((w.to(tl.int32, bitcast=True) & -256) | code).to(tl.float32, bitcast=True)
    tl.store(p_ptr + off, out, mask=mask)
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
        st = self.state[p]
        n = p.numel()
        if "v" not in st:
            st["step"] = 0
            st["K"] = SCHEDULE[0][1]
            st["v"] = torch.zeros((n + st["K"] - 1) // st["K"], device=p.device, dtype=p.dtype)
            p.data.view(torch.int32).bitwise_and_(-256)
        st["step"] += 1
        t = st["step"]
        k = _k_at(t)
        if k != st["K"]:
            st["v"] = _rescale(st["v"], n, st["K"], k) if INHERIT else torch.zeros((n + k - 1) // k, device=p.device, dtype=p.dtype)
            st["K"] = k
        nb = (n + k - 1) // k
        bb = max(1, min(BB, 4096 // k))    # at most 4096 weights per program (16384 spilled registers: 2.7x slower); same results
        _packed_kernel[((nb + bb - 1) // bb,)](
            p.data, p.grad, st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b1, b2,
            (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), group["eps"], K=k, BB=bb)
