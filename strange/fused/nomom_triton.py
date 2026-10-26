# IDEA: NO OPTIMIZER STATE. bf16 weights updated with stochastic rounding, no momentum and no weight correction:
#   u = g / scale,  w <- SR_bf16(w - lr u)
# The scale is per block of 16 weights:
#   STATELESS = True   scale = max |g| over the block, from the current gradient only: 2 bytes per weight in total
#   STATELESS = False  scale = v = max(b2 v, max |g|), BlockMax's decaying max: 2 + 4/16 = 2.25 bytes per weight
# Meant for streamed updates (step_param from a post-accumulate-grad hook, the gradient freed at once), where no gradient
# is held across a step either.
import torch
import triton
import triton.language as tl
from torch.optim.optimizer import Optimizer

NAME = "nomom_triton"
DEFAULT_LR = 1e-6
SCHEDULE = [(0, 16)]
INHERIT = True
STATELESS = True
BB = 64


@triton.jit
def _nomom_kernel(p_ptr, g_ptr, v_ptr, n, nb, lr, wd_mult, b2, seed, K: tl.constexpr, BB: tl.constexpr,
                  STATELESS: tl.constexpr):
    blk = tl.program_id(0) * BB + tl.arange(0, BB)
    off = blk[:, None] * K + tl.arange(0, K)[None, :]
    mask = off < n
    bmask = blk < nb
    g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + off, mask=mask, other=0.0).to(tl.float32)
    gmax = tl.max(tl.abs(g), axis=1)
    if STATELESS:
        v = gmax
    else:
        v = tl.maximum(tl.load(v_ptr + blk, mask=bmask, other=0.0) * b2, gmax)
        tl.store(v_ptr + blk, v, mask=bmask)
    u = tl.where(v[:, None] > 0, g / v[:, None], 0.0)
    w = p * wd_mult - lr * u
    r = tl.randint(seed, off).to(tl.int32) & 0xFFFF                      # stochastic rounding fp32 -> bf16
    bits = (w.to(tl.int32, bitcast=True) + r) & -65536
    tl.store(p_ptr + off, bits.to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)


class CandidateOptimizer(Optimizer):
    def __init__(self, params, lr=DEFAULT_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, seed=0, **kwargs):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._seed = seed

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
        assert p.dtype == torch.bfloat16 and p.is_contiguous(), "nomom needs contiguous bf16 weights"
        lr, b2, K = group["lr"], group["betas"][1], SCHEDULE[0][1]
        st = self.state[p]
        n = p.numel()
        nb = (n + K - 1) // K
        if "v" not in st:
            st["step"] = 0
            st["v"] = torch.zeros(1 if STATELESS else nb, device=p.device, dtype=torch.float32)
        st["step"] += 1
        bb = max(1, min(BB, 4096 // K))
        self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF
        _nomom_kernel[((nb + bb - 1) // bb,)](p.data, p.grad, st["v"], n, nb, lr, 1.0 - lr * group["weight_decay"], b2,
                                              self._seed, K=K, BB=bb, STATELESS=STATELESS)
