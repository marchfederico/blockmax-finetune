"""Checks flashbm_triton: (1) the ~24-bit weight follows a float64 reference of the same update; (2) its bf16 + correction pair
decodes identically with FlashOptim's own reconstruct_fp32_param; (3) steps of 1/100 bf16 ulp accumulate, within the bound of the format's deterministic rounding (at most half of 1/254 ulp per step)."""
import sys, torch
sys.path.insert(0, "strange/fused"); sys.path.insert(0, "pylib_fo")
import flashbm_triton as FB
from flashoptim import reconstruct_fp32_param
torch.manual_seed(0)


def mom_of(q):
    q = q.double(); mag = torch.exp2((q.abs() - 101) * 0.25)
    return torch.where(q > 0, mag, torch.where(q < 0, -mag, torch.zeros_like(mag)))


def code_of(m):
    idx = torch.floor(torch.log2(m.abs().clamp_min(1e-30)) * 4 + 0.5) + 101
    return (idx.clamp(1, 127) * torch.sign(m)).to(torch.int8)


n, K, lr, b1, b2, eps = 50000, 16, 1e-3, 0.9, 0.95, 1e-8
FB.SCHEDULE = [(0, K)]
p = torch.nn.Parameter(torch.randn(n, device="cuda").to(torch.bfloat16))
opt = FB.CandidateOptimizer([p], lr=lr)
w = p.detach().double().clone(); q = torch.zeros(n, dtype=torch.int8, device="cuda"); v = torch.zeros(n // K, dtype=torch.float64, device="cuda")
for t in range(1, 11):
    g = torch.randn(n, device="cuda").to(torch.bfloat16); p.grad = g.clone(); opt.step()
    gd = g.double(); v = torch.maximum(v * b2, gd.view(-1, K).abs().amax(1))
    m = mom_of(q); m = m + (1 - b1) * (gd - m); q = code_of(m.float())
    w = w - lr * ((1 - b1) / (1 - b1 ** t) * gd + b1 / (1 - b1 ** (t + 1)) * m) / (v.repeat_interleave(K) + eps)
    st = opt.state[p]; full = FB.full_precision(p, st).double()
    rel = ((full - w).abs() / w.abs().clamp_min(1e-3)).max().item()
    ulp_rel = 2 ** -8                                           # bf16 alone: up to half an ulp, 2^-8 relative
    print(f"step {t}: max relative error of the 24-bit weight {rel:.2e} (bf16 alone would be up to {ulp_rel:.1e}); "
          f"momentum codes differing {(st['m'] != q).float().mean().item():.4f}", flush=True)
    w = full.clone()                                            # follow the kernel so rounding does not accumulate in the check
assert rel < ulp_rel / 50
fo = reconstruct_fp32_param(p.detach(), opt.state[p]["c"]).double()
d = (fo - FB.full_precision(p, opt.state[p]).double()).abs().max().item()
print(f"FlashOptim's decoder vs ours: max difference {d:.2e}", flush=True)
assert d < 1e-7

x = torch.nn.Parameter(torch.ones(4096, device="cuda", dtype=torch.bfloat16))
o2 = FB.CandidateOptimizer([x], lr=2 ** -7 / 100, betas=(0.0, 0.0), eps=0.0)
for _ in range(200):
    x.grad = torch.full_like(x, -1.0); o2.step()            # each step +1/100 of a bf16 ulp at 1.0
moved = (FB.full_precision(x, o2.state[x]).double() - 1.0).mean().item() / 2 ** -7
bound = 200 * 0.5 / 254                                    # FlashOptim's format rounds to nearest 1/254 ulp at every step
print(f"200 steps of +0.01 ulp: moved {moved:.3f} ulp (exact 2.000; deterministic rounding allows +-{bound:.3f}); "
      f"bf16 part {x.detach().double().mean().item():.6f}", flush=True)
assert abs(moved - 2.0) <= bound + 1e-6
print("FLASHBM KERNEL OK")
