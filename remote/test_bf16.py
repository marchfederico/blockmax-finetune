"""Checks packedbf16_triton against a float64 torch reference (same update, momentum code and block scale) and checks that the
stochastic rounding is unbiased for steps far below one bf16 ulp."""
import sys, torch
sys.path.insert(0, "strange/fused")
import packedbf16_triton as PB
torch.manual_seed(0)


def code_of(m):
    idx = torch.floor(torch.log2(m.abs().clamp_min(1e-30)) * 4 + 0.5) + 101
    return (idx.clamp(1, 127) * torch.sign(m)).to(torch.int8)


def mom_of(q):
    q = q.double(); mag = torch.exp2((q.abs() - 101) * 0.25)
    return torch.where(q > 0, mag, torch.where(q < 0, -mag, torch.zeros_like(mag)))


n, K, lr, b1, b2, eps = 5000, 16, 1e-3, 0.9, 0.95, 1e-8
PB.SCHEDULE = [(0, K)]
p = torch.nn.Parameter(torch.randn(n, device="cuda").to(torch.bfloat16))
opt = PB.CandidateOptimizer([p], lr=lr)
w = p.detach().double().clone(); q = torch.zeros(n, dtype=torch.int8, device="cuda"); v = torch.zeros((n + K - 1) // K, dtype=torch.float64, device="cuda")
for t in range(1, 6):
    g = torch.randn(n, device="cuda").to(torch.bfloat16); p.grad = g.clone()
    opt.step()
    gd = g.double(); gp = torch.nn.functional.pad(gd, (0, v.numel() * K - n)).view(-1, K)
    v = torch.maximum(v * b2, gp.abs().amax(1))
    m = mom_of(q); m = m + (1 - b1) * (gd - m); q = code_of(m.float())
    upd = lr * ((1 - b1) / (1 - b1 ** t) * gd + b1 / (1 - b1 ** (t + 1)) * m) / (v.repeat_interleave(K)[:n] + eps)
    w = w - upd
    err = (p.detach().double() - w).abs().max().item(); ulp = (w.abs().max().item()) * 2 ** -7
    qm = (opt.state[p]["m"] != q).float().mean().item()
    print(f"step {t}: max |w - ref| {err:.2e} (one bf16 ulp at the largest weight {ulp:.2e}); momentum codes differing {qm:.4f}")
    w = p.detach().double().clone()                       # follow the kernel's rounded weights so errors do not accumulate
assert err < 2 * ulp and qm < 0.01

# unbiasedness: a step of 1/100 ulp, repeated over 2000 independent copies, should move the mean by 1/100 ulp
x0 = torch.full((2000 * 16,), 1.0, device="cuda", dtype=torch.bfloat16)
p2 = torch.nn.Parameter(x0.clone()); opt2 = PB.CandidateOptimizer([p2], lr=1.0, betas=(0.0, 0.0), eps=0.0)
target = 2 ** -7 / 100                                     # bf16 ulp at 1.0 is 2^-7
p2.grad = torch.full_like(p2, target).to(torch.bfloat16)   # with b1 = 0 and v = |g|: step = lr * g / |g| ... scaled below
opt2.param_groups[0]["lr"] = target
opt2.step()
moved = (1.0 - p2.detach().double()).mean().item()
print(f"stochastic rounding: asked to move {target:.3e}, mean move {moved:.3e} ({100 * moved / target:.1f}%)")
assert abs(moved / target - 1) < 0.15
print("BF16 KERNEL OK")
