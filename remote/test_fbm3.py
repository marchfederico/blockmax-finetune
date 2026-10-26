"""Checks flashbm3_triton: (1) the stored momentum is unbiased (mean of the decoded momentum over many identical blocks equals
the exact momentum); (2) steps of 1/100 bf16 ulp accumulate on average (stochastic rounding of the 4-bit correction); (3) over
10 steps the ~20-bit weight follows a float64 reference to within the format's resolution plus momentum-quantisation noise."""
import sys, torch
sys.path.insert(0, "strange/fused")
import flashbm3_triton as F3
torch.manual_seed(0)
F3.SCHEDULE = [(0, 16)]
K, B = 16, 20000
pattern = torch.tensor([1.0, -0.5, 0.25, -0.1, 0.05, 0.02, -0.01, 0.005, 0.8, -0.3, 0.15, 0.07, -0.03, 0.002, 0.6, -1.0], device="cuda") * 1e-2
p = torch.nn.Parameter(torch.zeros(B * K, device="cuda", dtype=torch.bfloat16))
o = F3.CandidateOptimizer([p], lr=0.0)
for t in range(3):
    p.grad = pattern.repeat(B).to(torch.bfloat16); o.step()
m_exact = (1 - 0.9 ** 3) * pattern.to(torch.bfloat16).float()                  # EMA of a constant gradient after 3 steps
m_mean = F3.momentum(p, o.state[p]).view(B, K).mean(0)
rel = ((m_mean - m_exact).abs() / m_exact.abs()).max().item()
print(f"(1) momentum after 3 steps, mean of {B} stochastic encodings vs exact: max relative error {rel:.3f}", flush=True)
assert rel < 0.05

x = torch.nn.Parameter(torch.ones(4096 * 4, device="cuda", dtype=torch.bfloat16))
o2 = F3.CandidateOptimizer([x], lr=2 ** -7 / 100, betas=(0.0, 0.0), eps=0.0)
for _ in range(200):
    x.grad = torch.full_like(x, -1.0); o2.step()
moved = (F3.full_precision(x, o2.state[x]).double() - 1.0).mean().item() / 2 ** -7
print(f"(2) 200 steps of +0.01 ulp: moved {moved:.3f} ulp on average (exact 2.000)", flush=True)
assert abs(moved - 2.0) < 0.05

n = 50000; lr, b1, b2, eps = 1e-3, 0.9, 0.95, 1e-8
q = torch.nn.Parameter(torch.randn(n, device="cuda").to(torch.bfloat16)); o3 = F3.CandidateOptimizer([q], lr=lr)
w = q.detach().double().clone(); m = torch.zeros_like(w); v = torch.zeros(n // K, dtype=torch.float64, device="cuda")
for t in range(1, 11):
    g = torch.randn(n, device="cuda").to(torch.bfloat16); q.grad = g.clone(); o3.step()
    gd = g.double(); v = torch.maximum(v * b2, gd.view(-1, K).abs().amax(1))
    m = m + (1 - b1) * (gd - m)
    w = w - lr * ((1 - b1) / (1 - b1 ** t) * gd + b1 / (1 - b1 ** (t + 1)) * m) / (v.repeat_interleave(K) + eps)
d = (F3.full_precision(q, o3.state[q]).double() - w)
print(f"(3) after 10 steps: mean |w - ref| = {d.abs().mean().item():.2e}, mean (w - ref) = {d.mean().item():+.1e}, "
      f"typical step {lr:.0e}", flush=True)
assert d.abs().mean().item() < 2 * lr and abs(d.mean().item()) < 1e-4
print("FBM3 KERNEL OK")
