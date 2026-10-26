"""Checks blockvar_triton: (1) packed + max is bit-identical to packedsched_triton through block-size switches; (2) fp32 momentum
+ max and (3) packed-free meansq both follow a float64 reference of their update rules."""
import sys, torch
sys.path.insert(0, "strange/fused")
import packedsched_triton as PS, blockvar_triton as BV
torch.manual_seed(0)
n = 1536 * 8960 + 123
sched = [(0, 4096), (2, 1024), (4, 64), (6, 16)]
PS.SCHEDULE = sched; BV.SCHEDULE = sched
w0 = torch.randn(n, device="cuda"); grads = [torch.randn(n, device="cuda") * 1e-2 for _ in range(8)]
a = torch.nn.Parameter(w0.clone()); b = torch.nn.Parameter(w0.clone())
oa = PS.CandidateOptimizer([a], lr=1e-3); BV.MOMENTUM, BV.STAT = "packed", "max"; ob = BV.CandidateOptimizer([b], lr=1e-3)
for g in grads:
    a.grad = g.clone(); b.grad = g.clone(); oa.step(); ob.step()
same = torch.equal(a.data.view(torch.int32), b.data.view(torch.int32)) and torch.equal(oa.state[a]["v"], ob.state[b]["v"])
print(f"(1) packed+max bit-identical to packedsched_triton over 8 steps and 4 block sizes: {same}", flush=True)
assert same

for stat in ("max", "meansq"):
    BV.MOMENTUM, BV.STAT = "fp32", stat; BV.SCHEDULE = [(0, 16)]
    K, lr, b1, b2, eps = 16, 1e-3, 0.9, 0.95, 1e-8
    m_ = 4096 * 16
    p = torch.nn.Parameter(torch.randn(m_, device="cuda")); o = BV.CandidateOptimizer([p], lr=lr)
    w = p.detach().double().clone(); m = torch.zeros_like(w); v = torch.zeros(m_ // K, dtype=torch.float64, device="cuda")
    for t in range(1, 7):
        g = torch.randn(m_, device="cuda"); p.grad = g.clone(); o.step(); gd = g.double().view(-1, K)
        if stat == "max":
            v = torch.maximum(v * b2, gd.abs().amax(1)); den = v + eps
        else:
            v = b2 * v + (1 - b2) * (gd ** 2).mean(1); den = (v / (1 - b2 ** t)).sqrt() + eps
        m = m + (1 - b1) * (g.double() - m)
        w = w - lr * ((1 - b1) / (1 - b1 ** t) * g.double() + b1 / (1 - b1 ** (t + 1)) * m) / den.repeat_interleave(K)
    err = ((p.detach().double() - w).abs().max() / w.abs().max()).item()
    print(f"({2 if stat == 'max' else 3}) fp32 momentum, {stat}: max error vs float64 reference {err:.2e} of the largest weight", flush=True)
    assert err < 1e-6
print("BLOCKVAR OK")
