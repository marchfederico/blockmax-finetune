"""Checks nomom_triton: the update equals a float64 reference in expectation (stochastic rounding unbiased), in both modes."""
import sys, torch
sys.path.insert(0, "strange/fused")
import nomom_triton as NM
torch.manual_seed(0)
for stateless in (True, False):
    NM.STATELESS = stateless
    n, lr = 4096 * 64, 1e-3
    w0 = torch.randn(n, device="cuda").to(torch.bfloat16)
    g1, g2 = torch.randn(n, device="cuda").to(torch.bfloat16), torch.randn(n, device="cuda").to(torch.bfloat16)
    ref = w0.double()
    v = torch.zeros(n // 16, dtype=torch.float64, device="cuda")
    acc = torch.zeros(n, dtype=torch.float64, device="cuda")
    trials = 200
    for tr in range(trials):
        p = torch.nn.Parameter(w0.clone()); o = NM.CandidateOptimizer([p], lr=lr, betas=(0.9, 0.95), seed=tr)
        for g in (g1, g2):
            p.grad = g.clone(); o.step()
        acc += p.detach().double()
    r = w0.double(); v = torch.zeros(n // 16, dtype=torch.float64, device="cuda")
    for g in (g1, g2):
        gm = g.double().view(-1, 16).abs().amax(1)
        v = gm if stateless else torch.maximum(v * 0.95, gm)
        r = r - lr * g.double() / v.repeat_interleave(16)
    bias = (acc / trials - r).abs().mean().item()
    ulp = (r.abs().mean().item()) * 2 ** -8
    print(f"stateless={stateless}: mean |E[w] - ref| = {bias:.2e} (one bf16 rounding step ~ {ulp:.1e}, step size {lr:.0e})", flush=True)
    assert bias < ulp / 5
print("NOMOM OK")
