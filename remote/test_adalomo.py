#!/usr/bin/env python3
"""strange/fused/adalomo_ref.py against the lomo_optim package (pip lomo-optim 0.1.1 in ../pylib_lomo): with SR = False the
weights after 5 fused backward passes on a small bf16 model must match bit for bit; with SR = True they must stay close
(stochastic rounding of the same update) and differ somewhere."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "strange", "fused")); sys.path.insert(0, os.path.join(ROOT, "pylib_lomo"))
import adalomo_ref as R
from lomo_optim import AdaLomo as PkgAdaLomo


def make():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Embedding(100, 64), torch.nn.Linear(64, 96), torch.nn.LayerNorm(96),
                               torch.nn.Linear(96, 100)).cuda().to(torch.bfloat16)


def train(model, opt):
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        x = torch.randint(0, 100, (8, 12), generator=g).cuda(); y = torch.randint(0, 100, (8, 12), generator=g).cuda()
        loss = torch.nn.functional.cross_entropy(model(x).float().view(-1, 100), y.view(-1))
        opt.fused_backward(loss, 1e-2)
    return [p.detach().clone() for p in model.parameters()]


ref = train(m := make(), PkgAdaLomo(m, lr=1e-2))
R.SR = False
mine = train(m := make(), R.AdaLomo(m, lr=1e-2))
assert all(torch.equal(a, b) for a, b in zip(ref, mine)), "round-to-nearest copy differs from the package"
print("SR=False: bit-identical to lomo_optim")
R.SR = True
sr = train(m := make(), R.AdaLomo(m, lr=1e-2))
d = max(float((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-12)) for a, b in zip(ref, sr))
assert any(not torch.equal(a, b) for a, b in zip(ref, sr)) and d < 0.05, d
print(f"SR=True: differs, max relative deviation {d:.4f}")
