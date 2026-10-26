# Derived from lomo_optim/adalomo.py (lomo-optim 0.1.1), which is under the MIT License:
#   Copyright (c) 2023 OpenLMLab
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#   persons to whom the Software is furnished to do so, subject to the following conditions:
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# BASELINE: AdaLomo (Lv et al. 2023, "AdaLomo: Low-memory Optimization with Adaptive Learning Rate"), the non-ZeRO path of
# lomo_optim 0.1.1 (pip lomo-optim, lomo_optim/adalomo.py) copied unchanged except for one switch:
#   SR = False  the weight is written back to bf16 by round-to-nearest, as in the package (p.data.copy_)
#   SR = True   the same update written back with stochastic rounding, as our bf16 kernels do
# The update: Adafactor's factored second moment (one row and one column vector per matrix, beta2_t = 1 - t^-0.8), no
# momentum, the update clipped to RMS 1 per tensor, the step scaled by the tensor's RMS (a relative learning rate). Each
# tensor is updated from a gradient hook during backward and its gradient freed, so it streams like our nomom kernel.
# remote/test_adalomo.py checks SR = False against the package bit for bit.
import math

import torch
from torch.optim import Optimizer

SR = False


class AdaLomo(Optimizer):
    def __init__(self, model, lr=1e-3, loss_scale=2**10, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8,
                 weight_decay=0.0, seed=0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        self.eps = eps
        self.step_num = 0
        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold
        self.gen = torch.Generator(device="cuda"); self.gen.manual_seed(seed)
        self.grad_func = self.fuse_update()
        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}
        for n, p in self.model.named_parameters():
            if len(p.data.shape) == 1:
                self.exp_avg_sq[n] = torch.zeros(p.data.shape[0], dtype=torch.float32).cuda()
            else:
                self.exp_avg_sq_row[n] = torch.zeros(p.data.shape[0], dtype=torch.float32).cuda()
                self.exp_avg_sq_col[n] = torch.zeros(p.data.shape[1], dtype=torch.float32).cuda()
            if p.requires_grad:
                p.register_hook(self.grad_func)
        super().__init__(self.model.parameters(), dict(lr=lr, eps=eps, weight_decay=weight_decay))

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _write(self, p, p_fp32):
        if SR and p.dtype == torch.bfloat16:                                 # stochastic rounding fp32 -> bf16
            r = torch.randint(0, 65536, p_fp32.shape, device=p.device, dtype=torch.int32, generator=self.gen)
            p.data.copy_(((p_fp32.view(torch.int32) + r) & -65536).view(torch.float32))
        else:
            p.data.copy_(p_fp32)

    def fuse_update(self):
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scale:
                            grad_fp32.div_(self.loss_scale)
                        if self.step_num == 0 and self.decay_rate < 0:
                            decay_rate = -self.decay_rate
                        else:
                            decay_rate = self.decay_rate
                        beta2t = 1.0 - math.pow(self.step_num, decay_rate)
                        update = (grad_fp32**2) + self.eps[0]
                        if len(p.data.shape) > 1:
                            self.exp_avg_sq_row[n].mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                            self.exp_avg_sq_col[n].mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)
                            update = self._approx_sq_grad(self.exp_avg_sq_row[n], self.exp_avg_sq_col[n])
                            update.mul_(grad_fp32)
                        else:
                            self.exp_avg_sq[n].mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                            update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)
                        update.div_((self._rms(update) / self.clip_threshold).clamp_(min=1.0))
                        p_fp32 = p.data.to(torch.float32)
                        p_rms = torch.norm(p_fp32, 2.0) / math.sqrt(p.numel())
                        lr = self.lr * max(self.eps[1], p_rms)
                        if self.weight_decay > 0.0:
                            p_fp32.mul_(1.0 - lr * self.weight_decay)
                        p_fp32.add_(update, alpha=-lr)
                        self._write(p, p_fp32)
            return x
        return func

    def fused_backward(self, loss, lr):
        self.lr = lr
        if self.loss_scale:
            loss = loss * self.loss_scale
        self.step_num += 1
        loss.backward()
        self.grad_func(0)                                                     # the last parameter's gradient arrives after its hook
