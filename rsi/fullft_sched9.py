#!/usr/bin/env python3
"""fullft.py with a block-size SCHEDULE (strange/fused/packedsched_triton.py) and a held-out loss.
   --sched 64,32,16   block sizes, equal shares of the optimizer steps; --no_inherit restarts the scales at each switch
   --heldout_problems N   all programs of N problems are kept out of training; their mean loss is measured before and after
   --decay            learning rate falls linearly from --lr to 0 over the optimizer steps (default: constant)
   --warmup N         learning rate rises linearly from 0 to --lr over the first N optimizer steps (with --decay: then decays)
   --mode adamw       baseline: AdamW on fp32 weights and fp32 gradients, both moments stored in bf16 (fp32 moments do not fit
                      an A10G at 1.5B), gradient checkpointing on to make room; same data, accumulation, decay and warmup
   --mode bf16        16-bit Packed BlockMax (strange/fused/packedbf16_triton.py): bf16 weights with stochastic rounding, int8 momentum
   --mode flashbm     Flash-BlockMax (strange/fused/flashbm_triton.py): FlashOptim's 24-bit weights (bf16 + int8 correction) with the
                      BlockMax update, int8 momentum and the block schedule
   --mode var         ablations (strange/fused/blockvar_triton.py): --momentum packed|fp32, --stat max|meansq; --b2 also for packed
   --mode flashbm3    3 bytes per weight (strange/fused/flashbm3_triton.py): bf16 + one byte = 4-bit correction + 4-bit momentum
   --stream           streamed updates for flashbm / flashbm3: each tensor is updated as its gradient completes, which is then freed
                      (no accumulation: one update per program; pass --b1/--b2 for the per-update decay rates)
   --no_save          skip saving the model (the held-out loss is written to <out>.json either way)
   --mode flash       FlashOptim's FlashAdamW (pip flashoptim 0.1.4 in ../pylib_fo): bf16 weights + 8-bit correction, 8-bit moments
   --mode adalomo     baseline AdaLomo (strange/fused/adalomo_ref.py, = lomo_optim 0.1.1): bf16 weights, factored second moment,
                      no momentum, its own fused backward (always streamed); --adalomo_sr writes the weights back with stochastic
                      rounding instead of round-to-nearest
   --chunked_loss     the loss computed without materialising full-vocabulary logits (same value; less memory)
   (otherwise packed mode only)

Original description of fullft.py:
Full fine-tuning of a pretrained model with Packed BlockMax, the fused Triton kernel (strange/fused/packed16_triton.py).

Every weight is trained. The model is held in 32 bits; the optimizer keeps its momentum in the low 8 bits of each weight and
one number per block of 16 weights, so its own memory is 1/16 of the model (AdamW: twice the model).

   --mode packed      fused kernel, gradients accumulated over --accum programs, then one update        (the adapter's regime)
   --mode stream      fused kernel called from autograd hooks: a weight is updated the moment its gradient is complete and
                      the gradient is released, so the full gradient never exists; one update per program
   --mode adamw16     what fits with AdamW on this card: the model, its gradient and AdamW's two moments all in 16 bits
   --lr 0 (packed)    the control for the kernel itself: all the packing and truncation, no learning

Same data, prompts and labels as the adapter runs (rsi_qlora.py); the result is a model directory validated the same way.
"""
import argparse
import json
import os
import random
import resource
import shutil
import sys
import time

import exp83_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def AdamWBf16State(params, lr, betas=(0.9, 0.999), eps=1e-8):
    """AdamW, no weight decay; arithmetic in fp32, moments stored in bf16."""
    import torch

    class _A(torch.optim.Optimizer):
        @torch.no_grad()
        def step(self, closure=None):
            for g_ in self.param_groups:
                b1, b2 = betas
                for p in g_["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st["t"] = 0
                        st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        st["v"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["t"] += 1; t = st["t"]
                    g = p.grad.float()
                    m = st["m"].float().mul_(b1).add_(g, alpha=1 - b1)
                    v = st["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                    st["m"].copy_(m); st["v"].copy_(v)
                    denom = v.div_(1 - b2 ** t).sqrt_().add_(eps)
                    p.addcdiv_(m, denom, value=-g_["lr"] / (1 - b1 ** t))
                    del g, m, v, denom
    return _A(params, dict(lr=lr))


def chunked_loss(model, ids, lab, chunk=256):
    """The same mean cross-entropy as model(..., labels=...).loss, without the full-vocabulary logits in memory: the LM head
    is applied only at positions whose next token is a label, in checkpointed chunks (logits rebuilt in backward)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    h = model.model(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0]
    tgt = torch.tensor(lab[1:], device="cuda")
    keep = (tgt != -100).nonzero().squeeze(1)
    h, tgt = h[keep], tgt[keep]

    def piece(hc, tc):
        return F.cross_entropy(model.lm_head(hc).float(), tc, reduction="sum")
    tot = sum(checkpoint(piece, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False) for i in range(0, len(tgt), chunk))
    return tot / max(1, len(tgt))


def main(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.join(ROOT, "strange", "fused"))
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.stream:
        a.accum = 1                                                           # streamed: one update per program, gradients freed at once
    dt = torch.bfloat16 if a.mode in ("adamw16", "bf16", "flash", "flashbm", "flashbm3", "nomom", "adalomo") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map={"": 0})
    model.config.use_cache = False; model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params); w_gb = sum(p.numel() * p.element_size() for p in params) / 2 ** 30
    # a CPU copy of every weight, to count what moved; skipped with --no_save (at 7B it alone is 15 GB, all of m4's RAM)
    before = None if a.no_save else [p.detach().to(torch.bfloat16).cpu() for p in params]
    print(f"  full fine-tune, mode {a.mode}, lr {a.lr}: {n / 1e9:.2f}B trainable weights in {len(params)} tensors, {w_gb:.1f} GB at {dt}", flush=True)

    import packedsched_triton as PK                                           # also read for its SCHEDULE/INHERIT bookkeeping in adamw mode
    if a.mode == "adamw16":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, eps=1e-6)
        state_gb = 2 * w_gb
    elif a.mode == "bf16":
        import packedbf16_triton as PK                                        # 16-bit Packed BlockMax: bf16 weights + int8 momentum
        PK.INHERIT = not a.no_inherit
        opt = PK.CandidateOptimizer(params, lr=a.lr, weight_decay=0.0, seed=a.seed)
        state_gb = w_gb / 2                                                   # int8 momentum (the block scales are negligible)
    elif a.mode == "flashbm":
        import flashbm_triton as PK                                           # Flash-BlockMax: bf16 + int8 correction (24-bit) weights,
        PK.INHERIT = not a.no_inherit                                         # BlockMax update with int8 momentum, block schedule
        opt = PK.CandidateOptimizer(params, lr=a.lr, betas=(a.b1, a.b2), weight_decay=0.0)
        state_gb = w_gb                                                       # int8 correction + int8 momentum, relative to bf16 weights
    elif a.mode == "nomom":
        import nomom_triton as PK                                             # no optimizer state: bf16 + stochastic rounding,
        PK.STATELESS = not a.stateful                                         # u = g / block max (current, or decaying with --stateful)
        opt = PK.CandidateOptimizer(params, lr=a.lr, betas=(a.b1, a.b2), weight_decay=0.0, seed=a.seed)
        state_gb = 0.0 if PK.STATELESS else w_gb / 32
    elif a.mode == "adalomo":
        import adalomo_ref as AL                                              # baseline: AdaLomo, updates every weight from its own
        AL.SR = a.adalomo_sr                                                  # gradient hooks inside opt.fused_backward
        opt = AL.AdaLomo(model, lr=a.lr, seed=a.seed)
        state_gb = 0.0
    elif a.mode == "flashbm3":
        import flashbm3_triton as PK                                          # 3 bytes per weight: bf16 + one byte (4-bit correction,
        PK.INHERIT = not a.no_inherit                                         # 4-bit block-relative momentum)
        opt = PK.CandidateOptimizer(params, lr=a.lr, betas=(a.b1, a.b2), weight_decay=0.0, seed=a.seed)
        state_gb = w_gb / 2
    elif a.mode == "flash":
        sys.path.insert(0, os.path.join(ROOT, "pylib_fo"))
        from flashoptim import FlashAdamW                                     # FlashOptim 0.1.4: bf16 weight + 8-bit correction, 8-bit moments
        opt = FlashAdamW(params, lr=a.lr, betas=(a.b1, a.flash_b2), weight_decay=0.0, master_weight_bits=24)
        state_gb = 1.5 * w_gb                                                 # correction + two 8-bit moments, relative to the bf16 weights
    elif a.mode == "adamw":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt = AdamWBf16State(params, lr=a.lr)
        state_gb = w_gb
    elif a.mode == "var":
        import blockvar_triton as PK                                          # ablation variants (fp32 weights)
        PK.INHERIT = not a.no_inherit; PK.MOMENTUM = a.momentum; PK.STAT = a.stat
        opt = PK.CandidateOptimizer(params, lr=a.lr, betas=(a.b1, a.b2), weight_decay=0.0)
        state_gb = w_gb / 16 if a.momentum == "packed" else w_gb
        if a.grad_ckpt:                                                       # room for the separate fp32 momentum (same maths)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        import packedsched_triton as PK                                       # the fused kernel, block size scheduled
        PK.INHERIT = not a.no_inherit
        if a.mode == "packed":
            opt = PK.CandidateOptimizer(params, lr=a.lr, betas=(a.b1, a.b2), weight_decay=0.0)
        else:
            K, BB, kernel = PK.K, PK.BB, PK._packed_kernel
            v = {}; clock = {"t": 0}

            @torch.no_grad()
            def hook(p):
                m = p.numel(); nb = (m + K - 1) // K; t = clock["t"]; b1, b2 = 0.9, 0.95
                kernel[((nb + BB - 1) // BB,)](p.data, p.grad, v[p], m, nb, a.lr, 1.0, b1, b2, (1 - b1) / (1 - b1 ** t), b1 / (1 - b1 ** (t + 1)), 1e-8, K=K, BB=BB)
                p.grad = None                                                 # this gradient is released at once

            for p in params:
                v[p] = torch.zeros((p.numel() + K - 1) // K, device=p.device, dtype=p.dtype)
                p.data.view(torch.int32).bitwise_and_(-256)
                p.register_post_accumulate_grad_hook(hook)
        state_gb = w_gb / 16

    probs = {p["pid"]: p for p in D.load("train")}
    data = [json.loads(l) for l in open(a.data)]; ex = []
    for d in data:
        pr = tok.apply_chat_template(D.messages(probs[d["pid"]]), tokenize=False, add_generation_prompt=True)
        p_ids = tok(pr, add_special_tokens=False)["input_ids"]; c_ids = tok(f"```python\n{d['code']}\n```" + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) <= a.max_len:
            ex.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids, d["pid"]))
    random.shuffle(ex); ex = ex[:a.limit] if a.limit else ex
    pids = sorted({d["pid"] for d in data}); random.Random(1234).shuffle(pids)  # the held-out problems do not depend on --seed
    held = set(pids[:a.heldout_problems])
    ex_all = ex
    ex = [e for e in ex_all if e[2] not in held]; ho = [e for e in ex_all if e[2] in held]
    ex = [(i, l) for i, l, _ in ex]; ho = [(i, l) for i, l, _ in ho]

    @torch.no_grad()
    def heldout_loss():
        model.eval(); tot = 0.0
        for ids, lab in ho:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot += float(model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([lab], device="cuda")).loss)
        model.train(); return tot / max(1, len(ho))

    n_steps = -(-len(ex) * a.epochs // a.accum)
    if a.stream:
        assert a.mode in ("flashbm", "flashbm3", "flash", "packed", "nomom", "adalomo"), "--stream: flashbm, flashbm3, flash, packed, nomom, adalomo"
        lr_now = {"v": a.lr}

        @torch.no_grad()
        def _release(p):
            for g in opt.param_groups:
                g["lr"] = lr_now["v"]
            opt.step_param(p)
            p.grad = None                                                     # this gradient is released at once

        if a.mode == "flash":                                                 # FlashOptim's own gradient release; its hooks read
            from flashoptim import enable_gradient_release                    # the group lr, which the loop sets before each backward
            enable_gradient_release(model, opt)

            def _set_lr():
                for g in opt.param_groups:
                    g["lr"] = lr_now["v"]
        elif a.mode == "adalomo":
            pass                                                              # AdaLomo registers its own hooks
        else:
            for p in params:
                p.register_post_accumulate_grad_hook(_release)
    if a.grad_ckpt:                                                           # recompute activations in backward (same maths, less memory)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ks = [int(k) for k in a.sched.split(",")]
    PK.SCHEDULE = [(n_steps * j // len(ks), k) for j, k in enumerate(ks)]
    ho0 = heldout_loss()
    print(f"  {len(ex)} training programs x {a.epochs} epochs = {n_steps} steps, {len(ho)} held-out programs of {len(held)} problems; "
          f"schedule {PK.SCHEDULE} inherit {PK.INHERIT}; held-out loss before {ho0:.4f}", flush=True)
    ex = [e for _ in range(a.epochs) for e in random.sample(ex, len(ex))]
    accum = 1 if a.mode == "stream" else a.accum
    losses = []; n_done = 0; skipped = 0; t0 = time.time(); t_opt = 0.0; torch.cuda.reset_peak_memory_stats()
    for i, (ids, lab) in enumerate(ex):
        try:
            if a.mode == "stream":
                clock["t"] += 1
            if a.stream:
                lr_now["v"] = a.lr * (max(0.0, 1.0 - n_done / n_steps) if a.decay else 1.0)
                if a.mode == "flash":
                    _set_lr()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.mode != "adamw16"):
                if a.chunked_loss:
                    loss = chunked_loss(model, ids, lab)
                else:
                    loss = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([lab], device="cuda")).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is not finite at program {i}")
            if a.mode == "adalomo":
                opt.fused_backward(loss, lr_now["v"])
            else:
                (loss / accum).backward()
            losses.append(float(loss))
            if a.stream:
                n_done += 1
        except torch.OutOfMemoryError:
            skipped += 1; torch.cuda.empty_cache()
            for p in params:
                p.grad = None
            continue
        if a.mode != "stream" and not a.stream and ((i + 1) % accum == 0 or i + 1 == len(ex)):
            torch.cuda.synchronize(); t1 = time.time()
            if a.decay or a.warmup:
                if n_done < a.warmup:
                    f = (n_done + 1) / a.warmup                                  # linear warmup to --lr
                elif a.decay:
                    f = max(0.0, 1.0 - (n_done - a.warmup) / max(1, n_steps - a.warmup))  # then linear decay to 0
                else:
                    f = 1.0
                for g in opt.param_groups:
                    g["lr"] = a.lr * f
            opt.step(); opt.zero_grad(set_to_none=True); n_done += 1
            torch.cuda.synchronize(); t_opt += time.time() - t1
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(ex)} loss {sum(losses[-100:]) / max(1, len(losses[-100:])):.3f} [{time.time() - t0:.0f}s]", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    ho1 = heldout_loss()
    print(f"  HELDOUT loss {ho0:.4f} -> {ho1:.4f}  train loss (last 100) {sum(losses[-100:]) / max(1, len(losses[-100:])):.4f}", flush=True)
    with open(a.out.rstrip("/") + ".json", "w") as fh:
        json.dump(dict(stream=a.stream, b1=a.b1, peak_gb=peak, momentum=a.momentum, stat=a.stat, adalomo_sr=a.adalomo_sr, b2=a.b2, mode=a.mode, chunked_loss=a.chunked_loss, skipped=skipped, sched=a.sched, inherit=PK.INHERIT, lr=a.lr, decay=a.decay, warmup=a.warmup, steps_done=n_done, seed=a.seed, heldout_before=ho0, heldout_after=ho1,
                       train_last100=sum(losses[-100:]) / max(1, len(losses[-100:])), steps=n_steps), fh)
    moved = tensors_moved = 0; d2 = 0.0
    with torch.no_grad():
        for p, b in zip(params, before or []):
            pc = p.detach().to(torch.bfloat16).cpu(); ne = pc != b                # on the CPU: the GPU can be full at this point
            moved += int(ne.sum()); tensors_moved += int(bool(ne.any())); d2 += float((pc.float() - b.float()).pow(2).sum()); del pc, ne
    print(f"  trained on {len(losses)} programs ({skipped} skipped for memory), loss {sum(losses[:50]) / max(1, len(losses[:50])):.3f} -> {sum(losses[-50:]) / max(1, len(losses[-50:])):.3f} "
          f"[{time.time() - t0:.0f}s, of which {t_opt:.1f}s in the optimizer]", flush=True)
    print(f"  memory: peak GPU {peak:.1f} GB = {peak / w_gb:.2f} copies of the model; optimizer state {state_gb:.2f} GB; peak RAM {resource.getrusage(0).ru_maxrss / 2 ** 20:.1f} GB", flush=True)
    print(f"  change: norm {d2 ** 0.5:.3f}; {100 * moved / n:.1f}% of all weights differ once rounded to 16 bits; {tensors_moved} of {len(params)} tensors changed", flush=True)
    del before
    if a.no_save:
        return
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    model.save_pretrained(a.out, max_shard_size="5GB"); tok.save_pretrained(a.out)
    print(f"  saved {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--mode", default="packed", choices=["packed", "adamw", "bf16", "flash", "flashbm", "flashbm3", "nomom", "var", "adalomo"])
    ap.add_argument("--data", default="results/sft_s_control.jsonl")
    ap.add_argument("--out", default="models/fullft")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sched", default="16")
    ap.add_argument("--no_inherit", action="store_true")
    ap.add_argument("--heldout_problems", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--decay", action="store_true")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--chunked_loss", action="store_true")
    ap.add_argument("--momentum", default="packed", choices=["packed", "fp32"])
    ap.add_argument("--stat", default="max", choices=["max", "meansq"])
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--flash_b2", type=float, default=0.999)
    ap.add_argument("--stateful", action="store_true")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--adalomo_sr", action="store_true")
    main(ap.parse_args())
