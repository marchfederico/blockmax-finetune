# Packed BlockMax recipe for full fine-tuning

Use **block 16 throughout, at a learning rate tuned for your model, decayed linearly to 0**, with one of these storage formats:

- **No momentum, 2.25 bytes per weight, streamed** (`nomom_triton.py`, next section). The new default: best and smallest
  on Qwen 1.5B (tie), Qwen 3B and SmolLM2 1.7B, and it fits a 7B full fine-tune on one 24 GB GPU.
- **3-byte kernel with streamed updates** (`flashbm3_triton.py`, section "3-byte kernel"). The previous default, if you
  want to keep momentum.
- **Flash-BlockMax, accumulated** (`flashbm_triton.py`). When memory is not a concern and you want the standard
  accumulate-then-step loop.

Flash-BlockMax stores weights in FlashOptim's 24-bit format and applies Packed BlockMax's update. On two models it matched
or slightly beat FlashOptim's tuned FlashAdamW, with less memory:
- **Qwen2.5-Coder-1.5B:** lower held-out loss at every seed.
- **SmolLM2-1.7B:** lower on average, at 2 of 3 seeds.

Do not use the coarse-to-fine block-size schedule. It helped on one model, hurt on the other, and hurts from scratch.

This replaces `DOWN4096_RECIPE.md`, which recommended the schedule. That recommendation came from comparing against a
block-16 baseline run at the wrong learning rate.

## Even lower: no momentum at all, 2.25 bytes per weight

`strange/fused/nomom_triton.py` with `STATELESS = False` (`rsi/fullft_sched9.py --mode nomom --stream --stateful --b2
0.99361`). It keeps the bf16 weights, updated with stochastic rounding, plus only BlockMax's decaying block maximum: 4 bytes
per 16 weights, and no momentum or correction byte. Each streamed update is the current gradient divided by its block's
decaying maximum.

On Qwen2.5-Coder-1.5B, 3 seeds, at a tuned learning rate of 1.41e-6:

| Setup | Bytes per weight | Held-out loss | Peak GPU |
|---|---|---|---|
| 3-byte kernel, streamed | 3 | 0.1190 | 10.1 GB |
| **No momentum, decaying block max, streamed** | **2.25** | **0.1191** | **8.7 GB** |
| No momentum, current block max only (no state at all) | 2 | 0.1201 | 8.3 GB |
| FlashOptim FlashAdamW, gradient release | about 5 | 0.1197 | 12.8 GB |

- **Momentum isn't needed** when updates are streamed and the learning rate decays. The decaying block max is, though:
  without it, the loss is 0.0010 worse.
- **It holds at 3B and on a second model family,** 3 seeds each, every row at its own tuned learning rate:

| Model | Setup | Learning rate | Held-out loss, seeds 0 / 1 / 2 | Mean | Peak GPU |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B | **No momentum, 2.25 bytes, streamed** | 5.66e-6 | 0.1503 / 0.1509 / 0.1517 | **0.1510** | **14.9 GB** |
| | 3-byte kernel, streamed | 4e-6 | | 0.1516 | 17.8 GB |
| | Flash-BlockMax, accumulated (checkpointing) | | | 0.1526 | 20.0 GB |
| | FlashAdamW, gradient release (checkpointing) | | | 0.1533 | 16.7 GB |
| SmolLM2-1.7B | **No momentum, 2.25 bytes, streamed** | 3.2e-5 | 0.1848 / 0.1847 / 0.1848 | **0.1848** | **8.6 GB** |
| | 3-byte kernel, streamed | 1.6e-5 | 0.1862 / 0.1867 / 0.1860 | 0.1863 | 10.2 GB |
| | Flash-BlockMax, accumulated | 3.96e-5 | 0.1873 / 0.1882 / 0.1883 | 0.1879 | about 13.5 GB |
| | FlashOptim FlashAdamW | 1.41e-5 | 0.1883 / 0.1880 / 0.1889 | 0.1884 | about 15.3 GB |

- **On both, it is the best setup and the smallest:** on SmolLM2 it beats FlashAdamW by 0.0036 at every seed, with
  about 45% less peak memory.
- **Against AdaLomo** (the closest prior method: streamed, no momentum, a factored per-row and per-column second moment;
  `strange/fused/adalomo_ref.py`, bit-identical to lomo-optim 0.1.1), each at its own bracketed lr, 3 seeds:

| Model | No momentum (ours) | AdaLomo with stochastic rounding | AdaLomo as published (round to nearest) |
|---|---|---|---|
| Qwen2.5-Coder-1.5B | 0.1191 (0.1190 / 0.1190 / 0.1192) | 0.1192 (0.1189 / 0.1193 / 0.1193), lr 1.56e-5 | 0.1195, lr 1.25e-4 |
| SmolLM2-1.7B | **0.1848** (0.1848 / 0.1847 / 0.1848) | 0.1855 (0.1850 / 0.1852 / 0.1863), lr 8.84e-5 | 0.1875, lr 2.5e-4 |

  Most of the gap to AdaLomo as published is the rounding: with stochastic rounding added, AdaLomo ties on Qwen and is
  0.0007 behind on SmolLM2, at the same memory (8.3 GB). Stochastic-rounded bf16 streaming is the key ingredient; the
  block-max scale is as good as AdaLomo's factored one or slightly better, and simpler.
- **7B on one 24 GB GPU:** with gradient checkpointing (`--grad_ckpt`), a full fine-tune of Qwen2.5-Coder-7B peaks at
  19.3 GB, about 25 minutes per run on an A10G. At a tuned learning rate of 8e-6, held-out loss goes from 0.2759 to
  0.1512 / 0.1532 / 0.1533 (mean 0.1526). No other optimizer fits at 7B on this GPU, so there is no comparison. The 7B
  starts much worse on this held-out set than the 3B (0.2759 vs 0.1985), so the end losses of the two can't be compared
  directly.

## 3-byte kernel with streamed updates

When memory is the constraint, use `strange/fused/flashbm3_triton.py` with streamed updates. Each weight is bf16 plus one
byte, which holds a 4-bit weight correction and a 4-bit momentum stored relative to its block's scale. Both halves are
rounded stochastically.

Each tensor is updated as soon as its gradient is complete, and the gradient is then freed, so no gradient memory is held
across a step. Streaming means one update per program (no accumulation). Set the per-update decay rates so their memory
spans the same number of programs: with 8 programs per accumulated step, `b1 = 0.9 ** (1/8)` and `b2 = 0.95 ** (1/8)`.

Qwen2.5-Coder-1.5B, block 16, 3 seeds, each at its tuned learning rate:

| Setup | Learning rate | Held-out loss | Peak GPU |
|---|---|---|---|
| Flash-BlockMax, accumulated | 3.5e-6 | 0.1188 | 14.4 GB |
| 3-byte kernel, accumulated | 2.47e-6 | 0.1190 | 13.0 GB |
| **3-byte kernel, streamed** | 1.41e-6 | **0.1190** | **10.1 GB** |
| Flash-BlockMax, streamed | 1.41e-6 | 0.1274 | 11.6 GB |
| FlashOptim FlashAdamW | 1.25e-6 | 0.1194 | 15.7 GB |

- **Don't stream Flash-BlockMax.** Its deterministically rounded correction (FlashOptim's format) loses the small
  per-program steps: 0.0080 worse, and noisy across seeds. The 3-byte kernel rounds stochastically and loses about
  0.0002.
- **Don't stream the original 32-bit kernel either.** Its weights are fine, but its 8-bit momentum is rounded to the
  nearest level and stalls when streamed updates move it by only about 1.3% (+0.0029 at 1.5B, +0.0072 at 3B). The rule:
  everything stored at low precision needs stochastic rounding when streaming.
- **At 3B (Qwen2.5-Coder-3B), the 3-byte streamed kernel was the best setup at every seed:** 0.1516 at 17.8 GB, without
  gradient checkpointing, at a learning rate of 4e-6. Flash-BlockMax accumulated gave 0.1526 (20.0 GB, needs
  checkpointing), and FlashAdamW with gradient release gave 0.1533 (16.7 GB, needs checkpointing).
- **How to stream:** register `p.register_post_accumulate_grad_hook(hook)` for every parameter. In the hook, set the
  group's learning rate, call `opt.step_param(p)`, then set `p.grad = None`. Reference implementation:
  `rsi/fullft_sched9.py --mode flashbm3 --stream --b1 0.98692 --b2 0.99361`.
- **Checked:** the momentum is stored without bias and steps of 1/100 bf16 ulp accumulate exactly
  (`remote/test_fbm3.py`).

## What the optimizer is

Packed BlockMax divides momentum by one shared scale per block of 16 weights: a slowly decaying maximum of |gradient| over
the block. AdamW keeps a second moment per weight; this keeps one number per 16 weights. Momentum is stored as 8 bits (a
sign and a quarter-octave exponent).

| Version | Weights | Momentum | Gradients | Bytes per weight (weights + optimizer) | File |
|---|---|---|---|---|---|
| **Flash-BlockMax (recommended)** | bf16 + int8 correction (FlashOptim's 24-bit format) | separate int8 | bf16 | 4 | `strange/fused/flashbm_triton.py` |
| Packed BlockMax 32-bit | fp32, momentum packed into its low 8 bits | inside the weight | fp32 | 4.25 | `strange/fused/packedsched_triton.py` |
| Packed BlockMax 16-bit | bf16, stochastic rounding | separate int8 | bf16 | about 3 | `strange/fused/packedbf16_triton.py` |

All three are one fused Triton kernel each. They share an interface: `CandidateOptimizer`, plus module globals `SCHEDULE`
and `INHERIT` for the (not recommended) block-size schedule.

## How to use it

```python
import sys, torch
sys.path.insert(0, "strange/fused")          # path to the repo's strange/fused
import flashbm_triton as FB

model = model.to(torch.bfloat16)             # bf16 weights; the kernel asserts bf16 and contiguous
FB.SCHEDULE = [(0, 16)]                      # block 16 throughout (the default)
opt = FB.CandidateOptimizer(model.parameters(), lr=peak_lr, weight_decay=0.0)   # betas default to (0.9, 0.95)

for step in range(total_steps):
    ...                                      # forward and backward, accumulating gradients as usual
    for g in opt.param_groups:
        g["lr"] = peak_lr * max(0.0, 1.0 - step / total_steps)   # linear decay to 0: essential
    opt.step()
    opt.zero_grad(set_to_none=True)
```

Reference script, which trains and measures held-out loss:

```
python rsi/fullft_sched9.py --mode flashbm --sched 16 --chunked_loss --decay --lr <peak lr> \
    --model <hf model id> --data <jsonl of {pid, code}> --out models/<name>
```

## Tune the learning rate for each model

This is the single most important setting. The best peak learning rates found, with 8 programs per step:

| Model | Flash-BlockMax, block 16 | FlashAdamW (for reference) |
|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 3.5e-6 (32-bit Packed BlockMax value; the same rule) | 1.25e-6 |
| SmolLM2-1.7B-Instruct | 4e-5 | 1.4e-5 |

- **Run a grid with about 1.4× steps** around a first guess. Extend it while the best rate is at an edge, then pick the
  best. SmolLM2 needed a rate about 10 times higher than Qwen.
- **The best rate is usually about 3 times FlashAdamW's** on the same model.
- **Retune whenever the block size or schedule changes,** because larger blocks shrink the steps. Comparing setups at one
  shared learning rate gave a false 0.0057 "gain" for the schedule in this project.

## Requirements and gotchas

- **Use a learning-rate decay to 0.** At 1.5B it lowered held-out loss by about 0.019 and removed most of the
  seed-to-seed noise.
- **Resuming** needs the optimizer state as well as the weights. For Flash-BlockMax that's `state["c"]` (int8
  correction), `state["m"]` (int8 momentum) and `state["v"]` (block scales). For the 32-bit version, the momentum lives
  in the fp32 weights, so save them as fp32.
- **Full-precision weights:** `flashbm_triton.full_precision(p, opt.state[p])` returns the 24-bit weight as fp32, for
  a full-precision checkpoint. Saving the bf16 parameters alone drops the correction, which is fine for inference.
- **Long sequences and large vocabularies:** computing the loss in chunks, without materialising full-vocabulary logits
  (`--chunked_loss`, gives the same value), removes the biggest memory spike.
- **First use of each block size is slow,** because the kernel compiles once, about 2 s.

## Results

Full fine-tunes on 1865 verified Python solutions, 234 optimizer steps (8 programs per step), learning rate decayed to 0.
Held-out loss is on 288 programs from 50 problems never trained on. Every row is at its own tuned learning rate, with 3
seeds.

| Model | Optimizer | Learning rate | Held-out loss, seeds 0 / 1 / 2 | Mean | vs FlashAdamW | Peak GPU |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B (0.1466 before) | Packed BlockMax 32-bit, block 16 | 3.54e-6 | 0.1188 / 0.1187 / 0.1191 | 0.1189 | −0.0005 | about 20 GB |
| | FlashOptim FlashAdamW | 1.25e-6 | 0.1194 / 0.1193 / 0.1194 | 0.1194 | — | 15.7 GB |
| SmolLM2-1.7B (0.2233 before) | **Flash-BlockMax, block 16** | 3.96e-5 | 0.1873 / 0.1882 / 0.1883 | 0.1879 | −0.0005 | about 13.5 GB |
| | FlashOptim FlashAdamW | 1.41e-5 | 0.1883 / 0.1880 / 0.1889 | 0.1884 | — | about 15.3 GB |

Flash-BlockMax with block 16 was not run on Qwen. On Qwen with the 4096 schedule it matched the 32-bit version exactly
(0.1185 both), so the 32-bit block-16 row stands in for it.

Ablations on Qwen, from `rsi/abl.py`:
- **Packing the momentum costs 0.0002** of held-out loss and saves a full fp32 momentum copy.
- **The block scale's statistic doesn't matter:** Adam-mini's mean of squares matches the max once its learning rate is
  tuned (about 16 times lower).
- **β₂ 0.95–0.99 is safe;** 0.9 is slightly worse.

## The block-size schedule: not recommended

Starting with blocks of 4096 weights and halving down to 16, with half the run at blocks of 1024 or larger:

| Setting | vs block 16 at its own tuned learning rate |
|---|---|
| Qwen2.5-Coder-1.5B fine-tune | −0.0004 (better at every seed) |
| SmolLM2-1.7B fine-tune | +0.0017 (worse at every seed) |
| From scratch, 25M | +0.0365 |
| From scratch, 152M | +0.0095 |

If you use it anyway:
- **Always copy the parent block's scale at a switch** (`INHERIT = True`). Resetting it diverged at every seed.
- **Expect to need a peak learning rate 2–3 times higher** than for block 16.

## Caveats

- **Scope:** two models (1.5–1.7B) and one task (Python code supervised fine-tuning), with a short fine-tune.
- **Margins:** the gains over FlashAdamW are small (about 0.0005) and not significant at every seed on SmolLM2.
- **Pass rates** (300 problems × 8 samples) never separated any of these setups.
