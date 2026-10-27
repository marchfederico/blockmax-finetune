# Full fine-tuning at 2.25 bytes per parameter

Code for streamed, momentum-free BlockMax updates with stochastic rounding. Training state is the bf16 weights plus one
fp32 scale per block of 16 weights (2.25 bytes per parameter); each tensor is updated as soon as its gradient is
complete, and the gradient is freed.

```
v_b <- max(b2 * v_b, max_{i in b} |g_i|)          # one scale per block of 16 weights
w_i <- SR_bf16(w_i - lr * g_i / v_b)               # stochastic rounding to bf16, no momentum
```

## Results

Python-code supervised fine-tuning (1,865 verified programs, one epoch, learning rate decayed to 0), held-out loss on
288 programs from 50 unseen problems. Each method at its own tuned learning rate, mean of 3 seeds, one A10G (24 GB).

| Model | Ours (2.25 B/param) | AdaLomo + stochastic rounding | AdaLomo | FlashAdamW |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | **0.1191** (8.7 GB) | 0.1192 (8.3 GB) | 0.1195 (8.3 GB) | 0.1194 (15.7 GB) |
| Qwen2.5-Coder-3B | **0.1510** (14.9 GB) | | | 0.1533 (16.7 GB, checkpointing) |
| SmolLM2-1.7B | **0.1848** (8.6 GB) | 0.1855 (8.3 GB) | 0.1875 (8.3 GB) | 0.1884 (15.3 GB) |

Qwen2.5-Coder-7B fully fine-tunes on the same GPU at 19.3 GB peak with gradient checkpointing (held-out loss
0.2759 to 0.1526). Full tables, ablations and learning-rate grids: `PACKED_BLOCKMAX_RECIPE.md`.

## Layout

| Path | What |
|---|---|
| `strange/fused/nomom_triton.py` | **The method**: fused Triton kernel, `STATELESS = False` for the decaying block max |
| `strange/fused/flashbm3_triton.py` | 3-byte variant: bf16 + 4-bit stochastic correction + 4-bit momentum |
| `strange/fused/flashbm_triton.py` | Flash-BlockMax: FlashOptim's 24-bit weights with the BlockMax update |
| `strange/fused/packedsched_triton.py` | Packed BlockMax: fp32 weights with an 8-bit momentum in their low bits |
| `strange/fused/packedbf16_triton.py`, `blockvar_triton.py` | 16-bit variant; ablation kernel (momentum, block statistic) |
| `strange/fused/adalomo_ref.py` | AdaLomo baseline, identical to lomo-optim 0.1.1, plus a stochastic-rounding switch |
| `blockmax.py` | Plain PyTorch reference of the BlockMax rule |
| `rsi/fullft_sched9.py` | Training harness: all methods, `--stream`, `--grad_ckpt`, held-out loss |
| `rsi/mem2.py` | Learning-rate grid driver (adaptive grid at seed 0, then seeds 1 and 2) |
| `rsi/exp83_data.py`, `rsi/data/exp83/` | Problem set (from DeepMind CodeContests) and prompt format |
| `rsi/results/rsi_xs_buffer.jsonl` | The 1,865 verified training programs |
| `remote/test_*.py` | Kernel tests against reference implementations |

## Setup

Python 3.11, a CUDA GPU. Tested with torch 2.14, triton 3.8, transformers 5.17.

```bash
pip install -r requirements.txt
pip install --no-deps --target pylib_fo flashoptim==0.1.4     # FlashAdamW and Flash-BlockMax only
pip install --no-deps --target pylib_lomo lomo-optim==0.1.1   # AdaLomo test only
```

## Run

From `rsi/`. Our method on Qwen2.5-Coder-1.5B:

```bash
cd rsi
python fullft_sched9.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --mode nomom --stream --stateful --b2 0.99361 \
    --sched 16 --chunked_loss --decay --no_save --lr 1.41e-6 --data results/rsi_xs_buffer.jsonl --out models/ours
```

Add `--grad_ckpt` for 7B. The held-out loss is written to `models/ours.json`. Full learning-rate grid for one arm
(arms are defined in `mem2.py`):

```bash
python mem2.py nomom_b
```

Tests (from the repository root):

```bash
python remote/test_nomom.py
```

## Using the optimizer

```python
import sys, torch
sys.path.insert(0, "strange/fused")
import nomom_triton as NM

NM.STATELESS = False                                  # keep the decaying block max (2.25 bytes per parameter)
model = model.to(torch.bfloat16)
opt = NM.CandidateOptimizer(model.parameters(), lr=peak_lr, betas=(0.9, 0.99361))
lr_now = peak_lr

def release(p):                                       # update each tensor as its gradient completes, then free it
    for g in opt.param_groups:
        g["lr"] = lr_now
    opt.step_param(p)
    p.grad = None

for p in model.parameters():
    p.register_post_accumulate_grad_hook(release)

for i, batch in enumerate(data):                      # one example per update
    lr_now = peak_lr * (1 - i / len(data))            # linear decay to 0: essential
    model(**batch).loss.backward()                    # the update happens inside backward
```

Tune the learning rate for each model: the best rate differed by about 20 times between Qwen and SmolLM2.

## License

MIT License (see `LICENSE`). `strange/fused/adalomo_ref.py` is derived from lomo-optim (MIT License, Copyright (c) 2023
OpenLMLab); its notice is kept in the file.

### Data

`rsi/data/exp83/` holds problems from DeepMind's [CodeContests](https://huggingface.co/datasets/deepmind/code_contests),
released under CC BY 4.0 (Li et al., 2022, "Competition-Level Code Generation with AlphaCode", arXiv:2203.07814). Problem
statements include material from codeforces.com, Description2Code (MIT License) and CodeNet (Apache License 2.0). The
training programs in `rsi/results/rsi_xs_buffer.jsonl` were generated by a model and verified against those problems' tests.
