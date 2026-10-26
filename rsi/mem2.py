"""Follow-ups to rsi/mem.py (chunked loss, lr decayed to 0, held-out loss only, block 16; every arm its own lr grid at seed 0,
then seeds 1 and 2). usage: mem2.py <arm> [<arm> ...]
  flash_stream      Qwen2.5-Coder-1.5B, FlashAdamW with FlashOptim's own gradient release (per-update betas matched to 8 programs)
  fbm3_stream_3b    Qwen2.5-Coder-3B, 3-byte kernel, streamed
  fbm16_3b          Qwen2.5-Coder-3B, Flash-BlockMax accumulated, gradient checkpointing (the 3B reference)
  flash_stream_3b   Qwen2.5-Coder-3B, FlashAdamW with gradient release
  adalomo(_sm)      AdaLomo baseline, Qwen 1.5B (SmolLM2); adalomo_sr(_sm) the same with stochastic rounding of the weights"""
import json, math, os, subprocess, sys
Q15, Q3 = "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"
SM = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
Q7 = "Qwen/Qwen2.5-Coder-7B-Instruct"
FS = ["--mode", "flash", "--stream", "--b1", "0.98692", "--flash_b2", "0.999875"]
BS = ["--mode", "flashbm3", "--stream", "--b1", "0.98692", "--b2", "0.99361"]
ARMS = {"flash_stream": (6e-7, Q15, FS), "fbm3_stream_3b": (1e-6, Q3, BS),
        "fbm16_3b": (2.5e-6, Q3, ["--mode", "flashbm", "--grad_ckpt"]), "flash_stream_3b": (6e-7, Q3, FS),
        "flash_stream_3b_ck": (6e-7, Q3, FS + ["--grad_ckpt"]),
        "pk_stream": (1.41e-6, Q15, ["--mode", "packed", "--stream", "--b1", "0.98692", "--b2", "0.99361"]),      # the original 32-bit
        "pk_stream_3b": (4e-6, Q3, ["--mode", "packed", "--stream", "--b1", "0.98692", "--b2", "0.99361"]),      # Packed BlockMax
        "nomom_a": (5e-7, Q15, ["--mode", "nomom", "--stream"]),                                       # 2 bytes, no state
        "nomom_b": (5e-7, Q15, ["--mode", "nomom", "--stream", "--stateful", "--b2", "0.99361"]),     # 2.25 bytes, decaying max
        "nomom_b_3b": (4e-6, Q3, ["--mode", "nomom", "--stream", "--stateful", "--b2", "0.99361"]),
        "nomom_b_sm": (1.6e-5, SM, ["--mode", "nomom", "--stream", "--stateful", "--b2", "0.99361"]),
        "fbm3_stream_sm": (1.6e-5, SM, BS),                                                       # the SmolLM2 reference
        "nomom_b_7b": (4e-6, Q7, ["--mode", "nomom", "--stream", "--stateful", "--b2", "0.99361", "--grad_ckpt"]),  # 19.3 GB peak
        "adalomo": (5e-4, Q15, ["--mode", "adalomo", "--stream"]),                                    # baseline AdaLomo (relative lr),
        "adalomo_sr": (5e-4, Q15, ["--mode", "adalomo", "--stream", "--adalomo_sr"]),                 # as published / with our SR
        "adalomo_sm": (5e-4, SM, ["--mode", "adalomo", "--stream"]),
        "adalomo_sr_sm": (5e-4, SM, ["--mode", "adalomo", "--stream", "--adalomo_sr"]),
        "pk_stream_3b_ck": (4e-6, Q3, ["--mode", "packed", "--stream", "--b1", "0.98692", "--b2", "0.99361", "--grad_ckpt"])}  # OOMs without             # without checkpointing FlashAdamW ran out of memory at 3B


def run(arm, seed, lr):
    n = f"mem2_{arm}_{lr:.3g}_s{seed}"
    if not os.path.exists(f"models/s_{n}.json"):
        with open(f"results/{n}.out", "w") as fh:
            subprocess.run([sys.executable, "-u", "fullft_sched9.py", "--model", ARMS[arm][1], "--sched", "16",
                            "--chunked_loss", "--decay", "--no_save", "--data", "results/rsi_xs_buffer.jsonl", "--out", f"models/s_{n}",
                            "--seed", str(seed), "--lr", f"{lr:.6g}", *ARMS[arm][2]], stdout=fh, stderr=subprocess.STDOUT)
    try:
        d = json.load(open(f"models/s_{n}.json")); h, tr, sk, pk = d["heldout_after"], d["train_last100"], d.get("skipped"), d.get("peak_gb")
    except Exception:
        h, tr, sk, pk = float("inf"), float("nan"), None, None
    print(f"{arm:16s} lr {lr:.3g} seed {seed}: held-out {h:.5f} train {tr:.5f} skipped {sk} peak {pk if pk is None else round(pk, 1)} GB", flush=True)
    return h


for arm in sys.argv[1:]:
    c = ARMS[arm][0]
    res = {lr: run(arm, 0, lr) for lr in (c, c * 2, c / 2)}
    for _ in range(4):
        best = min(res, key=res.get); lrs = sorted(res)
        if best == lrs[0]:
            res[best / 2] = run(arm, 0, best / 2)
        elif best == lrs[-1]:
            res[best * 2] = run(arm, 0, best * 2)
        else:
            break
    best = min(res, key=res.get)
    for lr in (best / math.sqrt(2), best * math.sqrt(2)):
        res[lr] = run(arm, 0, lr)
    best = min(res, key=res.get)
    print(f"BEST {arm} lr {best:.3g}; grid " + ", ".join(f"{lr:.3g}:{res[lr]:.5f}" for lr in sorted(res)), flush=True)
    for s in (1, 2):
        run(arm, s, best)
