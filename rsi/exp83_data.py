#!/usr/bin/env python3
"""Experiment 83, the data: a training pool with verified tests, and a contamination-free test set.

TRAIN / VAL   DeepMind CodeContests (Codeforces, AtCoder, CodeChef ... up to 2021), stdin/stdout problems only. A problem
              is kept only if one of its own PYTHON3 reference solutions passes the tests we select, in our sandbox,
              under our limits: that certifies the tests are sane and the problem is solvable in Python in the time
              allowed, so a failure later is the model's. The reference solutions are NOT kept: the loop trains on
              nothing but the model's own verified programs. Up to 12 tests per problem (public first, then private,
              then generated; inputs under 10 KB). Difficulty band from the Codeforces rating.
TEST          LiveCodeBench code_generation_lite, problems dated on or after --test_from (default 2024-10-01, after
              Qwen2.5-Coder's data; its own report treats 2024-07..09 as unseen). All tests kept. Never used for any
              decision inside the loop.
Both use LiveCodeBench's prompt format, so the model sees one format throughout.
"""
import argparse
import base64
import io
import json
import os
import pickle
import random
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "exp83")
SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct "
          "Python program that matches the specification and passes all tests.")
FMT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
             "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm "
             "and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
FMT_FUNC = "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n{starter}\n```\n\n"


def messages(p):
    fmt = FMT_FUNC.format(starter=p["starter"]) if p["kind"] == "functional" else FMT_STDIN
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"### Question:\n{p['question']}\n\n{fmt}### Answer: (use the provided format with backticks)\n\n"}]


def band(rating):
    return "U" if not rating else "A" if rating <= 1000 else "B" if rating <= 1400 else "C" if rating <= 1800 else "D"


def build_train(args):
    from datasets import load_dataset
    import exp83_sandbox as SB
    ds = load_dataset("deepmind/code_contests", split="train", streaming=True).shuffle(seed=args.seed, buffer_size=1000)
    cands = []
    for ex in ds:
        if ex.get("input_file") or ex.get("output_file") or len(ex["description"]) > 5000:
            continue
        refs = sorted((s for l, s in zip(ex["solutions"]["language"], ex["solutions"]["solution"]) if l == 3 and len(s) < 4000), key=len)[:3]
        if not refs:
            continue
        tests = []
        for grp in ("public_tests", "private_tests", "generated_tests"):
            for i, o in zip(ex[grp]["input"], ex[grp]["output"]):
                if len(i) <= 10000 and len(o) <= 10000 and len(tests) < 12:
                    tests.append([i, o])
        if len(tests) < 6:
            continue
        cands.append({"pid": ex["name"], "kind": "stdin", "question": ex["description"], "starter": "", "tests": tests, "band": band(ex.get("cf_rating")),
                      "rating": ex.get("cf_rating") or 0, "source": ex.get("source"), "_refs": refs})
        if len(cands) % 500 == 0:
            print(f"  {len(cands)} candidates", flush=True)
        if len(cands) >= args.candidates:
            break
    print(f"verifying {len(cands)} candidates with their own Python reference solutions ...", flush=True)
    kept = []
    for r in range(3):                                           # try the shortest reference first, then the next ...
        todo = [c for c in cands if not c.get("_ok") and len(c["_refs"]) > r]
        vs = SB.judge_many([(c["_refs"][r], c) for c in todo], limit=args.limit)
        for c, v in zip(todo, vs):
            if v["passed"]:
                c["_ok"] = True
        print(f"  reference {r}: {sum(1 for c in cands if c.get('_ok'))} verified", flush=True)
    kept = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands if c.get("_ok")]
    random.Random(args.seed).shuffle(kept)
    val, train = kept[:args.n_val], kept[args.n_val:args.n_val + args.n_train]
    os.makedirs(DATA, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(DATA, f"{name}.jsonl"), "w") as fh:
            for p in rows:
                fh.write(json.dumps(p) + "\n")
        print(f"{name}: {len(rows)} problems, bands {dict(sorted(__import__('collections').Counter(p['band'] for p in rows).items()))}")


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):                           # the payload is a pickled str; anything else is refused
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def decode_private(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load())


def build_test(args):
    from huggingface_hub import hf_hub_download
    out = []; seen = set()
    for suffix in ("", "2", "3", "4", "5", "6"):
        try:
            path = hf_hub_download("livecodebench/code_generation_lite", filename=f"test{suffix}.jsonl", repo_type="dataset")
        except Exception as e:
            print(f"  test{suffix}.jsonl: {type(e).__name__}"); continue
        n = 0
        for line in open(path):
            ex = json.loads(line)
            if ex["contest_date"][:10] < args.test_from or ex["question_id"] in seen:
                continue
            seen.add(ex["question_id"])
            public = json.loads(ex["public_test_cases"]); tests = public + decode_private(ex["private_test_cases"])
            kind = "functional" if tests[0]["testtype"] == "functional" else "stdin"
            meta = json.loads(ex["metadata"]) if ex.get("metadata") else {}
            out.append({"pid": ex["question_id"], "kind": kind, "question": ex["question_content"], "starter": ex.get("starter_code", ""), "fn": meta.get("func_name"),
                        "tests": [[t["input"], t["output"]] for t in tests], "n_public": len(public), "band": ex["difficulty"], "date": ex["contest_date"][:10], "platform": ex["platform"]})
            n += 1
        print(f"  test{suffix}.jsonl: {n} problems on or after {args.test_from}", flush=True)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "lcb_test.jsonl"), "w") as fh:
        for p in out:
            fh.write(json.dumps(p) + "\n")
    import collections
    print(f"lcb_test: {len(out)} problems, difficulty {dict(collections.Counter(p['band'] for p in out))}, kinds {dict(collections.Counter(p['kind'] for p in out))}, "
          f"dates {min(p['date'] for p in out)} .. {max(p['date'] for p in out)}")


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, f"{name}.jsonl"))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "test"])
    ap.add_argument("--candidates", type=int, default=6000)
    ap.add_argument("--n_train", type=int, default=2400)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--test_from", default="2024-10-01")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (build_train if a.what == "train" else build_test)(a)
