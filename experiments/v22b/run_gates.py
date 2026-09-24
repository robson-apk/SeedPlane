"""Execute the pre-registered V22b G1-G3 gates without changing their criteria.

Usage: python run_gates.py <isolated-sp-v22b> <reference-sp-v19>
The isolated root supplies native/build/qwen_vk.exe. Reference fixtures are read-only.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

root, refroot = map(Path, sys.argv[1:3])
exe = root / "native" / "build" / "qwen_vk.exe"
bundle = refroot / "run" / "qwen05_v1.sp"
data = refroot / "v20" / "data"
vectors = refroot / "v21"
work = root / "v22b"
work.mkdir(exist_ok=True)
result_path = work / "gates.json"
res = {"protocol": "experiments/v22b/PROTOCOL.md", "binary": str(exe)}


def save():
    result_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


def reference(logits, temperature, top_k, top_p):
    z = logits.astype(np.float64) / temperature
    if top_k > 0:
        cutoff = np.sort(z)[::-1][top_k - 1]
        z = np.where(z < cutoff, -np.inf, z)
    if top_p < 1.0:
        order = np.argsort(-z, kind="stable")
        q = np.exp(z[order] - z[order][0])
        q /= q.sum()
        z[order[np.cumsum(q) - q > top_p]] = -np.inf
    q = np.exp(z - np.max(z))
    return q / q.sum()


def run(args):
    proc = subprocess.run([str(exe), str(bundle), *map(str, args)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


# G1: 8 vectors × all 6 pre-registered configurations × 1,000,000 draws.
sources = ["2999", "3500", "4020", "1000", "2000", "3000", "synthetic_s2", "synthetic_s4"]
cases = [(0.7, 40, 0.9), (1.0, 0, 1.0), (0.7, 0, 0.5),
         (0.7, 0, 0.99), (0.7, 40, 1.0), (0.7, 200, 1.0)]
g1 = []
for source in sources:
    path = vectors / f"logits_{source}.f32"
    logits = np.fromfile(path, np.float32)
    if logits.size == 0:
        raise FileNotFoundError(path)
    for temperature, top_k, top_p in cases:
        counts_path = work / f"counts_{source}_{temperature}_{top_k}_{top_p}.u32"
        n = 1_000_000
        info = run(["--sample-test-gpu", path, temperature, top_k, top_p, n, 11, counts_path])
        counts = np.fromfile(counts_path, np.uint32)
        expected = reference(logits, temperature, top_k, top_p)
        empirical = counts.astype(np.float64) / n
        if temperature == 1.0 and top_k == 0 and top_p == 1.0:
            tail = expected < 1e-4
            tv = 0.5 * (np.abs(empirical[~tail] - expected[~tail]).sum() +
                        abs(empirical[tail].sum() - expected[tail].sum()))
        else:
            tv = 0.5 * np.abs(empirical - expected).sum()
        row = {"logits": source, "temperature": temperature, "top_k": top_k, "top_p": top_p,
               "path": info["path"], "candidates": info["candidates"],
               "support": int((expected > 0).sum()), "outside_draws": int(counts[expected == 0].sum()), "tv": float(tv)}
        g1.append(row)
        res["G1"] = {"cases": g1, "fallbacks": sum(x["path"] == "fallback" for x in g1),
                     "max_tv": max(x["tv"] for x in g1),
                     "pass": all(x["outside_draws"] == 0 and x["tv"] <= 0.01 for x in g1)}
        save()
        print("G1", source, temperature, top_k, top_p, row["path"], "TV", round(tv, 6), flush=True)

# G2: five configurations, three interleaved rounds, 1,024 generated tokens per run.
cfgs = {
    "greedy": [],
    "T1": ["--temperature", "1.0"],
    "k40p0.9": ["--temperature", "0.7", "--top-k", "40", "--top-p", "0.9"],
    "p0.9": ["--temperature", "0.7", "--top-p", "0.9"],
    "p0.99": ["--temperature", "0.7", "--top-p", "0.99"],
    "k200": ["--temperature", "0.7", "--top-k", "200"],
}
runs = {name: [] for name in cfgs}
for round_index in range(3):
    for name, args in cfgs.items():
        run_data = run(["--prompt-file", data / "speed_prompt.i32", "-n", 1024, "--runs", 1,
                        "--seed", 3, *args])["runs"][0]
        runs[name].append(run_data)
        print("G2", round_index + 1, name, run_data["decode_tok_s"], flush=True)
greedy = statistics.median(x["decode_tok_s"] for x in runs["greedy"])
medians = {name: statistics.median(x["decode_tok_s"] for x in rows) for name, rows in runs.items()}
ratios = {name: medians[name] / greedy for name in cfgs if name != "greedy"}
res["G2"] = {"tok_s": {name: [x["decode_tok_s"] for x in rows] for name, rows in runs.items()},
             "median": medians, "ratio_vs_greedy": ratios,
             "step_ms": {name: [x["step_ms_mean"] for x in rows] for name, rows in runs.items()},
             "sample_ms": {name: [x["sample_ms_mean"] for x in rows] for name, rows in runs.items()},
             "fallbacks": {name: [x.get("sample_fallbacks", 0) for x in rows] for name, rows in runs.items()},
             "pass": all(value >= 0.95 for value in ratios.values())}
ref_tokens_path = refroot / "v20" / "out" / "speed_shadow-batch.json"
ref_tokens = json.loads(ref_tokens_path.read_text(encoding="utf-8"))["runs"][0]["tokens"]
res["G3"] = {"greedy_tokens_equal_v20_all_rounds": all(x["tokens"] == ref_tokens for x in runs["greedy"])}
res["G3"]["pass"] = res["G3"]["greedy_tokens_equal_v20_all_rounds"]
save()
print(json.dumps({name: value["pass"] for name, value in res.items() if isinstance(value, dict) and "pass" in value}))
