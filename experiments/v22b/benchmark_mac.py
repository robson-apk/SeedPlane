"""Run the pre-registered V22b G2 matrix on the local macOS Vulkan/MoltenVK device."""
import json
import statistics
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
exe = repo / "native" / "vulkan_decode" / "build_codex_v22b" / "qwen_vk"
shared = Path("/Volumes/10.0.0.146/Users/Windows 11/sp_v19")
bundle = shared / "run" / "qwen05_v1.sp"
prompt = shared / "v20" / "data" / "speed_prompt.i32"
cfgs = {
    "greedy": [],
    "T1": ["--temperature", "1.0"],
    "k40p0.9": ["--temperature", "0.7", "--top-k", "40", "--top-p", "0.9"],
    "p0.9": ["--temperature", "0.7", "--top-p", "0.9"],
    "p0.99": ["--temperature", "0.7", "--top-p", "0.99"],
    "k200": ["--temperature", "0.7", "--top-k", "200"],
}
results = {key: [] for key in cfgs}
out = repo / "experiments" / "v22b" / "mac_g2.json"
for round_index in range(3):
    for name, args in cfgs.items():
        proc = subprocess.run([str(exe), str(bundle), "--prompt-file", str(prompt), "-n", "1024", "--runs", "1",
                               "--seed", "3", *args], capture_output=True, text=True, check=True)
        results[name].append(json.loads(proc.stdout)["runs"][0])
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(round_index + 1, name, results[name][-1]["decode_tok_s"], flush=True)
greedy = statistics.median(row["decode_tok_s"] for row in results["greedy"])
summary = {name: {"median_tok_s": statistics.median(row["decode_tok_s"] for row in rows),
                  "ratio_vs_greedy": statistics.median(row["decode_tok_s"] for row in rows) / greedy,
                  "step_ms": [row["step_ms_mean"] for row in rows], "fallbacks": [row.get("sample_fallbacks", 0) for row in rows]}
           for name, rows in results.items()}
final = {"device": "Apple M4 / MoltenVK", "runs": results, "summary": summary}
out.write_text(json.dumps(final, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
