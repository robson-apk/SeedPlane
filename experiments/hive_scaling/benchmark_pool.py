"""Small configurable burst/steady pool harness reusing the V27 stream client."""
import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.trio_v27 import benchmark_trio as base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="append", required=True, type=base.parse_worker)
    ap.add_argument("--condition", action="append", required=True,
                    help="pool membership, plus-separated worker names")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--steady-interval", type=float, default=0.5)
    ap.add_argument("--order-seed", type=int, default=20260924)
    ap.add_argument("--network-note", action="append", default=[])
    args = ap.parse_args()
    workers_spec = args.worker
    names = [row[0] for row in workers_spec]
    if len(names) != len(set(names)):
        ap.error("worker names must be unique")
    conditions = [tuple(part for part in value.split("+") if part) for value in args.condition]
    if any(not condition or len(condition) != len(set(condition)) for condition in conditions):
        ap.error("each condition must contain distinct worker names")
    unknown = sorted({name for condition in conditions for name in condition} - set(names))
    if unknown:
        ap.error(f"conditions reference unknown workers: {unknown}")
    if args.requests < 2 or args.tokens < 1 or args.rounds < 1 or args.steady_interval < 0:
        ap.error("requests >= 2, tokens/rounds > 0 and steady interval >= 0 required")
    if args.output.exists():
        ap.error(f"output exists; choose a fresh path: {args.output}")

    base.PROMPT = "Explain briefly how a computer memory cache improves performance."
    base.COUNT = args.requests
    base.MAX_NEW = args.tokens
    worker_map = {}
    try:
        for name, mode, target, command in workers_spec:
            worker_map[name] = base.Worker(name, mode, target, command)
        warmups = {name: worker.generate(-1, 0.0, time.perf_counter())
                   for name, worker in worker_map.items()}
        conditions_with_ids = list(enumerate(conditions))
        matrix = []
        for round_index in range(args.rounds):
            order = conditions_with_ids[:]
            random.Random(args.order_seed + round_index).shuffle(order)
            for condition_index, condition in order:
                pool = [worker_map[name] for name in condition]
                for arrival_name, interval in (("burst", 0.0),
                                               ("steady", args.steady_interval)):
                    result = base.run_workload(pool, interval, round_index, condition_index)
                    result.update({
                        "condition": "+".join(condition),
                        "round": round_index + 1,
                        "arrival_pattern": arrival_name,
                        "outputs_match_within_pool": result["all_request_outputs_identical_within_condition"],
                    })
                    matrix.append(result)
        try:
            source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=ROOT, text=True,
                                                    stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            source_commit = "working-tree snapshot (not a git checkout)"
        payload = {
            "protocol": "Hive local-CPU-GPU-pool exploratory v1",
            "source_commit": source_commit,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "controller_platform": __import__("platform").platform(),
            "kind": "aggregate throughput over independent requests; not collaborative single-request decode",
            "workload": {"prompt": base.PROMPT, "requests": args.requests,
                         "tokens_per_request": args.tokens, "temperature": 0,
                         "top_k": 0, "top_p": 1,
                         "steady_interarrival_s": args.steady_interval,
                         "rounds": args.rounds,
                         "percentile_method": "nearest rank"},
            "network_notes": args.network_note,
            "warmups": {name: {"ready": worker_map[name].ready,
                               "token_sha256": row["token_sha256"]}
                        for name, row in warmups.items()},
            "warmup_hashes_all_equal": len({row["token_sha256"] for row in warmups.values()}) == 1,
            "workers": {name: {"mode": worker.mode, "target": worker.target,
                               "launch_command": worker.command,
                               "ready": worker.ready}
                        for name, worker in worker_map.items()},
            "matrix": matrix,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(json.dumps({"output": str(args.output), "measurements": len(matrix),
                          "warmup_hashes_all_equal": payload["warmup_hashes_all_equal"],
                          "pool_output_mismatches": sum(not row["outputs_match_within_pool"]
                                                        for row in matrix)}, indent=2))
    finally:
        for worker in worker_map.values():
            worker.close()


if __name__ == "__main__":
    main()
