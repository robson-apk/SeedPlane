#!/usr/bin/env python3
"""Compare persistent direct stdio/SSH workers with HIVE pull/TCP on one fleet.

Exploratory by default (6 x 16 tokens). Increase requests/tokens/rounds for a
formal run. SSH is used only to launch/stop remote HIVE agents; request payloads
and results travel through the persistent authenticated HIVE TCP connection.
"""
import argparse
import concurrent.futures
import json
import math
import os
import platform
import secrets
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from seedplane.hive import HiveBroker
from experiments.trio_v27 import benchmark_trio as direct


MODEL_ID = "qwen05-sp"
PROMPT = direct.PROMPT


def parse_spec(value, label):
    try:
        name, mode, target, command = value.split("|", 3)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} format: NAME|local|_|COMMAND or NAME|ssh|HOST|COMMAND") from exc
    if mode not in ("local", "ssh"):
        raise argparse.ArgumentTypeError(f"{label} mode must be local or ssh")
    return name, mode, target, command


def ssh_argv(target, command):
    user, host = target.rsplit("@", 1) if "@" in target else ("", target)
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if " " in user:
        argv.extend(["-o", f'User="{user}"'])
    argv.extend([host if user else target, command])
    return argv


def launch_agent(spec, broker_host, broker_port, authkey, repo_root):
    name, mode, target, command = spec
    rendered = command.format(BROKER=f"{broker_host}:{broker_port}", AUTHKEY=authkey, WORKER_ID=name)
    env = dict(os.environ)
    env["SEEDPLANE_AUTHKEY"] = authkey
    if mode == "local":
        env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        argv = shlex.split(rendered)
    else:
        argv = ssh_argv(target, rendered)
    proc = subprocess.Popen(argv, cwd=str(repo_root) if mode == "local" else None,
                            env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return name, proc


def pull_request(tokens):
    return {"text": PROMPT, "max_new_tokens": tokens, "temperature": 0,
            "top_k": 0, "top_p": 1, "seed": 1, "stop": []}


def pull_workload(broker, count, tokens, interval):
    origin = time.perf_counter()
    rows = [None] * count
    futures = []
    for request_id in range(count):
        arrival = request_id * interval
        while time.perf_counter() - origin < arrival:
            time.sleep(min(0.005, arrival - (time.perf_counter() - origin)))
        future = broker.submit(pull_request(tokens), model=MODEL_ID,
                               required_caps=("throughput.generate",),
                               kind="throughput.generate", token_budget=tokens,
                               movable=True, stealable=True)
        futures.append(future)

        def completed(done, rid=request_id, arrival_s=arrival, ticket=future):
            finished = time.perf_counter()
            try:
                result = done.result()
                rows[rid] = {"request_id": rid, "worker": ticket.worker_id,
                             "arrival_s": arrival_s, "completion_s": finished - origin,
                             "completion_from_arrival_s": finished - origin - arrival_s,
                             "generated": result["stats"]["generated"],
                             "decode_tok_s": result["stats"]["decode_tok_s"],
                             "token_sha256": result["token_sha256"], "error": None}
            except Exception as exc:
                rows[rid] = {"request_id": rid, "worker": ticket.worker_id,
                             "error": f"{type(exc).__name__}: {exc}"}

        future.add_done_callback(completed)
    for future in futures:
        future.result(timeout=600)
    elapsed = max(row["completion_s"] for row in rows)
    latencies = sorted(row["completion_from_arrival_s"] for row in rows)
    def percentile(p):
        return latencies[max(0, math.ceil(p * len(latencies)) - 1)]
    return {"requests": count, "requested_tokens": count * tokens,
            "generated_tokens": sum(row["generated"] for row in rows),
            "elapsed_s": elapsed,
            "aggregate_tokens_per_second": sum(row["generated"] for row in rows) / elapsed,
            "arrival_to_completion_s": {f"p{int(p * 100)}": percentile(p) for p in (.50, .95, .99)},
            "assignments": {name: sum(row["worker"] == name for row in rows)
                            for name in sorted({row["worker"] for row in rows})},
            "all_request_outputs_identical": len({row["token_sha256"] for row in rows}) == 1,
            "per_request": rows}


def wait_for_workers(broker, expected, processes, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = broker.workers()
        if set(current) == set(expected):
            return current
        exited = {name: proc.poll() for name, proc in processes.items() if proc.poll() is not None}
        if exited:
            raise RuntimeError(f"agent launch/control session exited: {exited}; connected={sorted(current)}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for agents {expected}; connected={sorted(broker.workers())}")


def run_direct(specs, count, tokens):
    direct.COUNT, direct.MAX_NEW = count, tokens
    workers = {}
    try:
        for name, mode, target, command in specs:
            workers[name] = direct.Worker(name, mode, target, command)
        warmups = {name: worker.generate(-1, 0.0, time.perf_counter())
                   for name, worker in workers.items()}
        if len({x["token_sha256"] for x in warmups.values()}) != 1:
            raise RuntimeError("direct path warm-up token hashes differ")
        matrix = {}
        for pattern, interval in (("burst", 0.0), ("steady_0.3s", 0.3)):
            matrix[pattern] = direct.run_workload(list(workers.values()), interval, 0, 0)
        return {"warmup_token_sha256": next(iter(warmups.values()))["token_sha256"],
                "matrix": matrix}
    finally:
        for worker in workers.values():
            worker.close()


def run_pull(agent_specs, args, repo_root):
    authkey = secrets.token_hex(32)
    broker = HiveBroker(host="0.0.0.0", port=0, authkey=authkey, pull_poll_s=0.01).start()
    processes = {}
    try:
        for spec in agent_specs:
            name, proc = launch_agent(spec, args.broker_ip, broker.port, authkey, repo_root)
            processes[name] = proc
        workers = wait_for_workers(broker, sorted(processes), processes)
        warmups = {}
        for name in sorted(processes):
            result = broker.submit(pull_request(tokens=args.tokens), model=MODEL_ID,
                                   required_caps=("generate", f"island.{name.lower()}"),
                                   kind="throughput.warmup", token_budget=args.tokens,
                                   timeout=300).result(timeout=300)
            warmups[name] = result["token_sha256"]
        if len(set(warmups.values())) != 1:
            raise RuntimeError(f"pull agent warm-up token hashes differ: {warmups}")
        matrix = {}
        for pattern, interval in (("burst", 0.0), ("steady_0.3s", 0.3)):
            matrix[pattern] = pull_workload(broker, args.requests, args.tokens, interval)
        return {"warmup_token_sha256": next(iter(warmups.values())),
                "workers": workers, "matrix": matrix}
    finally:
        broker.close()
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="append", required=True,
                        help="direct baseline: NAME|local|_|qwen_vk BUNDLE --serve [args]")
    parser.add_argument("--agent", action="append", required=True,
                        help="HIVE command: NAME|local|_|COMMAND or NAME|ssh|HOST|COMMAND; placeholders {BROKER}, {AUTHKEY}, {WORKER_ID}")
    parser.add_argument("--broker-ip", required=True, help="LAN IP reachable from remote agents")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=1,
                        help="paired rounds; path order alternates to reduce thermal/order bias")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.tokens < 1 or args.rounds < 1:
        parser.error("requests, tokens, and rounds must be positive")
    direct_specs = [parse_spec(x, "worker") for x in args.worker]
    agent_specs = [parse_spec(x, "agent") for x in args.agent]
    expected = {"B580", "RX570", "M4"}
    if {x[0] for x in direct_specs} != expected or {x[0] for x in agent_specs} != expected:
        parser.error("provide exactly B580, RX570, and M4 for both --worker and --agent")
    if args.output.exists():
        parser.error(f"output exists; choose a new path: {args.output}")
    repo_root = Path(__file__).resolve().parents[2]
    round_results = []
    conditions = direct.CONDITIONS
    for round_index in range(args.rounds):
        shift = round_index % len(conditions)
        ordered_conditions = conditions[shift:] + conditions[:shift]
        condition_results = {}
        for condition_index, condition in enumerate(ordered_conditions):
            condition_key = "+".join(condition)
            chosen_direct = [spec for spec in direct_specs if spec[0] in condition]
            chosen_agents = [spec for spec in agent_specs if spec[0] in condition]
            pull_first = (round_index + condition_index) % 2 == 1
            if pull_first:
                pull_result = run_pull(chosen_agents, args, repo_root)
                direct_result = run_direct(chosen_direct, args.requests, args.tokens)
                order = ["pull", "direct"]
            else:
                direct_result = run_direct(chosen_direct, args.requests, args.tokens)
                pull_result = run_pull(chosen_agents, args, repo_root)
                order = ["direct", "pull"]
            comparisons = {}
            for pattern in ("burst", "steady_0.3s"):
                old = direct_result["matrix"][pattern]
                new = pull_result["matrix"][pattern]
                old_rows = sorted(old["per_request"], key=lambda x: x["request_id"])
                new_rows = sorted(new["per_request"], key=lambda x: x["request_id"])
                comparisons[pattern] = {
                    "same_token_hashes_by_request": [x["token_sha256"] for x in old_rows] ==
                                                      [x["token_sha256"] for x in new_rows],
                    "pull_vs_direct_throughput_ratio": new["aggregate_tokens_per_second"] /
                                                        old["aggregate_tokens_per_second"],
                }
            condition_results[condition_key] = {"path_order": order,
                                                "direct": direct_result, "pull": pull_result,
                                                "comparisons": comparisons}
        round_results.append({"round": round_index + 1,
                              "condition_order": list(condition_key for condition_key in condition_results),
                              "conditions": condition_results})
    summary = {}
    condition_keys = ["+".join(condition) for condition in conditions]
    for condition_key in condition_keys:
        summary[condition_key] = {}
        for pattern in ("burst", "steady_0.3s"):
            paired = [row["conditions"][condition_key] for row in round_results]
            old = [row["direct"]["matrix"][pattern] for row in paired]
            new = [row["pull"]["matrix"][pattern] for row in paired]
            summary[condition_key][pattern] = {
                "direct_median_tok_s": statistics.median(x["aggregate_tokens_per_second"] for x in old),
                "pull_median_tok_s": statistics.median(x["aggregate_tokens_per_second"] for x in new),
                "pull_direct_ratio_median": statistics.median(row["comparisons"][pattern]["pull_vs_direct_throughput_ratio"]
                                                               for row in paired),
                "direct_median_p95_s": statistics.median(x["arrival_to_completion_s"]["p95"] for x in old),
                "pull_median_p95_s": statistics.median(x["arrival_to_completion_s"]["p95"] for x in new),
                "all_round_token_hashes_match": all(row["comparisons"][pattern]["same_token_hashes_by_request"]
                                                     for row in paired),
            }
    for pattern in ("burst", "steady_0.3s"):
        for lane in ("direct", "pull"):
            ratios = []
            for row in round_results:
                trio_rate = row["conditions"]["B580+RX570+M4"][lane]["matrix"][pattern]["aggregate_tokens_per_second"]
                isolated = sum(row["conditions"][name][lane]["matrix"][pattern]["aggregate_tokens_per_second"]
                               for name in ("B580", "RX570", "M4"))
                ratios.append(trio_rate / isolated)
            summary["B580+RX570+M4"][pattern][f"fleet_efficiency_{lane}_median"] = statistics.median(ratios)
        common_denominator_ratios = []
        for row in round_results:
            trio_pull = row["conditions"]["B580+RX570+M4"]["pull"]["matrix"][pattern]["aggregate_tokens_per_second"]
            direct_isolated = sum(row["conditions"][name]["direct"]["matrix"][pattern]["aggregate_tokens_per_second"]
                                  for name in ("B580", "RX570", "M4"))
            common_denominator_ratios.append(trio_pull / direct_isolated)
        summary["B580+RX570+M4"][pattern]["fleet_efficiency_pull_vs_direct_isolated_median"] = statistics.median(common_denominator_ratios)
    result = {"protocol": "HIVE pull framed TCP v1 vs persistent JSON-lines control stream",
              "recorded_at": time.time(), "controller_platform": platform.platform(),
              "model_id": MODEL_ID, "workload": {"prompt": PROMPT, "requests": args.requests,
                          "tokens_per_request": args.tokens, "rounds": args.rounds, "greedy": True,
                          "patterns": {"burst": "all requests queued at start",
                                       "steady_0.3s": "arrivals every 0.3 seconds"}},
              "conditions": condition_keys, "summary": summary, "rounds": round_results,
              "qualification": "exploratory; short requests/sample count; no latency SLA gate"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary,
                      "condition_orders": [row["condition_order"] for row in round_results]}, indent=2))
    if not all(summary[condition][pattern]["all_round_token_hashes_match"]
               for condition in condition_keys for pattern in ("burst", "steady_0.3s")):
        raise SystemExit("token identity mismatch between direct and HIVE pull paths")


if __name__ == "__main__":
    main()
