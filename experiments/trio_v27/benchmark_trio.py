"""Run the pre-registered V27-T3 independent-request matrix over local stdio/SSH.

Example worker specifications (use fresh run directories and a single verified bundle):
  --worker 'B580|ssh|Windows 11@10.0.0.146|"C:\\...\\qwen_vk.exe" "C:\\...\\qwen05.sp" --serve'
  --worker 'RX570|ssh|robson@10.0.0.253|/home/robson/.../qwen_vk /home/robson/.../qwen05.sp --serve'
  --worker 'M4|local|/tmp/.../qwen_vk /tmp/.../qwen05.sp --serve'

See PROTOCOL.md. This measures independent queued requests, not one request split across devices.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import select
import shlex
import subprocess
import time
from collections import deque
from pathlib import Path


PROMPT = "Explain in one concise paragraph how a computer memory cache improves performance."
COUNT = 24
MAX_NEW = 128
ARRIVAL_INTERVAL_S = 1.0
ROUND_COUNT = 3
CONDITIONS = [
    ("B580",), ("RX570",), ("M4",),
    ("B580", "RX570"), ("B580", "M4"), ("RX570", "M4"),
    ("B580", "RX570", "M4"),
]


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def summarize(values):
    return {f"p{int(p * 100)}": percentile(values, p) for p in (.50, .95, .99)}


def token_digest(tokens):
    payload = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Worker:
    def __init__(self, name, mode, target, command):
        self.name = name
        self.mode = mode
        self.target = target
        self.command = command
        if mode == "local":
            argv = shlex.split(command)
        else:
            user, host = target.rsplit("@", 1) if "@" in target else ("", target)
            argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
            if " " in user:
                argv.extend(["-o", f'User="{user}"'])
            argv.extend([host if user else target, command])
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self.buffer = bytearray()
        self.bytes_sent = 0
        self.bytes_received = 0
        ready = self.read_json(timeout=240)
        if not ready.get("ready"):
            raise RuntimeError(f"{name}: server did not report ready: {ready}")
        self.ready = ready

    def read_json(self, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self.buffer[:newline])
                del self.buffer[:newline + 1]
                self.bytes_received += newline + 1
                return json.loads(raw.decode("utf-8"))
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"{self.name}: timed out waiting for JSON response")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"{self.name}: worker closed stdout (exit={self.proc.poll()})")
            self.buffer.extend(chunk)

    def generate(self, request_id, arrival_rel_s, origin):
        request = {
            "op": "generate", "text": PROMPT, "reset": True,
            "temperature": 0, "top_k": 0, "top_p": 1,
            "max_new_tokens": MAX_NEW, "stop": [],
        }
        wire = (json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        dispatched = time.perf_counter()
        self.proc.stdin.write(wire)
        self.proc.stdin.flush()
        self.bytes_sent += len(wire)
        tokens = []
        first_token = None
        terminal = None
        while True:
            response = self.read_json()
            if "error" in response:
                raise RuntimeError(f"{self.name}: {response['error']}")
            if "token" in response:
                tokens.append(response["token"])
                if first_token is None:
                    first_token = time.perf_counter()
            if response.get("done"):
                terminal = response
                break
        finished = time.perf_counter()
        generated = int(terminal.get("generated", len(tokens)))
        if generated != MAX_NEW or len(tokens) != MAX_NEW:
            raise RuntimeError(f"{self.name}: request {request_id} generated {generated}/{MAX_NEW} tokens")
        decode_tok_s = float(terminal.get("decode_tok_s", 0.0))
        decode_s = generated / decode_tok_s if decode_tok_s > 0 else None
        return {
            "request_id": request_id,
            "worker": self.name,
            "arrival_s": arrival_rel_s,
            "dispatch_s": dispatched - origin,
            "first_token_s": first_token - origin if first_token else None,
            "completion_s": finished - origin,
            "queue_wait_s": max(0.0, dispatched - origin - arrival_rel_s),
            "ttft_from_arrival_s": first_token - origin - arrival_rel_s if first_token else None,
            "completion_from_arrival_s": finished - origin - arrival_rel_s,
            "service_roundtrip_s": finished - dispatched,
            "generated": generated,
            "runtime_decode_tok_s": decode_tok_s,
            "non_decode_remainder_s": max(0.0, finished - dispatched - decode_s) if decode_s else None,
            "token_sha256": token_digest(tokens),
        }

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=45)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.terminate()
                self.proc.wait(timeout=10)
        if self.proc.stdout and not self.proc.stdout.closed:
            self.proc.stdout.close()


def parse_worker(value):
    fields = value.split("|", 3)
    if len(fields) == 3 and fields[1] == "local":
        return (fields[0], "local", "", fields[2])
    if len(fields) != 4 or fields[1] != "ssh":
        raise argparse.ArgumentTypeError("worker must be NAME|local|COMMAND or NAME|ssh|HOST|COMMAND")
    return (fields[0], fields[1], fields[2], fields[3])


def parse_identity(value):
    fields = value.split("|")
    if len(fields) != 6 or any(not part for part in fields):
        raise argparse.ArgumentTypeError(
            "identity must be NAME|RUNTIME_SHA256|BUNDLE_SHA256|WEIGHTS_SHA256|TOKENIZER_SHA256|PLAN_SHA256")
    return fields[0], dict(zip(("runtime_sha256", "bundle_sha256", "weights_sha256",
                               "tokenizer_sha256", "plan_sha256"), fields[1:]))


def run_workload(workers, arrival_interval, round_index, condition_index):
    count = COUNT
    arrivals = [i * arrival_interval for i in range(count)]
    worker_names = [w.name for w in workers]
    rotation = (round_index + condition_index) % len(workers)
    rotated = workers[rotation:] + workers[:rotation]
    available = deque(rotated)
    pending = deque()
    next_arrival = 0
    results = []
    futures = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(workers))
    byte_start = {w.name: (w.bytes_sent, w.bytes_received) for w in workers}
    origin = time.perf_counter()
    try:
        while len(results) < count:
            now_rel = time.perf_counter() - origin
            while next_arrival < count and arrivals[next_arrival] <= now_rel:
                pending.append(next_arrival)
                next_arrival += 1
            while pending and available:
                request_id = pending.popleft()
                worker = available.popleft()
                fut = executor.submit(worker.generate, request_id, arrivals[request_id], origin)
                futures[fut] = worker
            if futures:
                timeout = None
                if next_arrival < count:
                    timeout = max(0.0, arrivals[next_arrival] - (time.perf_counter() - origin))
                done, _ = concurrent.futures.wait(futures, timeout=timeout,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    worker = futures.pop(fut)
                    results.append(fut.result())
                    available.append(worker)
            elif next_arrival < count:
                time.sleep(max(0.0, arrivals[next_arrival] - (time.perf_counter() - origin)))
        elapsed = time.perf_counter() - origin
    finally:
        executor.shutdown(wait=True)
    if len(results) != count:
        raise RuntimeError(f"expected {count} results, received {len(results)}")
    by_id = {row["request_id"]: row for row in results}
    hashes = [by_id[i]["token_sha256"] for i in range(count)]
    return {
        "requests": count,
        "requested_tokens": count * MAX_NEW,
        "generated_tokens": sum(row["generated"] for row in results),
        "elapsed_from_first_arrival_s": elapsed,
        "aggregate_tokens_per_second": sum(row["generated"] for row in results) / elapsed,
        "completed_requests_per_second": count / elapsed,
        "arrival_to_ttft_s": summarize([row["ttft_from_arrival_s"] for row in results]),
        "arrival_to_completion_s": summarize([row["completion_from_arrival_s"] for row in results]),
        "queue_wait_s": summarize([row["queue_wait_s"] for row in results]),
        "worker_service_roundtrip_s": summarize([row["service_roundtrip_s"] for row in results]),
        "worker_service_roundtrip_by_worker_s": {
            name: summarize([row["service_roundtrip_s"] for row in results if row["worker"] == name])
            for name in worker_names if any(row["worker"] == name for row in results)
        },
        "assignments": {name: sum(row["worker"] == name for row in results) for name in worker_names},
        "worker_runtime_decode_tok_s": {
            name: summarize([row["runtime_decode_tok_s"] for row in results if row["worker"] == name])
            for name in worker_names if any(row["worker"] == name for row in results)
        },
        "non_decode_remainder_s": summarize([row["non_decode_remainder_s"] for row in results
                                             if row["non_decode_remainder_s"] is not None]),
        "control_stream_bytes": {
            name: {"sent": w.bytes_sent - byte_start[name][0],
                   "received": w.bytes_received - byte_start[name][1]}
            for name, w in ((worker.name, worker) for worker in workers)
        },
        "all_request_outputs_identical_within_condition": len(set(hashes)) == 1,
        "per_request": [by_id[i] for i in range(count)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="append", required=True, type=parse_worker,
                        help="repeat: NAME|local|COMMAND or NAME|ssh|HOST|REMOTE_COMMAND")
    parser.add_argument("--identity", action="append", required=True, type=parse_identity,
                        help="repeat: NAME|runtime_sha256|bundle_sha256|weights_sha256|tokenizer_sha256|plan_sha256")
    parser.add_argument("--output", required=True, type=Path, help="new output JSON path; existing file is rejected")
    parser.add_argument("--rounds", type=int, default=ROUND_COUNT)
    args = parser.parse_args()
    names = {row[0] for row in args.worker}
    if names != {"B580", "RX570", "M4"} or len(args.worker) != 3:
        parser.error("provide exactly one worker named B580, RX570, and M4")
    identities = dict(args.identity)
    if set(identities) != names:
        parser.error("provide one --identity for each worker")
    for field in ("bundle_sha256", "weights_sha256", "tokenizer_sha256", "plan_sha256"):
        if len({identity[field] for identity in identities.values()}) != 1:
            parser.error(f"{field} differs across workers; this invalidates the comparison")
    if args.output.exists():
        parser.error(f"output already exists; choose a new path: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workers = {}
    try:
        for name, mode, target, command in args.worker:
            workers[name] = Worker(name, mode, target, command)
        warmups = {name: worker.generate(-1, 0.0, time.perf_counter())
                   for name, worker in workers.items()}
        warmup_hashes = {name: row["token_sha256"] for name, row in warmups.items()}
        if len(set(warmup_hashes.values())) != 1:
            raise RuntimeError(f"warm-up tokens differ across devices: {warmup_hashes}")
        all_hashes = {}
        matrix = []
        for round_index in range(args.rounds):
            shift = round_index % len(CONDITIONS)
            ordered = CONDITIONS[shift:] + CONDITIONS[:shift]
            for condition_index, condition in enumerate(ordered):
                pool = [workers[name] for name in condition]
                for arrival_name, interval in (("synchronized_burst", 0.0), ("steady_1_request_per_second", ARRIVAL_INTERVAL_S)):
                    result = run_workload(pool, interval, round_index, condition_index)
                    condition_key = "+".join(condition)
                    hashes_by_id = [row["token_sha256"] for row in result["per_request"]]
                    expected = all_hashes.setdefault(arrival_name, {}).setdefault(condition_key, hashes_by_id)
                    result["all_outputs_match_b580_reference"] = condition_key == "B580" or hashes_by_id == all_hashes[arrival_name].get("B580", hashes_by_id)
                    result["all_outputs_match_first_round"] = hashes_by_id == expected
                    result["condition"] = condition_key
                    result["round"] = round_index + 1
                    result["arrival_pattern"] = arrival_name
                    matrix.append(result)
        payload = {
            "protocol": "V27-T3",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "kind": "independent queued requests; not one-request distributed decoding",
            "workload": {"prompt": PROMPT, "requests": COUNT, "tokens_per_request": MAX_NEW,
                         "temperature": 0, "top_k": 0, "top_p": 1,
                         "steady_interarrival_s": ARRIVAL_INTERVAL_S, "rounds": args.rounds,
                         "percentile_method": "nearest rank; p99 is the maximum for n=24"},
            "workers": {name: {"mode": worker.mode, "target": worker.target,
                               "launch_command": worker.command, "ready": worker.ready,
                               "warmup_token_sha256": warmup_hashes[name], **identities[name]}
                        for name, worker in workers.items()},
            "matrix": matrix,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "conditions": len(matrix),
                          "warmup_hashes_equal": len(set(warmup_hashes.values())) == 1,
                          "request_hash_mismatches": sum(not row["all_request_outputs_identical_within_condition"] or
                                                          not row["all_outputs_match_first_round"] for row in matrix)},
                         indent=2))
    finally:
        for worker in workers.values():
            worker.close()


if __name__ == "__main__":
    main()
