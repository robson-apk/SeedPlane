"""Feasibility test: independent greedy requests over persistent SSH stdio workers.

This measures aggregate batch throughput, not cooperative acceleration of one decode.
Run from the Mac with SSH access to the Windows B580 and Linux X79 hosts.
"""
import concurrent.futures
import json
import math
import os
import select
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fleet_g2.json"
COUNT = 24
MAX_NEW = 128
PROMPT = "Explain in one concise paragraph how a computer memory cache improves performance."
WEIGHTS_SHA256 = "ef9f3f59f925e46a303193d7b88a979a899c53b5fec2ed0ff679b61fdf3cbe49"


class Worker:
    def __init__(self, name, command, ssh_args):
        self.name = name
        self.proc = subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", *ssh_args, command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.buffer = bytearray()
        ready = self.read_json(timeout=180)
        if not ready.get("ready"):
            raise RuntimeError(f"{name}: server did not report ready: {ready}")
        self.ready = ready

    def read_json(self, timeout=120):
        deadline = time.monotonic() + timeout
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self.buffer[:newline])
                del self.buffer[:newline + 1]
                return json.loads(raw.decode("utf-8"))
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"{self.name}: timed out waiting for JSON response")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"{self.name}: SSH worker closed stdout (exit={self.proc.poll()})")
            self.buffer.extend(chunk)

    def generate(self):
        request = {
            "op": "generate", "text": PROMPT, "reset": True,
            "temperature": 0, "top_k": 0, "top_p": 1,
            "max_new_tokens": MAX_NEW, "stop": [],
        }
        self.proc.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        tokens = []
        started = time.perf_counter()
        while True:
            response = self.read_json()
            if "error" in response:
                raise RuntimeError(f"{self.name}: {response['error']}")
            if "token" in response:
                tokens.append(response["token"])
            if response.get("done"):
                return {
                    "worker": self.name, "generated": response["generated"],
                    "decode_tok_s": response["decode_tok_s"],
                    "seconds_roundtrip": time.perf_counter() - started,
                    "finished_at": time.perf_counter(),
                    "tokens": tokens,
                }

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)


def run_load(workers, count):
    rows = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {}
        submitted = 0
        for worker in workers[:count]:
            futures[pool.submit(worker.generate)] = worker
            submitted += 1
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                worker = futures.pop(future)
                row = future.result()
                row["completion_from_batch_start_s"] = row["finished_at"] - start
                rows.append(row)
                if submitted < count:
                    futures[pool.submit(worker.generate)] = worker
                    submitted += 1
    elapsed = time.perf_counter() - start
    expected = rows[0]["tokens"]
    matching = all(row["generated"] == MAX_NEW and row["tokens"] == expected for row in rows)
    service = sorted(row["seconds_roundtrip"] for row in rows)
    completion = sorted(row["completion_from_batch_start_s"] for row in rows)
    service_p95 = service[max(0, math.ceil(0.95 * len(service)) - 1)]
    service_p99 = service[max(0, math.ceil(0.99 * len(service)) - 1)]
    completion_p95 = completion[max(0, math.ceil(0.95 * len(completion)) - 1)]
    completion_p99 = completion[max(0, math.ceil(0.99 * len(completion)) - 1)]
    return {
        "requests": len(rows), "requested_tokens": count * MAX_NEW,
        "generated_tokens": sum(row["generated"] for row in rows),
        "wall_s": elapsed, "aggregate_tok_s": sum(row["generated"] for row in rows) / elapsed,
        "worker_service_p95_s": service_p95, "worker_service_p99_s": service_p99,
        "burst_completion_p95_s": completion_p95, "burst_completion_p99_s": completion_p99,
        "arrival_assumption": "all batch requests arrive together at batch start",
        "per_request_roundtrip_s": [row["seconds_roundtrip"] for row in rows],
        "per_request_device_tok_s": [row["decode_tok_s"] for row in rows],
        "workers_used": {name: sum(row["worker"] == name for row in rows) for name in sorted({r["worker"] for r in rows})},
        "all_greedy_tokens_identical": matching,
        "pass": len(rows) == count and matching,
    }


def main():
    windows = Worker(
        "B580",
        '"C:\\Users\\Windows 11\\sp_v22b_codex\\native\\build\\qwen_vk.exe" '
        '"C:\\Users\\Windows 11\\sp_v19\\run\\qwen05_v1.sp" --serve',
        ["-o", 'User="Windows 11"', "10.0.0.146"],
    )
    linux = Worker(
        "RX570",
        "~/sp_v22b_codex/build/qwen_vk ~/sp_v23/qwen05.sp --serve",
        ["robson@10.0.0.253"],
    )
    try:
        # Warm up both loaded replicas before timed rounds.
        warmup = [windows.generate(), linux.generate()]
        if any(x["tokens"] != warmup[0]["tokens"] for x in warmup):
            raise RuntimeError("B580 and RX570 greedy outputs differ on warm-up")
        rounds = []
        for _ in range(3):
            single = run_load([windows], COUNT)
            fleet = run_load([windows, linux], COUNT)
            rounds.append({"single_b580": single, "fleet_b580_rx570": fleet,
                           "speedup": fleet["aggregate_tok_s"] / single["aggregate_tok_s"]})
        result = {
            "kind": "independent-request aggregate throughput; not one-request decode speedup",
            "weights_sha256": WEIGHTS_SHA256, "prompt": PROMPT, "max_new_tokens": MAX_NEW,
            "requests_per_round": COUNT,
            "devices": {"B580": windows.ready, "RX570": linux.ready},
            "warmup_tokens_equal": True, "rounds": rounds,
            "median_speedup": sorted(r["speedup"] for r in rounds)[1],
            "aggregate_gate_min_speedup": 1.30,
            "aggregate_gate_pass": sorted(r["speedup"] for r in rounds)[1] >= 1.30,
            "latency_gate_max_p99_increase": 1.20,
            "latency_arrival_methodology_preregistered": False,
            "median_worker_service_p99_ratio": sorted(
                r["fleet_b580_rx570"]["worker_service_p99_s"] /
                r["single_b580"]["worker_service_p99_s"] for r in rounds
            )[1],
            "median_burst_completion_p99_ratio": sorted(
                r["fleet_b580_rx570"]["burst_completion_p99_s"] /
                r["single_b580"]["burst_completion_p99_s"] for r in rounds
            )[1],
            "measurement_integrity_pass": all(
                r["single_b580"]["pass"] and r["fleet_b580_rx570"]["pass"] for r in rounds
            ),
        }
        OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=True))
    finally:
        windows.close()
        linux.close()


if __name__ == "__main__":
    main()
