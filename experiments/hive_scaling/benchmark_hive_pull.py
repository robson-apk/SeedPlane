#!/usr/bin/env python3
"""Smoke/overhead comparison for one persistent local native HIVE pull agent.

This does not measure multi-device scaling or claim an optimized broker. It
checks real qwen_vk generation through the binary framed broker path against
the same resident runtime called directly, with identical independent jobs.
"""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

from seedplane.hive import HiveAgent, HiveBroker
from seedplane.hive.agent import _native_generate
from seedplane.native import NativeEngine


PROMPT = "Explain briefly how a computer memory cache improves performance."


def request(max_new_tokens):
    return {"text": PROMPT, "max_new_tokens": max_new_tokens,
            "temperature": 0, "top_k": 0, "top_p": 1, "seed": 1, "stop": []}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def measure_direct(bundle, executable, count, tokens):
    with NativeEngine(bundle, engine=executable, max_new_tokens=tokens) as engine:
        _native_generate(engine, request(8))  # warm the native device/runtime
        rows = []
        started = time.perf_counter()
        for _ in range(count):
            result = _native_generate(engine, request(tokens))
            rows.append(result)
        elapsed = time.perf_counter() - started
        return {"elapsed_s": elapsed, "generated_tokens": sum(x["stats"]["generated"] for x in rows),
                "tok_s": sum(x["stats"]["generated"] for x in rows) / elapsed,
                "text_sha256": [digest(x["text"]) for x in rows],
                "runtime_decode_tok_s": [x["stats"]["decode_tok_s"] for x in rows],
                "device": engine.info.get("device")}


def measure_pull(bundle, executable, count, tokens):
    with HiveBroker(pull_poll_s=0.005) as broker:
        with NativeEngine(bundle, engine=executable, max_new_tokens=tokens) as engine:
            agent = HiveAgent(*broker.address, "m4-pull-smoke", caps=("generate", "throughput.generate"),
                              models=("qwen05-sp",),
                              execute=lambda payload: _native_generate(engine, payload))
            thread = threading.Thread(target=agent.run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 30
            while "m4-pull-smoke" not in broker.workers() and time.monotonic() < deadline:
                time.sleep(0.01)
            if "m4-pull-smoke" not in broker.workers():
                raise TimeoutError("native HIVE agent did not connect")
            sessions = len(broker.workers())
            # Untimed warm-up ensures the device is initialized before the batch.
            broker.submit(request(8), model="qwen05-sp", required_caps=("generate",)).result(timeout=180)
            started = time.perf_counter()
            futures = [broker.submit(request(tokens), model="qwen05-sp",
                                     required_caps=("throughput.generate",)) for _ in range(count)]
            rows = [future.result(timeout=600) for future in futures]
            elapsed = time.perf_counter() - started
            agent.stop()
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("native HIVE agent failed to stop")
            return {"elapsed_s": elapsed, "generated_tokens": sum(x["stats"]["generated"] for x in rows),
                    "tok_s": sum(x["stats"]["generated"] for x in rows) / elapsed,
                    "text_sha256": [digest(x["text"]) for x in rows],
                    "runtime_decode_tok_s": [x["stats"]["decode_tok_s"] for x in rows],
                    "device": engine.info.get("device"), "completed_jobs": agent.completed,
                    "warmup_jobs": 1, "timed_jobs": agent.completed - 1,
                    "persistent_agent_sessions_at_start": sessions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    direct = measure_direct(args.bundle, args.engine, args.requests, args.tokens)
    pull = measure_pull(args.bundle, args.engine, args.requests, args.tokens)
    expected = direct["text_sha256"]
    same_outputs = len(set(expected)) == 1 and pull["text_sha256"] == expected
    result = {"runtime": str(args.engine), "bundle": str(args.bundle),
              "workload": {"independent_requests": args.requests, "tokens_per_request": args.tokens,
                           "prompt": PROMPT, "greedy": True},
              "direct": direct, "pull_agent": pull, "all_text_hashes_match": same_outputs,
              "pull_vs_direct_tok_s_ratio": pull["tok_s"] / direct["tok_s"],
              "scope": "single M4 host; throughput-lane transport smoke; not a fleet scaling result"}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    if not same_outputs or pull["timed_jobs"] != args.requests:
        raise SystemExit("HIVE pull smoke failed output integrity or job-count checks")


if __name__ == "__main__":
    main()
