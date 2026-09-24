"""Thread-scaling benchmark for the SeedPlane PyTorch Qwen2 reference runtime.

This measures one autoregressive request at a time. It is a CPU/MPS hardware
baseline, not a speculative-decoding benchmark. Use the same model snapshot,
prompt, token count, Python/PyTorch version and protocol on each host.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from seedplane.qwen_engine import Qwen2Engine


PROMPT = "Explain briefly how a computer memory cache improves performance."


def digest(tokens):
    payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def run(args):
    bundle = Path(args.bundle).expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False).ids
    engine = Qwen2Engine(bundle, device=args.device)

    results = []
    order = list(args.threads)
    random.Random(args.order_seed).shuffle(order)
    for threads in order:
        torch.set_num_threads(threads)
        # Warm up once; do not include model load or warmup in reported timings.
        list(engine.generate(prompt_ids, max_new_tokens=min(16, args.tokens), seed=1))
        times, hashes = [], []
        for rep in range(args.repetitions):
            start = time.perf_counter()
            tokens = list(engine.generate(prompt_ids, max_new_tokens=args.tokens, seed=1))
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            hashes.append(digest(tokens))
        if len(set(hashes)) != 1:
            raise RuntimeError(f"non-deterministic output at {threads} threads: {hashes}")
        results.append({
            "threads": threads,
            "seconds": times,
            "median_seconds": statistics.median(times),
            "median_tokens_per_second": args.tokens / statistics.median(times),
            "token_sha256": hashes[0],
        })

    hashes = {row["token_sha256"] for row in results}
    return {
        "protocol": "seedplane-qwen2-reference-thread-scaling-v1",
        "host": __import__("platform").node(),
        "platform": __import__("platform").platform(),
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "device": args.device,
        "model": str(bundle),
        "model_config_sha256": hashlib.sha256((bundle / "config.json").read_bytes()).hexdigest(),
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": args.tokens,
        "repetitions": args.repetitions,
        "thread_order_seed": args.order_seed,
        "thread_order": order,
        "all_thread_counts_same_tokens": len(hashes) == 1,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="Qwen2 HF snapshot directory")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=lambda s: [int(x) for x in s.split(",")],
                        default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(t < 1 for t in args.threads) or args.tokens < 1 or args.repetitions < 1:
        parser.error("threads, tokens, and repetitions must be positive")
    result = run(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
