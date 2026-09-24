#!/usr/bin/env python3
"""Exploratory greedy n-gram speculative decode on SeedPlane's PyTorch Qwen engine.

This is a single-device experiment, not the native Vulkan/tree/fleet Hive path.
Drafts are copied from matching suffixes in the already committed token history;
the target checks each draft wave with one multi-token forward call.
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

from seedplane.qwen_engine import Qwen2Engine, chat_prompt


def sync(device):
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def ngram_draft(history, width, max_order=8):
    """Copy continuation following the closest previous match of a history suffix."""
    end = len(history)
    for order in range(min(max_order, end - 1), 0, -1):
        suffix = history[end - order:]
        for start in range(end - order - 1, -1, -1):
            if history[start:start + order] == suffix:
                continuation = history[start + order:start + order + width]
                if continuation:
                    return continuation
    return []


@torch.inference_mode()
def normal_decode(engine, prompt_ids, count, device):
    cache = engine.new_cache(len(prompt_ids) + count + 8)
    logits, cache = engine.forward(prompt_ids, cache)
    out = []
    sync(device)
    started = time.perf_counter()
    for _ in range(count):
        token = int(logits[-1].argmax())
        out.append(token)
        if len(out) < count:
            logits, cache = engine.forward([token], cache)
    sync(device)
    elapsed = time.perf_counter() - started
    return out, elapsed


@torch.inference_mode()
def speculative_decode(engine, prompt_ids, count, width, device):
    cache = engine.new_cache(len(prompt_ids) + count + width + 8)
    logits, cache = engine.forward(prompt_ids, cache)
    committed = list(prompt_ids)
    out = []
    stats = {"drafted": 0, "accepted": 0, "target_batch_calls": 0,
             "target_single_token_calls": 0, "fallback_tokens": 0,
             "rejected_corrections": 0}
    sync(device)
    started = time.perf_counter()
    while len(out) < count:
        remaining = count - len(out)
        draft = ngram_draft(committed, min(width, remaining))
        if not draft:
            token = int(logits[-1].argmax())
            out.append(token)
            committed.append(token)
            if len(out) < count:
                logits, cache = engine.forward([token], cache)
                stats["target_single_token_calls"] += 1
            stats["fallback_tokens"] += 1
            continue

        # The current logits predict draft[0]. The batched pass computes the
        # target's continuation after each proposed token (including rejection).
        target_first = int(logits[-1].argmax())
        stats["drafted"] += len(draft)
        if draft[0] != target_first:
            out.append(target_first)
            committed.append(target_first)
            if len(out) < count:
                logits, cache = engine.forward([target_first], cache)
                stats["target_single_token_calls"] += 1
            stats["fallback_tokens"] += 1
            continue

        old_length = cache.length
        batch_logits, cache = engine.forward(draft, cache, all_logits=True)
        stats["target_batch_calls"] += 1
        accepted = 0
        rejected_token = None
        # Draft[0] is checked against pre-wave logits; remaining drafts are
        # checked against the corresponding rows produced by the block pass.
        accepted = 1
        for i in range(1, len(draft)):
            target_token = int(batch_logits[i - 1].argmax())
            if draft[i] != target_token:
                rejected_token = target_token
                break
            accepted += 1

        stats["accepted"] += accepted
        emitted = draft[:accepted]
        out.extend(emitted[:remaining])
        committed.extend(emitted[:remaining])
        if rejected_token is not None and len(out) < count:
            out.append(rejected_token)
            committed.append(rejected_token)
            stats["rejected_corrections"] += 1
            cache.length = old_length + accepted
            if len(out) < count:
                logits, cache = engine.forward([rejected_token], cache)
                stats["target_single_token_calls"] += 1
        else:
            # Cache includes all draft tokens only when the entire proposal was
            # accepted. If the output cap truncated it, trim unused KV slots.
            used = min(accepted, remaining)
            cache.length = old_length + used
            if used < accepted:
                logits, cache = engine.forward([out[-1]], cache)
            else:
                logits = batch_logits[-1:]

    sync(device)
    elapsed = time.perf_counter() - started
    return out, elapsed, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/Users/robson/qwen05_v1.sp")
    ap.add_argument("--tokenizer", default="/Users/robson/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--widths", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt = chat_prompt([{"role": "user", "content": "Explain briefly how a computer memory cache improves performance."}])
    prompt_ids = tokenizer.encode(prompt).ids
    engine = Qwen2Engine(args.bundle, device=device)
    results = {"device": str(device), "tokens": args.tokens, "repeats": args.repeats,
               "prompt_tokens": len(prompt_ids), "draft": "suffix n-gram copying",
               "measurement_order": "one full greedy warm-up, one 16-token warm-up per speculative width, then randomized paired order per repeat",
               "runs": []}
    shuffled_widths = list(args.widths)
    random.Random(1729).shuffle(shuffled_widths)
    configurations = [0] + shuffled_widths
    reference_output, _ = normal_decode(engine, prompt_ids, args.tokens, str(device))
    reference = hashlib.sha256(json.dumps(reference_output).encode()).hexdigest()
    warmup_tokens = min(16, args.tokens)
    for width in shuffled_widths:
        speculative_decode(engine, prompt_ids, warmup_tokens, width, str(device))
    measurements_by_width = {width: [] for width in configurations}
    for rep in range(args.repeats):
        order = configurations[:]
        random.Random(1729 + rep).shuffle(order)
        for width in order:
            if width == 0:
                output, seconds = normal_decode(engine, prompt_ids, args.tokens, str(device))
                stats = None
            else:
                output, seconds, stats = speculative_decode(engine, prompt_ids, args.tokens, width, str(device))
            digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
            measurements_by_width[width].append({"rep": rep, "seconds": seconds,
                                                  "tok_s": args.tokens / seconds,
                                                  "sha256": digest,
                                                  "matches_greedy": digest == reference,
                                                  "stats": stats})
    for width in configurations:
        measurements = measurements_by_width[width]
        results["runs"].append({"width": width, "mode": "greedy" if width == 0 else "ngram_speculative",
                                "median_tok_s": statistics.median(x["tok_s"] for x in measurements),
                                "median_seconds": statistics.median(x["seconds"] for x in measurements),
                                "all_match_greedy": all(x["matches_greedy"] for x in measurements),
                                "measurements": measurements})
    rendered = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
