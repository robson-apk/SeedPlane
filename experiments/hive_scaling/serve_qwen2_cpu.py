"""JSON-lines CPU worker matching qwen_vk's generate stream for pool tests."""
import argparse
import json
import sys
import time

import torch
from tokenizers import Tokenizer

from seedplane.qwen_engine import Qwen2Engine


def send(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = Tokenizer.from_file(args.bundle.rstrip("/\\") + "/tokenizer.json")
    engine = Qwen2Engine(args.bundle, device="cpu")
    send({"ready": True, "device": "CPU reference", "backend": "PyTorch",
          "torch": torch.__version__, "threads": args.threads,
          "precision": "float32", "model": engine.config.get("_name_or_path", "Qwen2")})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req.get("op") == "close":
                send({"ok": True})
                break
            if req.get("op") != "generate":
                send({"error": "only op=generate is supported"})
                continue
            if float(req.get("temperature", 0)) != 0:
                send({"error": "CPU reference worker only supports greedy generation"})
                continue
            prompt = req.get("text", "")
            ids = req.get("ids") or tok.encode(prompt, add_special_tokens=False).ids
            count = int(req.get("max_new_tokens", 128))
            start = time.perf_counter()
            generated = 0
            for token in engine.generate(ids, max_new_tokens=count, temperature=0):
                piece = tok.decode([token], skip_special_tokens=False)
                send({"token": token, "text": piece})
                generated += 1
            elapsed = time.perf_counter() - start
            send({"done": True, "reason": "length", "generated": generated,
                  "prefill_s": None, "decode_tok_s": generated / elapsed if elapsed else 0.0})
        except Exception as exc:
            send({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
