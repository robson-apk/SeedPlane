"""Run a persistent HIVE agent backed by a local SeedPlane native model."""
import argparse
import hashlib
import json
import os
import platform

from .pool import HiveAgent
from ..native import NativeEngine


def _native_generate(engine, payload):
    allowed = {"max_new_tokens", "temperature", "top_k", "top_p", "seed", "stop"}
    options = {key: value for key, value in payload.items() if key in allowed}
    text = payload.get("text", "")
    pieces = list(engine.generate(text=text, reset=True, **options))
    token_bytes = json.dumps(engine.last_tokens, separators=(",", ":")).encode("utf-8")
    return {"text": "".join(pieces), "tokens": engine.last_tokens,
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "stats": engine.last, "device": engine.info.get("device")}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent pull-based SeedPlane HIVE agent")
    parser.add_argument("--bundle", required=True, help="local SeedPlane native bundle")
    parser.add_argument("--engine", help="local qwen_vk executable")
    parser.add_argument("--shaders", help="shader directory for the local qwen_vk build")
    parser.add_argument("--model-id", required=True, help="model identity advertised to the broker")
    parser.add_argument("--worker-id", default=f"{platform.node()}-{platform.system().lower()}")
    parser.add_argument("--cap", action="append", default=[], help="additional execution capability (repeatable)")
    parser.add_argument("--connect", required=True, metavar="HOST:PORT", help="HIVE broker address")
    parser.add_argument("--authkey", default=os.environ.get("SEEDPLANE_AUTHKEY"),
                        help="shared secret; use only on a trusted network")
    args = parser.parse_args(argv)
    host, port = args.connect.rsplit(":", 1)
    engine_args = ("--shaders", args.shaders) if args.shaders else ()
    with NativeEngine(args.bundle, engine=args.engine, max_new_tokens=512,
                      extra_args=engine_args) as engine:
        agent = HiveAgent(host, int(port), args.worker_id,
                          caps=("generate", "throughput.generate", *args.cap),
                          models=(args.model_id,),
                          execute=lambda payload: _native_generate(engine, payload),
                          authkey=args.authkey)
        agent.run()


if __name__ == "__main__":
    main()
