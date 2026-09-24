"""V21 session continuation check against the isolated Linux/RADV runtime."""
import json
import sys
from pathlib import Path

root = Path.home() / "sp_v22b_codex"
sys.path.insert(0, str(root))
from seedplane.native import NativeEngine

bundle = Path.home() / "sp_v23" / "qwen05.sp"
engine = root / "build" / "qwen_vk"
context = "In this test, the system records an observation, checks it against prior context, and explains the result. " * 90
with NativeEngine(bundle, engine=engine, temperature=0) as eng:
    first = "".join(eng.chat("Summarize this text in two sentences:\n" + context, reset=True, max_new_tokens=64))
    first_status = dict(eng.last)
    second = "".join(eng.chat("Now give it a short title.", max_new_tokens=64))
    second_status = dict(eng.last)
    tokens = eng.state()["tokens"]
    generated = tokens[len(tokens) - second_status["generated"]:]
    prefix = tokens[:len(tokens) - second_status["generated"]]
    fresh = "".join(eng.generate(ids=prefix, reset=True, max_new_tokens=64))
    fresh_tokens = eng.state()["tokens"]
    fresh_generated = fresh_tokens[len(prefix):]
result = {"first_generated": first_status["generated"], "second_generated": second_status["generated"],
          "context_positions": len(prefix), "crossed_512": len(prefix) > 512, "tokens_equal": generated == fresh_generated,
          "turn2_text": second, "fresh_text": fresh}
result["pass"] = result["crossed_512"] and result["tokens_equal"]
out = root / "v22b" / "g5_session.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
