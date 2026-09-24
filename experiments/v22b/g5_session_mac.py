"""V21 session continuation check against the local Apple M4/MoltenVK runtime."""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
from seedplane.native import NativeEngine

engine = repo / "native" / "vulkan_decode" / "build_codex_v22b" / "qwen_vk"
shared = Path("/Volumes/10.0.0.146/Users/Windows 11/sp_v19")
bundle = shared / "run" / "qwen05_v1.sp"
context = "In this test, the system records an observation, checks it against prior context, and explains the result. " * 90
with NativeEngine(bundle, engine=engine, temperature=0) as eng:
    first_text = "".join(eng.chat("Summarize this text in two sentences:\n" + context, reset=True, max_new_tokens=64))
    first = dict(eng.last)
    second_text = "".join(eng.chat("Now give it a short title.", max_new_tokens=64))
    second = dict(eng.last)
    tokens = eng.state()["tokens"]
    generated = tokens[len(tokens) - second["generated"]:]
    prefix = tokens[:len(tokens) - second["generated"]]
    fresh_text = "".join(eng.generate(ids=prefix, reset=True, max_new_tokens=64))
    fresh_tokens = eng.state()["tokens"]
    fresh = fresh_tokens[len(prefix):]
result = {"first_generated": first["generated"], "second_generated": second["generated"],
          "context_positions": len(prefix), "crossed_512": len(prefix) > 512,
          "tokens_equal": generated == fresh, "turn2_text": second_text, "fresh_text": fresh_text}
result["pass"] = result["crossed_512"] and result["tokens_equal"]
out = repo / "experiments" / "v22b" / "g5_session_mac.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
