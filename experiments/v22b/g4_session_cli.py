"""Run V21 session continuation and CLI convert/chat checks against a selected native binary."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
engine = str(root / "native" / "build" / "qwen_vk.exe")
bundle = root.parent / "sp_v19" / "run" / "qwen05_v1.sp"
source = root.parent / "seedplane_v12" / "qwen05.sp"
sys.path.insert(0, str(root))
from seedplane.native import NativeEngine

# Deterministic text input long enough to cross the 512-token session window boundary.
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
g5 = {"first_generated": first_status["generated"], "second_generated": second_status["generated"],
      "context_positions": len(prefix), "crossed_512": len(prefix) > 512, "equal_tokens": generated == fresh_generated,
      "turn2_text": second, "fresh_text": fresh}
g5["pass"] = g5["crossed_512"] and g5["equal_tokens"]

cli_bundle = root / "v22b" / "cli.sp"
start = time.time()
convert = subprocess.run([sys.executable, "-m", "seedplane.cli", "convert", str(source), str(cli_bundle), "--native"],
                         cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
chat = subprocess.run([sys.executable, "-m", "seedplane.cli", "chat", str(cli_bundle), "--native", "--engine", engine,
                       "-n", "48", "--seed", "5"], cwd=root,
                      input="Olá! Em uma frase, o que é uma GPU?\nE uma CPU?\n".encode("utf-8"), capture_output=True)
chat_stdout = chat.stdout.decode("cp1252", errors="replace")
chat_stderr = chat.stderr.decode("cp1252", errors="replace")
answers = [line.split("Qwen> ", 1)[1] for line in chat_stdout.splitlines() if "Qwen> " in line]
g6 = {"convert_ok": convert.returncode == 0, "convert_output": convert.stdout[-300:], "chat_returncode": chat.returncode,
      "answers": answers, "stderr_tail": (convert.stderr + chat_stderr)[-600:], "seconds": time.time() - start}
g6["pass"] = g6["convert_ok"] and chat.returncode == 0 and len(answers) == 2 and all(x.strip() for x in answers)
result = {"G5_session": g5, "G6_cli": g6}
out = root / "v22b" / "g4_session_cli.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
