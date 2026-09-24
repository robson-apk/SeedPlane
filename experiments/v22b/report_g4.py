import json
import sys
from pathlib import Path

data = json.loads((Path(sys.argv[1]) / "v22b" / "g4_session_cli.json").read_text(encoding="utf-8"))
print(json.dumps({"G5": data["G5_session"]["pass"], "G5_positions": data["G5_session"]["context_positions"],
                  "G5_tokens_equal": data["G5_session"]["equal_tokens"], "G6": data["G6_cli"]["pass"],
                  "convert_ok": data["G6_cli"]["convert_ok"], "chat_returncode": data["G6_cli"]["chat_returncode"],
                  "answers": data["G6_cli"]["answers"], "stderr_tail": data["G6_cli"]["stderr_tail"]}, ensure_ascii=True))
