"""V20 data: tokenize WikiText-2 test with the bundle tokenizer and cut the pre-registered, never-used ranges."""
import json, sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

text_file, tok_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ids = np.array(Tokenizer.from_file(str(tok_file)).encode(text_file.read_text(encoding='utf-8')).ids, dtype=np.int32)
out.mkdir(parents=True, exist_ok=True)
ranges = {'quality_1': (200000, 204096), 'quality_2': (204096, 208192), 'quality_3': (208192, 212288), 'speed_prompt': (220000, 223000)}
if len(ids) < max(b for _, b in ranges.values()): raise SystemExit(f'corpus too short: {len(ids)} tokens')
for name, (a, b) in ranges.items(): ids[a:b].tofile(out / f'{name}.i32')
json.dump({'corpus_tokens': int(len(ids)), 'ranges': ranges}, open(out / 'data.json', 'w'), indent=1)
print('corpus tokens', len(ids), {k: b - a for k, (a, b) in ranges.items()})
