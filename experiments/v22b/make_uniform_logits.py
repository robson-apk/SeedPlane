"""Create an all-equal binary32 vector sized from an existing logits fixture to exercise overflow fallback."""
import struct
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:3])
size = source.stat().st_size
if size == 0 or size % 4:
    raise SystemExit("source must be a non-empty float32 logits file")
output.write_bytes(struct.pack("<f", 0.0) * (size // 4))
print(size // 4)
