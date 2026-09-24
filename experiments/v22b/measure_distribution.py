"""Compare a GPU sample-test count file with the float64 reference distribution."""
import json
import sys
from pathlib import Path

import numpy as np

logits_path, counts_path = map(Path, sys.argv[1:3])
temperature, top_k, top_p = float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
logits = np.fromfile(logits_path, np.float32)
counts = np.fromfile(counts_path, np.uint32)
z = logits.astype(np.float64) / temperature
if top_k > 0:
    z = np.where(z < np.sort(z)[::-1][top_k - 1], -np.inf, z)
if top_p < 1.0:
    order = np.argsort(-z, kind="stable")
    probs = np.exp(z[order] - z[order][0]); probs /= probs.sum()
    z[order[np.cumsum(probs) - probs > top_p]] = -np.inf
probs = np.exp(z - np.max(z)); probs /= probs.sum()
empirical = counts.astype(np.float64) / counts.sum()
print(json.dumps({"support": int((probs > 0).sum()), "outside_draws": int(counts[probs == 0].sum()),
                  "tv": float(0.5 * np.abs(empirical - probs).sum()), "draws": int(counts.sum())}, indent=2))
