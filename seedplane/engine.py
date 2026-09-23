"""SeedPlane engine for existing Hugging Face causal LMs (no retraining).

A long prompt is cut into shards of S tokens. Shard k is processed as an independent window
  [optional sink tokens] + [H halo tokens from the previous shard] + [the S tokens of shard k]
with the ORIGINAL position ids, so each token sees its own shard (causally), the halo and the sinks.
Windows do not talk to each other, so they can run on any mix of devices in one pass.
"""
from dataclasses import dataclass, asdict
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ShardPlan:
    shard: int = 512      # S: tokens owned by each shard
    halo: int = 256       # H: tokens borrowed from the previous shard
    sinks: int = 4        # first tokens of the sequence visible to every shard (0 = off)

    def windows(self, L):
        """Yield (core_start, core_end, index_array) with the token indices each window reads, in order."""
        for c0 in range(0, L, self.shard):
            c1 = min(L, c0 + self.shard); h0 = max(0, c0 - self.halo)
            sink = list(range(min(self.sinks, h0))) if self.sinks else []
            yield c0, c1, np.array(sink + list(range(h0, c1)), dtype=np.int64)


def load_model(path_or_id, device='cpu', dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_id)
    model = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=dtype).to(device).eval()
    return model, tok


@torch.inference_mode()
def window_logits(model, ids, index, device):
    """Logits for every token of one window (causal inside the window, original positions)."""
    x = torch.as_tensor(ids[index], device=device)[None]; pos = torch.as_tensor(index, device=device)[None]
    return model(input_ids=x, position_ids=pos).logits[0].float()


@torch.inference_mode()
def full_logits(model, ids, device):
    x = torch.as_tensor(ids, device=device)[None]
    return model(input_ids=x).logits[0].float()


def core_logits(model, ids, plan, device):
    """Stitch the logits of each shard's core tokens (single device, sequential windows)."""
    L = len(ids); out = torch.empty(L, model.config.vocab_size)
    for c0, c1, idx in plan.windows(L):
        z = window_logits(model, ids, idx, device); out[c0:c1] = z[-(c1 - c0):].cpu()
    return out


def nll_from_logits(logits, ids):
    y = torch.as_tensor(ids[1:]); return float(F.cross_entropy(logits[:-1], y, reduction='sum')), len(y)


def save_bundle(out_dir, model_id, plan):
    """`seedplane convert`: a bundle = the original safetensors + tokenizer + seedplane.json (shard plan, provenance)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    (out / 'seedplane.json').write_text(json.dumps({'source': model_id, 'plan': asdict(plan), 'format': 'seedplane-bundle/1',
                                                    'note': 'weights unchanged; SeedPlane only changes how attention is scheduled'}, indent=1))
    return out
