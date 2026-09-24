"""Validate SeedPlane's independent Qwen2 graph against Transformers."""
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from seedplane.qwen_engine import Qwen2Engine


def main():
    p = argparse.ArgumentParser(); p.add_argument('bundle'); p.add_argument('--tokens', default='9707,11,1879,0'); p.add_argument('--device', default='cpu')
    a = p.parse_args(); ids = list(map(int, a.tokens.split(',')))
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(a.bundle, dtype=torch.float32).eval()
    ours = Qwen2Engine(a.bundle, a.device)
    with torch.inference_mode(): expected = ref(torch.tensor([ids])).logits[0, -1].float()
    actual, cache = ours.forward(ids, ours.new_cache(32)); actual = actual[-1]
    diff = (actual.cpu() - expected).abs(); ref_ids = []; our_ids = []
    ref_cache = None; ref_input = torch.tensor([ids])
    our_cache = ours.new_cache(32); our_logits, our_cache = ours.forward(ids, our_cache)
    for _ in range(8):
        with torch.inference_mode():
            out = ref(ref_input, past_key_values=ref_cache, use_cache=True); ref_cache = out.past_key_values
        r = int(out.logits[0, -1].argmax()); o = int(our_logits[-1].argmax()); ref_ids.append(r); our_ids.append(o)
        ref_input = torch.tensor([[r]]); our_logits, our_cache = ours.forward([o], our_cache)
    result = {'max_abs_logit_error': float(diff.max()), 'mean_abs_logit_error': float(diff.mean()),
              'reference_tokens': ref_ids, 'seedplane_tokens': our_ids, 'tokens_identical': ref_ids == our_ids,
              'cache_length': our_cache.length}
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result['tokens_identical'] else 1)


if __name__ == '__main__': main()
