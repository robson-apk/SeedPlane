"""V16: saturated independent-request throughput on B580 + Mac. See PROTOCOL.md."""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from seedplane import llama_backend as lb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', default='127.0.0.1:54990'); ap.add_argument('--mac', required=True)
    ap.add_argument('--text', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--length', type=int, default=4096); ap.add_argument('--jobs', type=int, default=18)
    ap.add_argument('--out', default=str(ROOT / 'results.json'))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.bundle)
    ids_all = np.asarray(tok(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)
    ids = ids_all[:a.length]; win = (0, a.length, np.arange(a.length)); jobs = [(ids, win)] * a.jobs
    gpu = lb.connect([a.gpu]); mac = lb.connect([a.mac]); workers = gpu + mac
    try:
        # Admission decisions need the actual job size. The 1k probe used for
        # within-request sharding is optimistic when extrapolated to a full 4k
        # request, especially on the remote CPU worker.
        rates = lb.measure_rates(workers, ids, win, 'span', repeats=1)
        estimates = {name: a.length / rate for name, rate in rates.items()}
        lb.run_batch(gpu, jobs[:2], 'span', {gpu[0].name: estimates[gpu[0].name]})
        _, gpu_seconds, gpu_count = lb.run_batch(gpu, jobs, 'span', {gpu[0].name: estimates[gpu[0].name]})
        timeline = []
        output, pool_seconds, pool_count = lb.run_batch(workers, jobs, 'span', estimates, timeline)
        result = {
            'length': a.length, 'jobs': a.jobs, 'end_to_end_rates': rates,
            'gpu_only': {'seconds': gpu_seconds, 'jobs_per_worker': gpu_count,
                         'requests_s': a.jobs / gpu_seconds, 'tokens_s': a.jobs * a.length / gpu_seconds},
            'pool': {'seconds': pool_seconds, 'jobs_per_worker': pool_count,
                     'requests_s': a.jobs / pool_seconds, 'tokens_s': a.jobs * a.length / pool_seconds},
            'speedup': gpu_seconds / pool_seconds, 'complete_results': sum(x is not None for x in output),
            'timeline': timeline,
        }
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ('gpu_only', 'pool', 'speedup', 'complete_results')}, indent=1))
    finally:
        for w in workers: w.close()


if __name__ == '__main__':
    main()
