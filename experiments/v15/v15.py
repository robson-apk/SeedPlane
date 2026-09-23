"""V15: compare compute-only and end-to-end SeedPlane span planning. See PROTOCOL.md."""
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from seedplane import llama_backend as lb


def timed(fn, reps):
    fn()
    values = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); values.append(time.perf_counter() - t0)
    return values


def compute_rates(workers, ids, win, repeats=2):
    work = {w.name: [win] * repeats for w in workers}
    lb.run_pieces(workers, ids, {w.name: [win] for w in workers}, 'span')
    out, _ = lb.run_pieces(workers, ids, work, 'span')
    n = repeats * len(win[2])
    return {name: n / (sum(row[3] for row in rows) / 1e3) for name, rows in out.items()}


def allocation(pieces):
    return {name: sum(c1 - c0 for c0, c1 in spans) for name, spans in pieces.items()}


def run_plan(workers, ids, pieces):
    return lb.run_pieces(workers, ids, lb.windows_from_pieces(pieces, H=256), 'span')[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', default='127.0.0.1:54990'); ap.add_argument('--mac', required=True)
    ap.add_argument('--text', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--lengths', default='4096,8192,16384'); ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'v15.json'))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.bundle)
    ids_all = np.asarray(tok(Path(a.text).read_text(encoding='utf-8', errors='ignore')).input_ids, dtype=np.int64)
    gpu = lb.connect([a.gpu]); mac = lb.connect([a.mac]); workers = gpu + mac
    result = {'reps': a.reps, 'rows': {}}
    try:
        for L in map(int, a.lengths.split(',')):
            ids = ids_all[:L]; probe_n = min(1024, L); probe = (0, probe_n, np.arange(probe_n))
            wall_rates = lb.measure_rates(workers, ids, probe, 'span', repeats=2)
            raw_compute_rates = compute_rates(workers, ids, probe)
            old_pieces, old_T = lb.plan_pieces(L, raw_compute_rates, H=256, span=True, min_gain=0)
            new_pieces, new_T = lb.plan_pieces(L, wall_rates, H=256, span=True)
            solo = {gpu[0].name: [(0, L)], mac[0].name: []}
            timings = {
                'gpu_only': timed(lambda: run_plan(workers, ids, solo), a.reps),
                'compute_only': timed(lambda: run_plan(workers, ids, old_pieces), a.reps),
                'end_to_end': timed(lambda: run_plan(workers, ids, new_pieces), a.reps),
            }
            result['rows'][str(L)] = {
                'compute_rates': raw_compute_rates, 'end_to_end_rates': wall_rates,
                'compute_allocation': allocation(old_pieces), 'end_to_end_allocation': allocation(new_pieces),
                'compute_predicted_seconds': old_T, 'end_to_end_predicted_seconds': new_T,
                'seconds': timings,
                'median_tok_s': {name: L / statistics.median(ts) for name, ts in timings.items()},
            }
            out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=1))
            print(L, result['rows'][str(L)]['median_tok_s'], result['rows'][str(L)]['end_to_end_allocation'], flush=True)
    finally:
        for w in workers: w.close()


if __name__ == '__main__':
    main()
