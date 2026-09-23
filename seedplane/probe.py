"""seedplane probe — find out what this machine can do and which API is fastest on each device.

For every seedplane-worker build given (one per backend flavour: Vulkan, SYCL, CUDA, Metal, CPU-only...), list the
devices it sees, then self-benchmark each device on this model: GPUs as one slot, CPUs with several slot x thread
layouts. Output: a profile JSON with, per physical device, the fastest (build, device, slots, threads) — what
`seedplane run --native` should start on this machine.

    python -m seedplane.probe --model qwen.gguf --worker build_vk/bin/seedplane-worker --worker build_sycl/bin/seedplane-worker
"""
import argparse, json, os, platform, subprocess, sys


def _run(cmd, env=None, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    txt = r.stdout; starts = [i for i in (txt.find('['), txt.find('{')) if i >= 0]
    return json.loads(txt[min(starts):]) if r.returncode == 0 and starts else None


def cpu_layouts(n_cores):
    """(slots, threads) pairs to try: all-parallel windows, all-threads-in-one, and a middle ground."""
    out = {(n_cores, 1), (1, n_cores)}
    for t in (2, 3):
        if n_cores % t == 0 and n_cores // t > 1: out.add((n_cores // t, t))
    return sorted(out)


def probe(model, workers, tokens=512, cores=None, env_prefix=None):
    cores = cores or os.cpu_count(); rows = []
    for exe in workers:
        pre = env_prefix.get(exe, []) if env_prefix else []
        devs = _run(pre + [exe, '--list-devices'])
        if devs is None: rows.append({'build': exe, 'error': 'cannot list devices'}); continue
        for d in devs:
            if d['type'] == 'accel': continue                       # BLAS etc. are helpers of the CPU, not devices
            layouts = cpu_layouts(cores) if d['type'] == 'cpu' else [(1, 1)]
            for slots, threads in layouts:
                res = _run(pre + [exe, '-m', model, '--dev', d['name'], '--slots', str(slots), '-t', str(threads), '--bench', str(tokens)])
                row = {'build': exe, 'dev': d['name'], 'backend': d['backend'], 'type': d['type'], 'description': d['description'],
                       'mem_mb': d['total_mb'], 'slots': slots, 'threads': threads}
                if res is None: row['error'] = 'bench failed'
                else: row['tok_s'] = res['tok_s']                  # all slots measured running at the same time
                rows.append(row); print(json.dumps(row), flush=True)
    # best entry per physical device (same description + type), across builds/APIs
    best = {}
    for r in rows:
        if 'tok_s' not in r: continue
        k = (r['type'], r['description'])
        if k not in best or r['tok_s'] > best[k]['tok_s']: best[k] = r
    return {'host': platform.node(), 'model': model, 'tokens': tokens, 'all': rows, 'best': list(best.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True); ap.add_argument('--worker', action='append', required=True)
    ap.add_argument('--tokens', type=int, default=512); ap.add_argument('--cores', type=int)
    ap.add_argument('--out', default='seedplane_profile.json')
    a = ap.parse_args()
    prof = probe(a.model, a.worker, a.tokens, a.cores)
    with open(a.out, 'w') as f: json.dump(prof, f, indent=1)
    print('best per device:'); [print(f"  {b['description']:<40} {b['backend']:<8} slots={b['slots']} threads={b['threads']}  ~{b['tok_s']:.0f} tok/s") for b in prof['best']]


if __name__ == '__main__':
    main()
