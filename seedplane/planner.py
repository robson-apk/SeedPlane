"""Plan how to spread ONE model over heterogeneous devices (V14): which devices to use and how many layers each gets.

Pipeline model (exact, no quality loss): device i holds n_i consecutive layers; a prompt flows through the stages in
micro-batches, so steady-state throughput is set by the slowest stage (compute of its layers + sending activations to
the next stage). The planner measures nothing itself; feed it per-device speeds (e.g. llama-bench tok/s of the whole
model on that device alone) and it returns the best subset + integer layer split, including "use one device only".

    from seedplane.planner import Device, plan_pipeline
    devs = [Device('B580', 20000, mem_gb=12), Device('CPU', 300, mem_gb=8), Device('Mac', 150, mem_gb=4, link_gbps=1)]
    p = plan_pipeline(devs, n_layers=24, layer_gb=0.04, hidden=896)
    p.tensor_split   # '24,0,0' -> pass to llama.cpp as -ts (order = devs)
"""
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class Device:
    name: str
    tok_s: float                 # whole-model prompt throughput on this device alone (tokens/s)
    mem_gb: float = 1e9          # memory available for weights + cache
    link_gbps: float = 10.0      # bandwidth of the link that feeds this device's output to the next stage
    link_ms: float = 0.2         # per-micro-batch latency of that link


@dataclass
class Plan:
    devices: list
    layers: list                 # layers per device, same order as the input device list (0 = unused)
    tok_s: float                 # predicted pipeline throughput
    bottleneck: str
    tensor_split: str = field(init=False)

    def __post_init__(self):
        self.tensor_split = ','.join(str(n) for n in self.layers)


def _stage_time(dev, n, n_layers, ubatch, act_bytes, last):
    """Seconds per micro-batch for a stage holding n layers on dev."""
    compute = n / n_layers * ubatch / dev.tok_s
    comm = 0.0 if last else ubatch * act_bytes * 8 / (dev.link_gbps * 1e9) + dev.link_ms / 1e3
    return compute + comm


def _split(devs, n_layers, layer_gb):
    """Integer split that minimizes the slowest stage's compute, respecting memory (greedy by marginal cost)."""
    cap = [int(d.mem_gb // layer_gb) if layer_gb > 0 else n_layers for d in devs]
    n = [0] * len(devs)
    for _ in range(n_layers):   # give the next layer to the device whose stage stays fastest after taking it
        ok = [i for i in range(len(devs)) if n[i] < cap[i]]
        if not ok: return None
        i = min(ok, key=lambda i: (n[i] + 1) / devs[i].tok_s); n[i] += 1
    return n


def plan_pipeline(devs, n_layers, layer_gb=0.0, hidden=4096, ubatch=512, act_dtype_bytes=2):
    """Try every device subset (in the given order = pipeline order) and return the fastest predicted Plan."""
    act = hidden * act_dtype_bytes; best = None
    for k in range(1, len(devs) + 1):
        for sub in combinations(range(len(devs)), k):
            sd = [devs[i] for i in sub]; n = _split(sd, n_layers, layer_gb)
            if n is None or 0 in n: continue
            t = [_stage_time(d, m, n_layers, ubatch, act, j == len(sd) - 1) for j, (d, m) in enumerate(zip(sd, n))]
            tps = ubatch / max(t)
            if best is None or tps > best.tok_s:
                layers = [0] * len(devs)
                for i, m in zip(sub, n): layers[i] = m
                best = Plan([d.name for d in devs], layers, tps, sd[t.index(max(t))].name)
    return best


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser(description='seedplane plan: layer split for llama.cpp -ts from per-device speeds')
    ap.add_argument('--device', action='append', required=True, help='NAME:TOK_S[:MEM_GB[:LINK_GBPS[:LINK_MS]]], in pipeline order')
    ap.add_argument('--layers', type=int, required=True); ap.add_argument('--layer-gb', type=float, default=0.0)
    ap.add_argument('--hidden', type=int, default=4096); ap.add_argument('--ubatch', type=int, default=512)
    a = ap.parse_args()
    ds = []
    for s in a.device:
        f = s.split(':'); ds.append(Device(f[0], *map(float, f[1:])))
    p = plan_pipeline(ds, a.layers, a.layer_gb, a.hidden, a.ubatch)
    if p is None: raise SystemExit('no device subset has enough memory for all layers')
    print(json.dumps({'tensor_split': p.tensor_split, 'layers': dict(zip(p.devices, p.layers)), 'predicted_tok_s': round(p.tok_s, 1), 'bottleneck': p.bottleneck}, indent=1))
