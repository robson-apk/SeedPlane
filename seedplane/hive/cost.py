"""Online measured cost model for HIVE work and lease sizing.

The first version intentionally stores observations in memory and uses robust
recent medians. It is a scheduling aid, not a learned policy or durable trace
store. A size bucket is log2(token_budget), so estimates generalize modestly
between nearby request sizes without pretending cost is linear.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import math
import statistics
import threading
from typing import Optional


def _bucket(value):
    return 0 if value <= 1 else int(math.ceil(math.log2(value)))


@dataclass(frozen=True)
class CostKey:
    kind: str
    model: str
    worker: str
    size_bucket: int
    batch_bucket: int = 0
    load_bucket: int = 0


@dataclass(frozen=True)
class CostEstimate:
    p50_ms: float
    p95_ms: float
    samples: int


class MeasuredCostModel:
    """Bounded per-key timing history with robust quantile estimates."""

    def __init__(self, *, history=128, min_samples=3):
        if history < 1 or min_samples < 1:
            raise ValueError("history and min_samples must be positive")
        self.history = int(history)
        self.min_samples = int(min_samples)
        self._samples = defaultdict(lambda: deque(maxlen=self.history))
        self._lock = threading.RLock()

    @staticmethod
    def key(kind, model, worker, token_budget, batch_size=1, load=0.0):
        return CostKey(str(kind), str(model), str(worker), _bucket(token_budget),
                       _bucket(batch_size), min(9, max(0, int(load * 10))))

    def observe(self, *, kind, model, worker, token_budget, duration_ms,
                batch_size=1, load=0.0):
        duration_ms = float(duration_ms)
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("duration_ms must be finite and positive")
        key = self.key(kind, model, worker, token_budget, batch_size, load)
        with self._lock:
            self._samples[key].append(duration_ms)
        return key

    def estimate(self, *, kind, model, worker, token_budget,
                 batch_size=1, load=0.0) -> Optional[CostEstimate]:
        key = self.key(kind, model, worker, token_budget, batch_size, load)
        with self._lock:
            values = list(self._samples.get(key, ()))
        if len(values) < self.min_samples:
            return None
        ordered = sorted(values)
        # Nearest-rank p95, useful as a conservative deadline estimate.
        p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        return CostEstimate(statistics.median(ordered), p95, len(ordered))

    @staticmethod
    def choose_lease_tokens(*, throughput_tokens_s, fixed_overhead_ms,
                            max_fixed_fraction=0.2, min_tokens=1, max_tokens=4096):
        """Find the smallest lease whose fixed-cost share meets the target.

        Returns ``None`` when there is no measured positive throughput. This
        helper only sizes divisible work (for example prefill/verify tiles);
        it does not split semantically indivisible generation requests.
        """
        rate = float(throughput_tokens_s)
        fixed = float(fixed_overhead_ms)
        fraction = float(max_fixed_fraction)
        if rate <= 0 or fixed < 0 or not 0 < fraction < 1:
            raise ValueError("invalid throughput, overhead, or fixed-cost fraction")
        if fixed == 0:
            return int(min_tokens)
        compute_ms = fixed * (1.0 - fraction) / fraction
        needed = math.ceil(rate * compute_ms / 1000.0)
        if needed > max_tokens:
            return int(max_tokens)
        return int(max(min_tokens, needed))

    def snapshot(self):
        with self._lock:
            return {key: tuple(values) for key, values in self._samples.items()}
