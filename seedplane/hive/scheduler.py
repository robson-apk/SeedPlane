"""No-regression assignment for a batch of homogeneous independent requests."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BatchPlan:
    assignments: dict[str, int]
    predicted_makespan_ms: float


def plan_homogeneous_batch(task_count, worker_service_ms):
    """Minimize predicted batch makespan from measured per-request service times.

    The scheduler may leave a slow worker unused. Adding an eligible worker
    cannot worsen the *predicted* makespan because the optimizer can assign it
    zero work. This is exact for identical independent jobs with deterministic
    service-time estimates; it does not claim a hard wall-clock guarantee.
    """
    count = int(task_count)
    durations = {str(worker): float(ms) for worker, ms in worker_service_ms.items()}
    if count < 0 or not durations:
        raise ValueError("task_count must be non-negative and worker profiles non-empty")
    if any(not math.isfinite(ms) or ms <= 0 for ms in durations.values()):
        raise ValueError("worker service estimates must be finite and positive")
    if count == 0:
        return BatchPlan({worker: 0 for worker in durations}, 0.0)

    # Binary-search the smallest deadline by which the fleet can complete N
    # identical jobs. Each worker contributes floor(deadline / service time).
    low, high = 0.0, count * min(durations.values())
    for _ in range(80):
        middle = (low + high) / 2.0
        capacity = sum(int((middle + 1e-9) // service_ms)
                       for service_ms in durations.values())
        if capacity >= count:
            high = middle
        else:
            low = middle

    plan = {worker: 0 for worker in durations}
    remaining = count
    capacities = sorted(
        ((int((high + 1e-7) // service_ms), service_ms, worker)
         for worker, service_ms in durations.items()),
        key=lambda row: (row[1], row[2]))
    # Use the fastest capacity first. This avoids giving a slow island a tail
    # task when faster workers can finish the same batch sooner.
    for capacity, _, worker in capacities:
        assigned = min(remaining, capacity)
        plan[worker] = assigned
        remaining -= assigned
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("internal error: computed batch deadline is infeasible")
    makespan = max(plan[worker] * service_ms
                   for worker, service_ms in durations.items())
    return BatchPlan(plan, makespan)
