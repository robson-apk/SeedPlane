#!/usr/bin/env python3
"""Dependency-free tour of SeedPlane's current shard-window design.

This is an explanatory layout demo, not a model benchmark. For measured results
and runnable inference commands, see README.md and docs/API.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    shard: int = 128
    halo: int = 32
    sinks: int = 4

    def windows(self, length):
        for core_start in range(0, length, self.shard):
            core_end = min(length, core_start + self.shard)
            halo_start = max(0, core_start - self.halo)
            sink_end = min(self.sinks, halo_start)
            yield core_start, core_end, list(range(sink_end)), list(range(halo_start, core_end))


def ranges(values):
    if not values:
        return "none"
    return f"{values[0]}..{values[-1]}"


def main():
    length = 512
    plan = Plan()
    windows = list(plan.windows(length))

    print("SeedPlane — shard-window layout demo")
    print("=" * 50)
    print(f"sequence: {length} tokens | shard: {plan.shard} | halo: {plan.halo} | sinks: {plan.sinks}\n")
    print("Each core token keeps its original position. A worker reads only")
    print("the optional sinks, the preceding halo, and its owned core.\n")

    for worker, (c0, c1, sinks, visible) in enumerate(windows):
        halo = [i for i in visible if i < c0]
        print(
            f"worker {worker}: sinks={ranges(sinks):>7}  "
            f"halo={ranges(halo):>8}  core={c0}..{c1 - 1}"
        )

    full_pairs = length * length
    window_pairs = sum((len(sinks) + len(visible)) ** 2 for _, _, sinks, visible in windows)
    print("\nIllustrative attention-pair upper bound")
    print(f"full attention: {full_pairs:,}")
    print(f"shard windows:  {window_pairs:,} ({full_pairs / window_pairs:.1f}x less work)")
    print("\nThis script does not measure quality or speed. Run the commands in")
    print("docs/API.md with a real model, or inspect the raw experiment results.")


if __name__ == "__main__":
    main()
