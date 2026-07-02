"""Benchmark harness shared by every task.

Methodology (applied everywhere, per the brief):
  * time with ``time.perf_counter`` (monotonic, highest resolution);
  * one untimed warm-up run before measuring (primes caches / branch predictor);
  * repeat each measurement N times and report mean +/- standard deviation;
  * fix random seeds upstream so runs are reproducible.

Everything funnels through ``measure`` / ``scaling`` so timing is consistent and
the figures are regenerable from a single command.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List


@dataclass
class TimingResult:
    label: str
    n: int
    mean: float            # seconds
    stdev: float           # seconds
    runs: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.label:<28} n={self.n:<8} {self.mean*1e3:9.4f} ms +/- {self.stdev*1e3:.4f}"


def measure(
    func: Callable[[], Any],
    *,
    label: str,
    n: int,
    repeats: int = 7,
    warmup: bool = True,
) -> TimingResult:
    """Time a zero-arg callable ``repeats`` times; return mean/stdev.

    Build any per-run state *inside* ``func`` (or via a factory) so setup cost
    is not folded into the measured region.
    """
    if warmup:
        func()
    runs: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        runs.append(time.perf_counter() - start)
    return TimingResult(
        label=label,
        n=n,
        mean=statistics.fmean(runs),
        stdev=statistics.pstdev(runs) if len(runs) > 1 else 0.0,
        runs=runs,
    )


def scaling(
    factory: Callable[[int], Callable[[], Any]],
    sizes: Iterable[int],
    *,
    label: str,
    repeats: int = 7,
) -> List[TimingResult]:
    """Run ``measure`` across a sweep of input sizes.

    ``factory(n)`` returns the zero-arg callable to time for size ``n`` (it owns
    any data generation, which stays outside the timed region).
    """
    results = []
    for n in sizes:
        res = measure(factory(n), label=label, n=n, repeats=repeats)
        results.append(res)
        print(f"  {res}")
    return results


def results_to_records(results: List[TimingResult]) -> List[Dict[str, Any]]:
    """Flatten TimingResults into dict rows for a pandas DataFrame / CSV."""
    return [
        {
            "label": r.label,
            "n": r.n,
            "mean_s": r.mean,
            "stdev_s": r.stdev,
            "mean_ms": r.mean * 1e3,
        }
        for r in results
    ]
