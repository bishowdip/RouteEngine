"""Task 5 benchmarks: threading vs multiprocessing speedup + Amdahl's Law.

Run:  python -m src.bench.bench_task5
Generates into figures/:
  * task5_speedup.png   -- speedup vs worker count, threads vs processes,
                           with Amdahl's-Law prediction overlaid
and prints the correctness check and the timing table.

Must be run as a script (``__main__``) so multiprocessing can spawn workers.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.plots import save  # noqa: E402
from src.concurrency.parallel_shortest_paths import (  # noqa: E402
    multiprocessing_multi_source,
    results_equal,
    sequential_multi_source,
    threaded_multi_source,
)
from src.data.loader import load_city_graph  # noqa: E402

WORKER_COUNTS = [1, 2, 4, 8]


def _time(fn) -> float:
    fn()  # warm-up
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def amdahl(p_serial: float, workers: int) -> float:
    """Predicted speedup with serial fraction ``p_serial`` on ``workers`` cores."""
    return 1.0 / (p_serial + (1 - p_serial) / workers)


def main():
    # Heavier per-task workload (full city graph) + a scalar reducer so compute,
    # not pickling, dominates -- the regime where multiprocessing actually wins.
    g, src = load_city_graph(max_nodes=None)
    sources = g.nodes()[:96]   # 96 independent, CPU-bound Dijkstra runs
    R = "farness"              # each task returns one float (closeness metric)
    print(f"graph: {g} ({src} data); {len(sources)} source nodes; reducer={R}")

    # ---- correctness: threaded / mp results identical to sequential ----
    seq = sequential_multi_source(g, sources, reduce=R)
    thr = threaded_multi_source(g, sources, n_workers=4, reduce=R)
    mp = multiprocessing_multi_source(g, sources, n_workers=4, reduce=R)
    print(f"correctness: threaded==sequential -> {results_equal(seq, thr)}; "
          f"multiprocessing==sequential -> {results_equal(seq, mp)}")

    # ---- timing baseline ----
    t_seq = _time(lambda: sequential_multi_source(g, sources, reduce=R))
    # "fork" lets workers inherit the graph (no re-pickle) -> clean speedup on
    # POSIX. Fall back to the platform default if fork is unavailable (Windows).
    sm = "fork" if "fork" in __import__("multiprocessing").get_all_start_methods() else None
    print(f"\nsequential baseline: {t_seq*1e3:.1f} ms; mp start method: {sm or 'default'}")

    thr_speedup, mp_speedup = [], []
    for w in WORKER_COUNTS:
        t_thr = _time(lambda: threaded_multi_source(g, sources, n_workers=w, reduce=R))
        t_mp = _time(lambda: multiprocessing_multi_source(
            g, sources, n_workers=w, reduce=R, start_method=sm))
        thr_speedup.append(t_seq / t_thr)
        mp_speedup.append(t_seq / t_mp)
        print(f"  workers={w}: threading {t_thr*1e3:8.1f} ms (x{thr_speedup[-1]:.2f}) | "
              f"multiprocessing {t_mp*1e3:8.1f} ms (x{mp_speedup[-1]:.2f})")

    # Fit Amdahl serial fraction to the multiprocessing curve (rough).
    best = max(mp_speedup)
    max_w = WORKER_COUNTS[mp_speedup.index(best)]
    p_serial = max((max_w / best - 1) / (max_w - 1), 0.0) if best > 1 and max_w > 1 else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(WORKER_COUNTS, mp_speedup, "o-", color="seagreen", label="multiprocessing (measured)")
    ax.plot(WORKER_COUNTS, thr_speedup, "s-", color="crimson", label="threading (measured)")
    ax.plot(WORKER_COUNTS, [amdahl(p_serial, w) for w in WORKER_COUNTS], "--",
            color="seagreen", alpha=0.6, label=f"Amdahl (serial~{p_serial:.2f})")
    ax.plot(WORKER_COUNTS, WORKER_COUNTS, ":", color="gray", alpha=0.6, label="ideal linear")
    ax.set_title("Task 5: multi-source Dijkstra speedup\n"
                 "processes scale (escape the GIL); threads do not (CPU-bound)")
    ax.set_xlabel("worker count")
    ax.set_ylabel("speedup vs sequential")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "task5_speedup.png")


if __name__ == "__main__":
    main()
