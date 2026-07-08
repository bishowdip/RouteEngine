"""Regenerate every figure for the report in one command.

Run:  python -m src.bench.run_all
Produces all task1..task5 figures (and the route map) into figures/.
"""

from __future__ import annotations

import time


def main() -> None:
    from src.bench import (
        bench_astar,
        bench_task1,
        bench_task2,
        bench_task3,
        bench_task4,
        bench_task5,
    )

    steps = [
        ("Task 1 -- data structures", bench_task1.main),
        ("Task 2 -- graph algorithms", bench_task2.main),
        ("Task 2 -- A* vs Dijkstra", bench_astar.main),
        ("Task 3 -- strategies", bench_task3.main),
        ("Task 4 -- TSP heuristics", bench_task4.main),
        ("Task 5 -- concurrency", bench_task5.main),
    ]
    t0 = time.perf_counter()
    for name, fn in steps:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        fn()
    print(f"\nAll figures regenerated in {time.perf_counter() - t0:.1f}s -> figures/")


if __name__ == "__main__":
    main()
