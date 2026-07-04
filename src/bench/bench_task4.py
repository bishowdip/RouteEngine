"""Task 4 benchmarks: TSP heuristic quality, runtime and SA convergence.

Run:  python -m src.bench.bench_task4
Generates into figures/:
  * task4_sa_convergence.png    -- SA best-cost vs iteration
  * task4_quality_vs_runtime.png-- heuristics: tour length vs wall-clock
and prints an optimality-gap table (vs Held-Karp on a small instance).
"""

from __future__ import annotations

import math
import random
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.plots import save  # noqa: E402
from src.data.loader import load_city_graph  # noqa: E402
from src.heuristics.hill_climbing import two_opt  # noqa: E402
from src.heuristics.nearest_neighbour import best_nearest_neighbour, nearest_neighbour  # noqa: E402
from src.heuristics.simulated_annealing import simulated_annealing  # noqa: E402
from src.heuristics.tsp import TSPInstance, held_karp  # noqa: E402

SEED = 42
EARTH_R = 6_371_000.0  # metres


def _project(lat, lon, lat0):
    """Equirectangular projection to metres so Euclidean distance is sensible."""
    x = EARTH_R * math.radians(lon) * math.cos(math.radians(lat0))
    y = EARTH_R * math.radians(lat)
    return x, y


def tsp_from_city(n: int, seed: int = SEED):
    """Sample n nodes (with coordinates) from the city graph -> TSPInstance."""
    g, src = load_city_graph(max_nodes=4000)
    rng = random.Random(seed)
    usable = [u for u in g.nodes() if g.coords.get(u) and g.coords[u][0] is not None]
    chosen = rng.sample(usable, min(n, len(usable)))
    lat0 = statistics.fmean(g.coords[u][0] for u in chosen)
    pts = [_project(g.coords[u][0], g.coords[u][1], lat0) for u in chosen]
    return TSPInstance(pts, labels=chosen), src


def optimality_gap_table():
    """Small instance with an exact Held-Karp optimum -> real % gaps."""
    print("=== Optimality gaps vs exact (Held-Karp), n=11 ===")
    inst, src = tsp_from_city(11, seed=SEED)
    _, opt = held_karp(inst)
    rows = []
    nn = nearest_neighbour(inst)
    bnn = best_nearest_neighbour(inst)
    hc = two_opt(inst, nn[0])
    sa = simulated_annealing(inst, iterations=10_000, seed=SEED)
    for name, length in [
        ("nearest-neighbour", nn[1]),
        ("best-start NN", bnn[1]),
        ("2-opt hill climb", hc[1]),
        ("simulated annealing", sa.length),
    ]:
        gap = (length - opt) / opt * 100
        rows.append((name, length, gap))
        print(f"  {name:<22} length={length:10.1f}  gap={gap:5.2f}%")
    print(f"  {'EXACT optimum':<22} length={opt:10.1f}  gap= 0.00%  (data: {src})")
    return rows


def sa_convergence_and_variance():
    """Plot SA convergence and report variance over several seeds."""
    print("\n=== SA convergence (n=60) ===")
    inst, src = tsp_from_city(60, seed=SEED)
    fig, ax = plt.subplots(figsize=(8, 5))
    finals = []
    for seed in range(5):
        sa = simulated_annealing(inst, iterations=30_000, alpha=0.9997, seed=seed)
        finals.append(sa.length)
        ax.plot(sa.history, alpha=0.7, label=f"seed {seed} (final {sa.length:.0f})")
    ax.set_title(f"Task 4: simulated-annealing convergence on {inst.n}-stop tour\n"
                 f"schedule: T0~avg-edge, alpha=0.9997, 30k iters ({src} data)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best tour length so far (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "task4_sa_convergence.png")
    print(f"  final lengths over 5 seeds: mean={statistics.fmean(finals):.1f} "
          f"+/- {statistics.pstdev(finals):.1f}")


def quality_vs_runtime():
    """Compare heuristics on a larger instance: tour length vs wall-clock."""
    print("\n=== Quality vs runtime (n=120) ===")
    inst, src = tsp_from_city(120, seed=SEED)
    results = {}

    t = time.perf_counter()
    _, nn_len = nearest_neighbour(inst)
    results["nearest-neighbour"] = (nn_len, (time.perf_counter() - t) * 1e3)

    t = time.perf_counter()
    _, bnn_len = best_nearest_neighbour(inst)
    results["best-start NN"] = (bnn_len, (time.perf_counter() - t) * 1e3)

    t = time.perf_counter()
    _, hc_len = two_opt(inst, nearest_neighbour(inst)[0])
    results["2-opt hill climb"] = (hc_len, (time.perf_counter() - t) * 1e3)

    t = time.perf_counter()
    sa = simulated_annealing(inst, iterations=40_000, alpha=0.9998, seed=SEED)
    results["simulated annealing"] = (sa.length, (time.perf_counter() - t) * 1e3)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (length, ms) in results.items():
        ax.scatter(ms, length, s=80)
        ax.annotate(name, (ms, length), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
        print(f"  {name:<22} length={length:10.1f}  time={ms:8.2f} ms")
    ax.set_xscale("log")
    ax.set_title(f"Task 4: TSP heuristic quality vs runtime, n={inst.n} ({src} data)\n"
                 "lower-left is better")
    ax.set_xlabel("runtime (ms, log scale)")
    ax.set_ylabel("tour length (m)")
    ax.grid(True, alpha=0.3)
    save(fig, "task4_quality_vs_runtime.png")


def main():
    optimality_gap_table()
    sa_convergence_and_variance()
    quality_vs_runtime()


if __name__ == "__main__":
    main()
