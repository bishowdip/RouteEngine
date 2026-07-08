"""Benchmarks comparing alternative algorithms for the same problem.

Run:  python -m src.bench.bench_comparisons
Generates into figures/:
  * cmp_mst_prim_vs_kruskal.png   -- two MST algorithms, runtime vs graph size
  * cmp_colouring_exact_vs_dsatur.png -- exact backtracking vs DSATUR heuristic
and prints the supporting tables. These reinforce the critical-evaluation theme:
different algorithms for one task trade runtime against guarantees.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.harness import measure  # noqa: E402
from src.bench.plots import plot_measured_vs_theory, save  # noqa: E402
from src.data.loader import synthetic_city_graph  # noqa: E402
from src.graph.graph import Graph  # noqa: E402
from src.graph.kruskal import kruskal_mst  # noqa: E402
from src.graph.prim import prim_mst  # noqa: E402
from src.strategies.backtracking import colour_pruned  # noqa: E402
from src.strategies.dsatur import dsatur_colouring  # noqa: E402

SEED = 42


def bench_mst():
    sizes = [100, 250, 500, 1000, 2000, 4000]
    prim_t, kruskal_t = [], []
    for n in sizes:
        g = synthetic_city_graph(n, seed=SEED)
        prim_t.append(measure(lambda: prim_mst(g), label="prim", n=n, repeats=5).mean * 1e3)
        kruskal_t.append(measure(lambda: kruskal_mst(g), label="kruskal", n=n, repeats=5).mean * 1e3)
        # sanity: same total weight
        assert abs(prim_mst(g)[1] - kruskal_mst(g)[1]) < 1e-6
        print(f"n={n:<5} prim={prim_t[-1]:.3f}  kruskal={kruskal_t[-1]:.3f} ms")
    plot_measured_vs_theory(
        {"Prim (heap)": {"n": sizes, "ms": prim_t},
         "Kruskal (sort + union-find)": {"n": sizes, "ms": kruskal_t}},
        title="MST: Prim vs Kruskal on sparse city graphs (identical tree weight)",
        theory_for={"Prim (heap)": "O(n log n)", "Kruskal (sort + union-find)": "O(n log n)"},
        filename="cmp_mst_prim_vs_kruskal.png",
        xlabel="number of nodes V",
    )


def _random_graph(n, p, seed):
    rng = random.Random(seed)
    g = Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j, 1.0)
    return g


def bench_colouring():
    """Exact backtracking (small only) vs DSATUR heuristic: colours and time."""
    print("\n=== colouring: exact backtracking vs DSATUR ===")
    sizes = [6, 8, 10, 12, 14, 16, 20, 40, 80]
    dsatur_colours, dsatur_time = [], []
    exact_sizes, exact_time = [], []
    for n in sizes:
        g = _random_graph(n, 0.3, SEED + n)
        _, kd = dsatur_colouring(g)
        dt = measure(lambda: dsatur_colouring(g), label="dsatur", n=n, repeats=5).mean * 1e3
        dsatur_colours.append(kd)
        dsatur_time.append(dt)
        line = f"n={n:<3} DSATUR: {kd} colours, {dt:.3f} ms"
        if n <= 16:  # exact backtracking only stays tractable for small n
            et = measure(lambda: colour_pruned(g, kd), label="exact", n=n, repeats=3).mean * 1e3
            exact_sizes.append(n)
            exact_time.append(et)
            line += f"  | exact(k={kd}): {et:.3f} ms"
        print(line)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, dsatur_time, "o-", color="seagreen", label="DSATUR heuristic")
    ax.plot(exact_sizes, exact_time, "s-", color="crimson", label="exact backtracking")
    ax.set_yscale("log")
    ax.set_title("Colouring: DSATUR scales while exact backtracking explodes")
    ax.set_xlabel("number of vertices")
    ax.set_ylabel("runtime (ms, log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "cmp_colouring_exact_vs_dsatur.png")


def main():
    print("=== MST: Prim vs Kruskal ===")
    bench_mst()
    bench_colouring()


if __name__ == "__main__":
    main()
