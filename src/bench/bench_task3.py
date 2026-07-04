"""Task 3 benchmarks: DP scaling + backtracking pruning effectiveness.

Run:  python -m src.bench.bench_task3
Generates into figures/:
  * task3_knapsack_scaling.png  -- knapsack DP runtime vs n (fixed W) and vs W
  * task3_pruning.png           -- nodes explored, pruned vs naive backtracking
and prints a small DP table for the report.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.harness import measure  # noqa: E402
from src.bench.plots import plot_measured_vs_theory, save  # noqa: E402
from src.graph.graph import Graph  # noqa: E402
from src.strategies.backtracking import colour_naive, colour_pruned  # noqa: E402
from src.strategies.knapsack import knapsack_full_table  # noqa: E402

SEED = 42


def print_dp_table():
    """Small worked instance whose DP table goes straight into the report."""
    w = [1, 3, 4, 5]
    v = [1.0, 4.0, 5.0, 7.0]
    cap = 7
    best, chosen, dp = knapsack_full_table(w, v, cap)
    print("Knapsack DP table  (rows = first i items, cols = capacity 0..W)")
    header = "      " + " ".join(f"{c:4d}" for c in range(cap + 1))
    print(header)
    for i, row in enumerate(dp):
        print(f"i={i}: " + " ".join(f"{x:4.0f}" for x in row))
    print(f"optimum value = {best}, chosen item indices = {chosen}\n")


def bench_knapsack_scaling():
    """Runtime vs item count n (W fixed) and vs capacity W (n fixed): both O(nW)."""
    rng = random.Random(SEED)
    ns = [50, 100, 200, 400, 800, 1600]
    W_fixed = 500
    t_vs_n = []
    for n in ns:
        w = [rng.randint(1, 50) for _ in range(n)]
        v = [float(rng.randint(1, 100)) for _ in range(n)]
        t_vs_n.append(measure(lambda: knapsack_full_table(w, v, W_fixed),
                              label="knapsack n", n=n, repeats=4).mean * 1e3)
        print(f"n={n:<5} (W={W_fixed}) -> {t_vs_n[-1]:.3f} ms")

    Ws = [100, 250, 500, 1000, 2000, 4000]
    n_fixed = 200
    w = [rng.randint(1, 50) for _ in range(n_fixed)]
    v = [float(rng.randint(1, 100)) for _ in range(n_fixed)]
    t_vs_W = []
    for W in Ws:
        t_vs_W.append(measure(lambda: knapsack_full_table(w, v, W),
                              label="knapsack W", n=W, repeats=4).mean * 1e3)
        print(f"W={W:<5} (n={n_fixed}) -> {t_vs_W[-1]:.3f} ms")

    plot_measured_vs_theory(
        {"vary n (W fixed)": {"n": ns, "ms": t_vs_n}},
        title="Task 3: knapsack DP runtime is linear in item count n (W fixed)",
        theory_for={"vary n (W fixed)": "O(n)"},
        filename="task3_knapsack_scaling.png",
        xlabel="number of items n",
    )
    plot_measured_vs_theory(
        {"vary W (n fixed)": {"n": Ws, "ms": t_vs_W}},
        title="Task 3: knapsack DP runtime is linear in capacity W (n fixed)\n"
              "-> O(nW) pseudo-polynomial",
        theory_for={"vary W (n fixed)": "O(n)"},
        filename="task3_knapsack_scaling_W.png",
        xlabel="capacity W",
    )


def _random_sparse_graph(n: int, p: float, seed: int) -> Graph:
    rng = random.Random(seed)
    g = Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j, 1.0)
    return g


def bench_pruning():
    """Nodes explored: pruned vs naive backtracking on k-colouring."""
    sizes = list(range(4, 11))     # naive is exponential -> cap at 10 nodes
    k = 3
    pruned_nodes, naive_nodes = [], []
    for n in sizes:
        g = _random_sparse_graph(n, p=0.4, seed=SEED + n)
        p = colour_pruned(g, k)
        nv = colour_naive(g, k)
        pruned_nodes.append(p.nodes_explored)
        naive_nodes.append(nv.nodes_explored)
        print(f"n={n:<3} pruned={p.nodes_explored:<8} naive={nv.nodes_explored:<10} "
              f"(found={p.found})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, naive_nodes, "s-", color="crimson", label="naive (generate-and-test)")
    ax.plot(sizes, pruned_nodes, "o-", color="seagreen", label="pruned (constraint check)")
    ax.set_yscale("log")
    ax.set_title("Task 3: backtracking 3-colouring -- search nodes explored\n"
                 "pruning collapses the exponential generate-and-test tree")
    ax.set_xlabel("number of vertices")
    ax.set_ylabel("search-tree nodes explored (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "task3_pruning.png")


def main():
    print("=== Task 3: knapsack DP table ===")
    print_dp_table()
    print("=== Task 3: knapsack scaling ===")
    bench_knapsack_scaling()
    print("\n=== Task 3: backtracking pruning ===")
    bench_pruning()


if __name__ == "__main__":
    main()
