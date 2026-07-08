"""Benchmark: Johnson vs Floyd-Warshall all-pairs shortest paths.

Run:  python -m src.bench.bench_allpairs
Generates figures/cmp_allpairs_johnson_vs_floyd.png and prints the table.
On sparse city graphs Johnson (O(VE + V^2 log V)) should overtake Floyd-Warshall
(O(V^3)) as V grows, since E ~ O(V) keeps Johnson near O(V^2 log V).
"""

from __future__ import annotations

from src.bench.harness import measure
from src.bench.plots import plot_measured_vs_theory
from src.data.loader import synthetic_city_graph
from src.graph.floyd_warshall import floyd_warshall
from src.graph.johnson import johnson

SEED = 42


def main():
    sizes = [50, 100, 150, 200, 300, 400]
    johnson_t, floyd_t = [], []
    for n in sizes:
        g = synthetic_city_graph(n, seed=SEED)
        johnson_t.append(measure(lambda: johnson(g), label="johnson", n=n, repeats=3).mean * 1e3)
        floyd_t.append(measure(lambda: floyd_warshall(g), label="floyd", n=n, repeats=3).mean * 1e3)
        print(f"n={n:<4} johnson={johnson_t[-1]:.2f}  floyd-warshall={floyd_t[-1]:.2f} ms")

    plot_measured_vs_theory(
        {"Johnson": {"n": sizes, "ms": johnson_t},
         "Floyd-Warshall": {"n": sizes, "ms": floyd_t}},
        title="All-pairs shortest paths on sparse graphs:\nJohnson vs Floyd-Warshall",
        theory_for={"Floyd-Warshall": "O(n^2)"},   # n^3 overlay too steep to fit; n^2 ref
        filename="cmp_allpairs_johnson_vs_floyd.png",
        xlabel="number of nodes V",
    )


if __name__ == "__main__":
    main()
