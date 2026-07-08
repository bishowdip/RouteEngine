"""Benchmark: A* vs Dijkstra search effort on the real city graph.

Run:  python -m src.bench.bench_astar
Generates figures/task2_astar_vs_dijkstra.png and prints the node-expansion
table. A* with the admissible haversine heuristic should expand markedly fewer
nodes than Dijkstra for the same point-to-point queries, while returning the
same optimal cost -- the empirical payoff of the heuristic discussed in Task 2.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.plots import save  # noqa: E402
from src.data.loader import load_city_graph  # noqa: E402
from src.graph.astar import astar  # noqa: E402

SEED = 42


def _dijkstra_expanded(graph, src, tgt):
    """Count nodes Dijkstra settles before reaching the target."""
    from src.structures.min_heap import MinHeap
    dist = {src: 0.0}
    settled = set()
    pq = MinHeap()
    pq.insert(0.0, src)
    expanded = 0
    while not pq.is_empty():
        d, u = pq.extract_min()
        if u in settled:
            continue
        settled.add(u)
        expanded += 1
        if u == tgt:
            return dist[u], expanded
        for v, w in graph.neighbours(u):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                pq.push_or_decrease(nd, v)
    return float("inf"), expanded


def main():
    g, src_kind = load_city_graph(max_nodes=4000)
    print(f"graph: {g} ({src_kind} data)")
    rng = random.Random(SEED)
    nodes = [u for u in g.nodes() if g.coords.get(u) and g.coords[u][0] is not None]

    dij_exp, astar_exp = [], []
    for _ in range(40):
        s, t = rng.sample(nodes, 2)
        d_cost, d_exp = _dijkstra_expanded(g, s, t)
        a_cost, _, a_exp = astar(g, s, t)
        if d_cost == float("inf"):
            continue
        assert abs(d_cost - a_cost) < 1e-6, "A* and Dijkstra disagree on cost"
        dij_exp.append(d_exp)
        astar_exp.append(a_exp)

    mean_d = sum(dij_exp) / len(dij_exp)
    mean_a = sum(astar_exp) / len(astar_exp)
    print(f"mean nodes expanded -- Dijkstra: {mean_d:.0f}, A*: {mean_a:.0f} "
          f"({mean_a / mean_d * 100:.0f}% of Dijkstra)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dij_exp, astar_exp, alpha=0.6)
    lim = max(max(dij_exp), max(astar_exp))
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.6, label="equal effort")
    ax.set_title("Task 2: A* vs Dijkstra -- nodes expanded per query\n"
                 "points below the line: A* searched less for the same optimal route")
    ax.set_xlabel("Dijkstra nodes expanded")
    ax.set_ylabel("A* nodes expanded")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "task2_astar_vs_dijkstra.png")


if __name__ == "__main__":
    main()
