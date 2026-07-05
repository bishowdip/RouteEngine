"""Task 2 benchmarks: pathfinding runtime + representation trade-off.

Run:  python -m src.bench.bench_task2
Generates into figures/:
  * task2_runtime_scaling.png   -- Dijkstra/Prim/Bellman-Ford vs graph size
  * task2_list_vs_matrix.png    -- adjacency list vs matrix on sparse vs dense
  * task2_route_map.html        -- a real shortest path drawn on a map
and prints the negative-weight demonstration (Dijkstra fails, BF succeeds).
"""

from __future__ import annotations

import random

from src.bench.harness import measure
from src.bench.plots import plot_measured_vs_theory
from src.data.loader import (
    connected_subgraph,
    load_city_graph,
    random_dense_graph,
    synthetic_city_graph,
)
from src.graph.bellman_ford import bellman_ford
from src.graph.dijkstra import dijkstra, reconstruct_path
from src.graph.graph import Graph
from src.graph.prim import prim_mst

SEED = 42


def bench_runtime_scaling():
    """Dijkstra & Prim on sparse city graphs; Bellman-Ford on modest sizes."""
    sizes = [100, 250, 500, 1000, 2000, 4000]
    bf_sizes = [100, 250, 500, 1000]   # O(V*E): keep modest, by design
    dij, prim_t, bf_t = [], [], []
    rng = random.Random(SEED)
    for n in sizes:
        g = synthetic_city_graph(n, seed=SEED)
        src = rng.choice(g.nodes())
        dij.append(measure(lambda: dijkstra(g, src), label="dijkstra", n=n, repeats=5).mean * 1e3)
        prim_t.append(measure(lambda: prim_mst(g, src), label="prim", n=n, repeats=5).mean * 1e3)
        print(f"n={n:<5} dijkstra={dij[-1]:.3f}  prim={prim_t[-1]:.3f} ms")
    for n in bf_sizes:
        g = synthetic_city_graph(n, seed=SEED)
        src = rng.choice(g.nodes())
        bf_t.append(measure(lambda: bellman_ford(g, src), label="bellman-ford", n=n, repeats=3).mean * 1e3)
        print(f"n={n:<5} bellman-ford={bf_t[-1]:.3f} ms")

    plot_measured_vs_theory(
        {
            "Dijkstra (heap)": {"n": sizes, "ms": dij},
            "Prim (heap)": {"n": sizes, "ms": prim_t},
            "Bellman-Ford": {"n": bf_sizes, "ms": bf_t},
        },
        title="Task 2: pathfinding runtime on sparse city graphs",
        theory_for={
            "Dijkstra (heap)": "O(n log n)",
            "Prim (heap)": "O(n log n)",
            "Bellman-Ford": "O(n^2)",   # E ~ O(V) on sparse graph -> V*E ~ V^2
        },
        filename="task2_runtime_scaling.png",
        xlabel="number of nodes V",
    )


def _dijkstra_on_matrix(mat, order, src_idx):
    """A deliberately matrix-backed Dijkstra: O(V^2) neighbour scan per node."""
    n = len(order)
    INF = float("inf")
    dist = [INF] * n
    dist[src_idx] = 0.0
    settled = [False] * n
    for _ in range(n):
        u, best = -1, INF
        for i in range(n):              # O(V) min scan -> O(V^2) total
            if not settled[i] and dist[i] < best:
                best, u = dist[i], i
        if u == -1:
            break
        settled[u] = True
        row = mat[u]
        for v in range(n):              # O(V) neighbour scan
            w = row[v]
            if w != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


def bench_list_vs_matrix():
    """Compare list-backed Dijkstra vs matrix-backed Dijkstra on sparse & dense."""
    sizes = [50, 100, 200, 400, 600]
    sparse_list, sparse_mat, dense_list, dense_mat = [], [], [], []
    for n in sizes:
        sparse = synthetic_city_graph(n, seed=SEED)
        dense = random_dense_graph(n, density=0.5, seed=SEED)
        for g, lst, matl in ((sparse, sparse_list, sparse_mat),
                             (dense, dense_list, dense_mat)):
            src = g.nodes()[0]
            lst.append(measure(lambda: dijkstra(g, src), label="list", n=n, repeats=4).mean * 1e3)
            mat, order = g.to_adjacency_matrix()
            si = order.index(src)
            matl.append(measure(lambda: _dijkstra_on_matrix(mat, order, si),
                                label="matrix", n=n, repeats=4).mean * 1e3)
        print(f"n={n:<4} sparse: list={sparse_list[-1]:.3f} mat={sparse_mat[-1]:.3f} | "
              f"dense: list={dense_list[-1]:.3f} mat={dense_mat[-1]:.3f} ms")

    plot_measured_vs_theory(
        {
            "list, sparse city": {"n": sizes, "ms": sparse_list},
            "matrix, sparse city": {"n": sizes, "ms": sparse_mat},
            "list, dense graph": {"n": sizes, "ms": dense_list},
            "matrix, dense graph": {"n": sizes, "ms": dense_mat},
        },
        title="Task 2: adjacency list vs matrix (sparse vs dense)",
        theory_for={"matrix, sparse city": "O(n^2)", "matrix, dense graph": "O(n^2)"},
        filename="task2_list_vs_matrix.png",
        xlabel="number of nodes V",
    )


def demo_negative_weight():
    """Concrete demonstration: Dijkstra fails on a negative edge; BF succeeds."""
    print("\n=== Negative-weight demonstration ===")
    g = Graph(directed=True)
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 5)
    g.add_edge(1, 2, -3)   # true cost 0->1->2 = 1, but Dijkstra settles 2 at 5
    g.add_edge(2, 3, 2)
    try:
        dijkstra(g, 0)
        print("Dijkstra: returned a result (UNSAFE on negative weights)")
    except ValueError as e:
        print(f"Dijkstra: refused -> {e}")
    dist, _, neg = bellman_ford(g, 0)
    print(f"Bellman-Ford: dist={dist}  negative_cycle={neg}")
    print("  -> correct cost to node 2 is 1 (via 0->1->2), not 5.")


def render_real_route():
    """Draw a real Dijkstra route on a folium map (OSM if available)."""
    print("\n=== Map visualisation ===")
    g, source_kind = load_city_graph(max_nodes=4000)
    print(f"graph source: {source_kind}; {g}")
    g = connected_subgraph(g, min(g.num_nodes, 4000))
    nodes = g.nodes()
    src, tgt = nodes[0], nodes[-1]
    dist, prev = dijkstra(g, src, target=tgt)
    path = reconstruct_path(prev, src, tgt)
    if path:
        from src.graph.route_map import render_route
        render_route(g, path)
        print(f"route: {len(path)} nodes, cost {dist[tgt]:.1f} m ({source_kind} data)")
    else:
        print("no path between chosen endpoints")


def main():
    print("=== Task 2: runtime scaling ===")
    bench_runtime_scaling()
    print("\n=== Task 2: list vs matrix ===")
    bench_list_vs_matrix()
    demo_negative_weight()
    render_real_route()


if __name__ == "__main__":
    main()
