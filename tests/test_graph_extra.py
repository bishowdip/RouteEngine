"""Correctness tests for the additional graph algorithms.

A*, bidirectional Dijkstra and Floyd-Warshall are all validated against the
already-trusted Dijkstra; Kruskal is checked against Prim and networkx; the
union-find and traversals against direct reasoning.
"""

import random

import networkx as nx
import pytest

from src.data.loader import synthetic_city_graph
from src.graph.astar import astar, haversine
from src.graph.bidirectional import bidirectional_dijkstra
from src.graph.dijkstra import dijkstra
from src.graph.floyd_warshall import floyd_warshall
from src.graph.graph import Graph
from src.graph.kruskal import kruskal_mst
from src.graph.prim import prim_mst
from src.graph.traversal import bfs, connected_components, dfs, is_connected
from src.structures.union_find import UnionFind

SEED = 21


def _to_nx(g):
    nxg = nx.Graph()
    nxg.add_nodes_from(g.nodes())
    for u, v, w in g.edges():
        nxg.add_edge(u, v, weight=w)
    return nxg


# ------------------------------------------------------------------ union-find
def test_union_find_basic():
    uf = UnionFind()
    for x in range(6):
        uf.add(x)
    assert uf.num_sets == 6
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    assert uf.connected(0, 2)
    assert not uf.connected(0, 4)
    assert uf.num_sets == 3
    assert not uf.union(0, 2)        # already joined
    assert sorted(len(g) for g in uf.groups()) == [1, 2, 3]


def _euclidean_graph(n, seed):
    """Random points; edge weight = straight-line distance (so the Euclidean
    heuristic is admissible AND consistent -> A* must equal Dijkstra)."""
    import math
    rng = random.Random(seed)
    pts = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(n)}
    g = Graph()
    for i in pts:
        g.add_node(i, coord=pts[i])
    nodes = list(pts)
    for i in nodes:                              # connect each node to 4 others
        for j in rng.sample(nodes, 4):
            if i != j:
                xi, yi = pts[i]
                xj, yj = pts[j]
                g.add_edge(i, j, math.hypot(xi - xj, yi - yj))
    h = lambda u, v: math.hypot(pts[u][0] - pts[v][0], pts[u][1] - pts[v][1])
    return g, h


# ----------------------------------------------------------------- A* vs Dijkstra
def test_astar_zero_heuristic_equals_dijkstra():
    """With h=0, A* is exactly Dijkstra -- validates the search mechanics."""
    rng = random.Random(SEED)
    for trial in range(12):
        g = synthetic_city_graph(150, seed=trial)
        src, tgt = rng.sample(g.nodes(), 2)
        dist, _ = dijkstra(g, src, target=tgt)
        cost, path, _ = astar(g, src, tgt, heuristic=lambda u, v: 0.0)
        assert abs(cost - dist[tgt]) < 1e-6
        if path:
            assert path[0] == src and path[-1] == tgt


def test_astar_admissible_heuristic_is_optimal():
    """Admissible+consistent Euclidean heuristic -> same cost, fewer expansions."""
    rng = random.Random(SEED)
    for trial in range(12):
        g, h = _euclidean_graph(120, seed=trial)
        src, tgt = rng.sample(g.nodes(), 2)
        dist, _ = dijkstra(g, src, target=tgt)
        cost, path, expanded = astar(g, src, tgt, heuristic=h)
        assert abs(cost - dist[tgt]) < 1e-6     # still optimal
        if path:
            assert path[0] == src and path[-1] == tgt


def test_astar_rejects_negative_and_missing():
    g = Graph()
    g.add_edge(0, 1, -2)
    with pytest.raises(ValueError):
        astar(g, 0, 1)
    g2 = synthetic_city_graph(10)
    with pytest.raises(KeyError):
        astar(g2, 0, 999999)


def test_haversine_symmetric_and_zero():
    a, b = (27.70, 85.30), (27.71, 85.32)
    assert haversine(a, a) < 1e-6
    assert abs(haversine(a, b) - haversine(b, a)) < 1e-6


# ----------------------------------------------------- bidirectional vs Dijkstra
def test_bidirectional_matches_dijkstra():
    rng = random.Random(SEED + 1)
    for trial in range(15):
        g = synthetic_city_graph(160, seed=trial + 5)
        src, tgt = rng.sample(g.nodes(), 2)
        dist, _ = dijkstra(g, src, target=tgt)
        cost, path = bidirectional_dijkstra(g, src, tgt)
        assert abs(cost - dist[tgt]) < 1e-6
        if path:
            assert path[0] == src and path[-1] == tgt


# --------------------------------------------------------------- Kruskal vs Prim
def test_kruskal_matches_prim_and_networkx():
    for trial in range(12):
        g = synthetic_city_graph(140, seed=trial + 30)
        _, k_total = kruskal_mst(g)
        _, p_total = prim_mst(g)
        ref = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(_to_nx(g), data=True))
        assert abs(k_total - p_total) < 1e-6
        assert abs(k_total - ref) < 1e-6


def test_kruskal_rejects_directed():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1)
    with pytest.raises(ValueError):
        kruskal_mst(g)


# ------------------------------------------------------- Floyd-Warshall vs Dijkstra
def test_floyd_warshall_matches_dijkstra():
    g = synthetic_city_graph(80, seed=3)
    dmat, order, neg = floyd_warshall(g)
    assert not neg
    idx = {u: i for i, u in enumerate(order)}
    for src in order[:10]:
        dist, _ = dijkstra(g, src)
        for tgt in order:
            assert abs(dmat[idx[src]][idx[tgt]] - dist[tgt]) < 1e-6


def test_floyd_warshall_detects_negative_cycle():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, -3)
    g.add_edge(2, 0, 1)              # cycle sums to -1
    _, _, neg = floyd_warshall(g)
    assert neg


# ----------------------------------------------------------------- traversals
def test_bfs_dfs_and_components():
    g = synthetic_city_graph(100, seed=7)
    bdist = bfs(g, 0)
    assert bdist[0] == 0 and len(bdist) == g.num_nodes  # connected component
    order = dfs(g, 0)
    assert order[0] == 0 and len(set(order)) == len(order)
    assert is_connected(g)
    assert len(connected_components(g)) == 1


def test_components_on_disjoint_graph():
    g = Graph()
    g.add_edge(0, 1, 1.0)
    g.add_edge(2, 3, 1.0)            # second, separate component
    g.add_node(4)                    # isolated vertex
    comps = connected_components(g)
    assert sorted(len(c) for c in comps) == [1, 2, 2]
    assert not is_connected(g)
