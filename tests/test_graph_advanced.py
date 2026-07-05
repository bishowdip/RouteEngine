"""Tests for the advanced graph algorithms and the trie."""

import random

import pytest

from src.data.loader import synthetic_city_graph
from src.graph.dijkstra import dijkstra
from src.graph.graph import Graph
from src.graph.k_shortest import k_shortest_paths
from src.graph.scc import strongly_connected_components
from src.graph.topological_sort import topological_sort
from src.structures.trie import Trie

SEED = 33


# ----------------------------------------------------------- k-shortest paths
def test_k_shortest_first_is_dijkstra_and_sorted():
    rng = random.Random(SEED)
    g = synthetic_city_graph(120, seed=2)
    src, tgt = rng.sample(g.nodes(), 2)
    dist, _ = dijkstra(g, src, target=tgt)
    paths = k_shortest_paths(g, src, tgt, 4)
    assert paths, "expected at least one path"
    assert abs(paths[0][0] - dist[tgt]) < 1e-6      # first == shortest
    costs = [c for c, _ in paths]
    assert costs == sorted(costs)                   # ascending
    for _, p in paths:
        assert p[0] == src and p[-1] == tgt
        assert len(p) == len(set(p))                # loopless


def test_k_shortest_small_explicit():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1); g.add_edge(1, 3, 1)        # path A: 0-1-3 = 2
    g.add_edge(0, 2, 2); g.add_edge(2, 3, 2)        # path B: 0-2-3 = 4
    paths = k_shortest_paths(g, 0, 3, 2)
    assert [c for c, _ in paths] == [2.0, 4.0]


# -------------------------------------------------------------- topological sort
def test_topological_sort_valid_order():
    g = Graph(directed=True)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    for u, v in edges:
        g.add_edge(u, v, 1.0)
    order, is_dag = topological_sort(g)
    assert is_dag
    pos = {node: i for i, node in enumerate(order)}
    for u, v in edges:
        assert pos[u] < pos[v]                      # every edge points forward


def test_topological_sort_detects_cycle():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1); g.add_edge(1, 2, 1); g.add_edge(2, 0, 1)
    order, is_dag = topological_sort(g)
    assert order is None and not is_dag


def test_topological_sort_rejects_undirected():
    with pytest.raises(ValueError):
        topological_sort(synthetic_city_graph(5))


# ----------------------------------------------------------------------- SCC
def test_scc_partitions_directed_graph():
    g = Graph(directed=True)
    # one SCC {0,1,2}, then 3, then SCC {4,5}
    g.add_edge(0, 1, 1); g.add_edge(1, 2, 1); g.add_edge(2, 0, 1)
    g.add_edge(2, 3, 1); g.add_edge(3, 4, 1)
    g.add_edge(4, 5, 1); g.add_edge(5, 4, 1)
    sccs = strongly_connected_components(g)
    as_sets = sorted([sorted(c) for c in sccs])
    assert [0, 1, 2] in as_sets
    assert [4, 5] in as_sets
    assert [3] in as_sets
    assert sum(len(c) for c in sccs) == 6           # every node once


def test_scc_each_node_once_on_random():
    rng = random.Random(SEED)
    g = Graph(directed=True)
    for i in range(40):
        g.add_node(i)
    for _ in range(80):
        a, b = rng.randrange(40), rng.randrange(40)
        if a != b:
            g.add_edge(a, b, 1.0)
    sccs = strongly_connected_components(g)
    seen = [n for c in sccs for n in c]
    assert sorted(seen) == list(range(40))


# ----------------------------------------------------------------------- trie
def test_trie_insert_search_prefix():
    t = Trie()
    names = ["Thamel", "Thapathali", "Thimi", "Baneshwor", "Balaju"]
    for nm in names:
        t.insert(nm, value=len(nm))
    assert "Thamel" in t
    assert "Thame" not in t
    assert t.get("Thimi") == 5
    assert sorted(t.starts_with("Th")) == ["Thamel", "Thapathali", "Thimi"]
    assert sorted(t.starts_with("Ba")) == ["Balaju", "Baneshwor"]
    assert t.starts_with("Zzz") == []
    assert len(t) == 5
