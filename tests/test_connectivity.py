"""Tests for articulation points, bridges and Eulerian trails."""

import networkx as nx
import pytest

from src.data.loader import synthetic_city_graph
from src.graph.connectivity import find_bridges_and_articulation
from src.graph.eulerian import euler_status, eulerian_trail
from src.graph.graph import Graph


def _line(n):
    g = Graph()
    for i in range(n - 1):
        g.add_edge(i, i + 1, 1.0)        # a path graph: every internal node/edge is critical
    return g


# ------------------------------------------------------ articulation / bridges
def test_path_graph_all_internal_critical():
    g = _line(5)                          # 0-1-2-3-4
    bridges, arts = find_bridges_and_articulation(g)
    assert sorted(bridges) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert arts == {1, 2, 3}              # endpoints are not articulation points


def test_cycle_has_no_critical_parts():
    g = Graph()
    n = 6
    for i in range(n):
        g.add_edge(i, (i + 1) % n, 1.0)   # a cycle is 2-edge-connected
    bridges, arts = find_bridges_and_articulation(g)
    assert bridges == []
    assert arts == set()


def test_bridges_match_networkx_oracle():
    for trial in range(8):
        g = synthetic_city_graph(80, seed=trial)
        bridges, _ = find_bridges_and_articulation(g)
        ref = {frozenset(e) for e in nx.bridges(_to_nx(g))}
        mine = {frozenset(b) for b in bridges}
        assert mine == ref


def test_articulation_rejects_directed():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1.0)
    with pytest.raises(ValueError):
        find_bridges_and_articulation(g)


def _to_nx(g):
    nxg = nx.Graph()
    nxg.add_nodes_from(g.nodes())
    for u, v, w in g.edges():
        nxg.add_edge(u, v, weight=w)
    return nxg


# ----------------------------------------------------------------- Eulerian
def test_euler_circuit_on_even_cycle():
    g = Graph()
    for i in range(4):
        g.add_edge(i, (i + 1) % 4, 1.0)   # square: all degree 2 -> circuit
    assert euler_status(g) == "circuit"
    trail = eulerian_trail(g)
    assert trail is not None
    assert len(trail) == g.num_edges + 1
    assert trail[0] == trail[-1]          # circuit returns to start


def test_euler_path_with_two_odd_vertices():
    g = Graph()
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    g.add_edge(2, 0, 1.0)
    g.add_edge(2, 3, 1.0)                 # nodes 2 and 3 become odd -> path
    assert euler_status(g) == "path"
    trail = eulerian_trail(g)
    assert trail is not None and len(trail) == g.num_edges + 1


def test_euler_none_with_four_odd_vertices():
    g = Graph()
    # complete graph K4: every vertex degree 3 (odd) -> no Eulerian trail
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j, 1.0)
    assert euler_status(g) == "none"
    assert eulerian_trail(g) is None
