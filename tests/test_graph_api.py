"""Tests for the Graph query/mutation API (has_edge, weights, removal, degree)."""

import pytest

from src.graph.graph import Graph


def _g():
    g = Graph()
    g.add_edge(0, 1, 5.0)
    g.add_edge(1, 2, 3.0)
    g.add_edge(0, 2, 9.0)
    return g


def test_has_edge_and_weight():
    g = _g()
    assert g.has_edge(0, 1) and g.has_edge(1, 0)     # undirected
    assert not g.has_edge(0, 3)
    assert g.get_edge_weight(0, 1) == 5.0
    assert g.get_edge_weight(0, 3) is None
    with pytest.raises(KeyError):
        g.get_edge_weight(99, 0)


def test_degree():
    g = _g()
    assert g.degree(0) == 2
    assert g.degree(2) == 2
    with pytest.raises(KeyError):
        g.degree(42)


def test_remove_edge_undirected():
    g = _g()
    assert g.remove_edge(0, 1)
    assert not g.has_edge(0, 1) and not g.has_edge(1, 0)
    assert not g.remove_edge(0, 1)                    # already gone
    assert g.num_edges == 2


def test_remove_node_clears_incident_edges():
    g = _g()
    assert g.remove_node(0)
    assert not g.has_node(0)
    assert not g.has_edge(1, 0) and not g.has_edge(2, 0)
    assert g.num_nodes == 2
    assert not g.remove_node(0)


def test_directed_remove_edge_one_way():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 0, 2.0)
    assert g.remove_edge(0, 1)
    assert not g.has_edge(0, 1)
    assert g.has_edge(1, 0)                            # reverse arc survives
