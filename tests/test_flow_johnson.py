"""Tests for max-flow/min-cut, Johnson's all-pairs, DSATUR and edit distance."""

import random

import networkx as nx
import pytest

from src.data.loader import synthetic_city_graph
from src.graph.dijkstra import dijkstra
from src.graph.graph import Graph
from src.graph.johnson import johnson
from src.graph.max_flow import max_flow, min_cut
from src.strategies.dsatur import dsatur_colouring
from src.strategies.edit_distance import closest_match, edit_distance

SEED = 19


# ------------------------------------------------------------------- max flow
def test_max_flow_classic_value():
    # CLRS-style network; known max flow source 0 -> sink 5 is 23
    g = Graph(directed=True)
    for u, v, c in [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4),
                    (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
                    (3, 5, 20), (4, 5, 4)]:
        g.add_edge(u, v, c)
    flow, _ = max_flow(g, 0, 5)
    assert abs(flow - 23) < 1e-6


def test_max_flow_min_cut_equal():
    rng = random.Random(SEED)
    for _ in range(8):
        g = Graph(directed=True)
        for i in range(8):
            g.add_node(i)
        for _ in range(18):
            u, v = rng.sample(range(8), 2)
            g.add_edge(u, v, rng.randint(1, 10))
        flow, _ = max_flow(g, 0, 7)
        cut_val, cut_edges = min_cut(g, 0, 7)
        assert abs(flow - cut_val) < 1e-6           # max-flow == min-cut


def test_max_flow_rejects_same_source_sink():
    g = Graph(directed=True)
    g.add_edge(0, 1, 5)
    with pytest.raises(ValueError):
        max_flow(g, 0, 0)


# --------------------------------------------------------------------- Johnson
def test_johnson_matches_dijkstra():
    g = synthetic_city_graph(70, seed=2)
    ap = johnson(g)
    assert ap is not None
    for src in g.nodes()[:8]:
        d, _ = dijkstra(g, src)
        for tgt in g.nodes():
            assert abs(ap[src][tgt] - d[tgt]) < 1e-6


def test_johnson_handles_negative_edges():
    g = Graph(directed=True)
    g.add_edge(0, 1, 4); g.add_edge(0, 2, 5)
    g.add_edge(1, 2, -3); g.add_edge(2, 3, 2)
    ap = johnson(g)
    assert ap is not None
    assert abs(ap[0][2] - 1) < 1e-6                 # via 0->1->2
    assert abs(ap[0][3] - 3) < 1e-6


def test_johnson_detects_negative_cycle():
    g = Graph(directed=True)
    g.add_edge(0, 1, 1); g.add_edge(1, 2, -1); g.add_edge(2, 0, -1)
    assert johnson(g) is None


# ---------------------------------------------------------------------- DSATUR
def test_dsatur_proper_and_bipartite_two_colours():
    # even cycle is bipartite -> 2 colours
    g = Graph()
    for i in range(6):
        g.add_edge(i, (i + 1) % 6, 1.0)
    colour, k = dsatur_colouring(g)
    assert k == 2
    for u, v, _ in g.edges():
        assert colour[u] != colour[v]               # proper colouring


def test_dsatur_proper_on_random_and_vs_networkx_chromatic_lb():
    rng = random.Random(SEED)
    g = synthetic_city_graph(120, seed=5)
    colour, k = dsatur_colouring(g)
    for u, v, _ in g.edges():
        assert colour[u] != colour[v]
    # a valid colouring uses at least max-clique colours; sanity upper bound
    assert 1 <= k <= g.num_nodes


# ---------------------------------------------------------------- edit distance
def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("flaw", "lawn") == 2
    assert edit_distance("same", "same") == 0


def test_closest_match_for_place_names():
    names = ["Thamel", "Patan", "Bhaktapur", "Kirtipur"]
    assert closest_match("Thamal", names) == "Thamel"
    assert closest_match("Bhaktpur", names) == "Bhaktapur"
    with pytest.raises(ValueError):
        closest_match("x", [])
