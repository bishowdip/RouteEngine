"""Tests for the data loader utilities (offline: synthetic graphs only)."""

import pytest

from src.data.loader import (
    as_travel_time_graph,
    connected_subgraph,
    random_dense_graph,
    synthetic_city_graph,
)
from src.graph.dijkstra import dijkstra


def test_synthetic_city_graph_is_connected_and_sparse():
    g = synthetic_city_graph(200, seed=1)
    assert g.num_nodes <= 200
    assert g.density() < 0.05            # road networks are sparse
    # reachability from node 0 covers the whole (largest) component
    dist, _ = dijkstra(g, g.nodes()[0])
    assert all(d < float("inf") for d in dist.values())


def test_connected_subgraph_size_and_connectivity():
    g = synthetic_city_graph(400, seed=2)
    sub = connected_subgraph(g, 100, seed=2)
    assert sub.num_nodes <= 100
    dist, _ = dijkstra(sub, sub.nodes()[0])
    assert all(d < float("inf") for d in dist.values())


def test_travel_time_scales_distances():
    g = synthetic_city_graph(80, seed=3)
    tt = as_travel_time_graph(g, default_kmph=36.0)   # 36 km/h = 10 m/s
    # every edge weight should be length / 10
    for (u, v, w_len), (_, _, w_tt) in zip(sorted(g.edges()), sorted(tt.edges())):
        assert abs(w_tt - w_len / 10.0) < 1e-6


def test_travel_time_rejects_bad_speed():
    g = synthetic_city_graph(20, seed=4)
    with pytest.raises(ValueError):
        as_travel_time_graph(g, default_kmph=0)


def test_random_dense_graph_density():
    g = random_dense_graph(40, density=0.5, seed=1)
    assert 0.3 < g.density() < 0.7        # roughly the requested density
