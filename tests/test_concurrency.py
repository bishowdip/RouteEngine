"""Task 5 -- parallel multi-source shortest paths correctness.

The whole point of the synchronisation work is that the threaded and
multiprocessing results are IDENTICAL to the sequential baseline -- no races, no
lost updates. We assert exact equality on a small city graph.
"""

from src.concurrency.parallel_shortest_paths import (
    multiprocessing_multi_source,
    results_equal,
    sequential_multi_source,
    threaded_multi_source,
)
from src.data.loader import synthetic_city_graph


def test_threaded_matches_sequential():
    g = synthetic_city_graph(300, seed=1)
    sources = g.nodes()[:25]
    seq = sequential_multi_source(g, sources)
    thr = threaded_multi_source(g, sources, n_workers=4)
    assert results_equal(seq, thr)


def test_multiprocessing_matches_sequential():
    g = synthetic_city_graph(300, seed=2)
    sources = g.nodes()[:25]
    seq = sequential_multi_source(g, sources)
    mp = multiprocessing_multi_source(g, sources, n_workers=2)
    assert results_equal(seq, mp)


def test_results_equal_detects_difference():
    a = {0: {0: 0.0, 1: 1.0}}
    b = {0: {0: 0.0, 1: 2.0}}
    assert not results_equal(a, b)
    assert results_equal(a, dict(a))
