"""Tests for the or-opt TSP local search."""

import random

from src.heuristics.nearest_neighbour import nearest_neighbour
from src.heuristics.or_opt import or_opt
from src.heuristics.tsp import TSPInstance, held_karp


def _instance(n, seed):
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return TSPInstance(pts)


def test_or_opt_never_worsens_and_valid():
    for seed in range(10):
        inst = _instance(11, seed)
        start, start_len = nearest_neighbour(inst)
        tour, length = or_opt(inst, start)
        assert set(tour) == set(range(inst.n))           # valid permutation
        assert length <= start_len + 1e-9                # never worse than start
        _, opt = held_karp(inst)
        assert length >= opt - 1e-6                       # cannot beat optimum


def test_or_opt_close_to_optimum():
    gaps = []
    for seed in range(6):
        inst = _instance(10, seed)
        _, opt = held_karp(inst)
        _, length = or_opt(inst, nearest_neighbour(inst)[0])
        gaps.append((length - opt) / opt)
    assert sum(gaps) / len(gaps) < 0.10
