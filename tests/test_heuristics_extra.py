"""Tests for the additional TSP construction/metaheuristics."""

import random

from src.heuristics.grasp import grasp
from src.heuristics.insertion import cheapest_insertion
from src.heuristics.tsp import TSPInstance, held_karp


def _instance(n, seed):
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return TSPInstance(pts)


def _valid(tour, n):
    return len(tour) == n and set(tour) == set(range(n))


def test_cheapest_insertion_valid_and_reasonable():
    for seed in range(8):
        inst = _instance(9, seed)
        tour, length = cheapest_insertion(inst)
        assert _valid(tour, inst.n)
        _, opt = held_karp(inst)
        assert length >= opt - 1e-6                 # cannot beat optimum
        assert (length - opt) / opt < 0.15          # insertion stays close


def test_grasp_valid_and_close_to_optimum():
    gaps = []
    for seed in range(6):
        inst = _instance(10, seed)
        tour, length = grasp(inst, iterations=40, alpha=0.3, seed=seed)
        assert _valid(tour, inst.n)
        _, opt = held_karp(inst)
        assert length >= opt - 1e-6
        gaps.append((length - opt) / opt)
    assert sum(gaps) / len(gaps) < 0.05             # avg within 5% of optimum


def test_grasp_reproducible_with_seed():
    inst = _instance(12, 1)
    a = grasp(inst, iterations=20, seed=7)[1]
    b = grasp(inst, iterations=20, seed=7)[1]
    assert abs(a - b) < 1e-9
