"""Task 4 -- TSP exact solvers and heuristics.

Held-Karp and brute force must agree on the exact optimum (cross-validation of
the two exact methods); heuristics must return valid tours that are no better
than the optimum and, for SA/2-opt, close to it.
"""

import random

import pytest

from src.heuristics.hill_climbing import two_opt
from src.heuristics.nearest_neighbour import best_nearest_neighbour, nearest_neighbour
from src.heuristics.simulated_annealing import simulated_annealing
from src.heuristics.tsp import TSPInstance, brute_force, held_karp

SEED = 5


def _random_instance(n, seed):
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return TSPInstance(pts)


def _is_valid_tour(tour, n):
    return len(tour) == n and set(tour) == set(range(n))


def test_brute_force_matches_held_karp():
    for seed in range(10):
        inst = _random_instance(8, seed)
        bf_tour, bf_len = brute_force(inst)
        hk_tour, hk_len = held_karp(inst)
        assert abs(bf_len - hk_len) < 1e-6          # two exact methods agree
        assert _is_valid_tour(hk_tour, inst.n)


def test_heuristics_return_valid_tours_no_better_than_optimum():
    inst = _random_instance(9, SEED)
    _, opt = held_karp(inst)
    nn_tour, nn_len = nearest_neighbour(inst)
    hc_tour, hc_len = two_opt(inst, nn_tour)
    sa = simulated_annealing(inst, iterations=5000, seed=SEED)
    for tour in (nn_tour, hc_tour, sa.tour):
        assert _is_valid_tour(tour, inst.n)
    # no heuristic can beat the true optimum (allow tiny float slack)
    assert nn_len >= opt - 1e-6
    assert hc_len >= opt - 1e-6
    assert sa.length >= opt - 1e-6
    # local search should not be worse than its NN starting point
    assert hc_len <= nn_len + 1e-9


def test_2opt_and_sa_close_to_optimum():
    gaps_hc, gaps_sa = [], []
    for seed in range(6):
        inst = _random_instance(10, seed)
        _, opt = held_karp(inst)
        _, hc_len = two_opt(inst, nearest_neighbour(inst)[0])
        sa = simulated_annealing(inst, iterations=8000, seed=seed)
        gaps_hc.append((hc_len - opt) / opt)
        gaps_sa.append((sa.length - opt) / opt)
    # on tiny instances both should be within a few percent on average
    assert sum(gaps_sa) / len(gaps_sa) < 0.05
    assert sum(gaps_hc) / len(gaps_hc) < 0.10


def test_tour_length_validates_permutation():
    inst = _random_instance(5, 1)
    with pytest.raises(ValueError):
        inst.tour_length([0, 1, 2])          # not all cities
    with pytest.raises(ValueError):
        inst.tour_length([0, 1, 2, 3, 3])    # repeats


def test_brute_force_refuses_large_n():
    inst = _random_instance(13, 1)
    with pytest.raises(ValueError):
        brute_force(inst)


def test_best_nn_not_worse_than_single_start():
    inst = _random_instance(12, 2)
    _, single = nearest_neighbour(inst, 0)
    _, best = best_nearest_neighbour(inst)
    assert best <= single + 1e-9
