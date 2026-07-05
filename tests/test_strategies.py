"""Task 3 -- DP / greedy / backtracking correctness.

DP and greedy are validated against brute-force optima on small instances;
backtracking colourings are checked for properness and the pruned/naive variants
must agree on feasibility.
"""

import random

import pytest

from src.graph.graph import Graph
from src.strategies.backtracking import (
    colour_naive,
    colour_pruned,
    verify_colouring,
)
from src.strategies.greedy import (
    select_activities,
    select_activities_bruteforce,
)
from src.strategies.knapsack import (
    knapsack_bruteforce,
    knapsack_full_table,
    knapsack_space_optimised,
)

SEED = 13


# ----------------------------------------------------------------- knapsack
def test_knapsack_textbook_instance():
    # classic CLRS-style instance
    w = [1, 3, 4, 5]
    v = [1, 4, 5, 7]
    best, chosen, _ = knapsack_full_table(w, v, 7)
    assert best == 9                       # items {3,4} -> weight 7, value 4+5
    assert sum(w[i] for i in chosen) <= 7
    assert sum(v[i] for i in chosen) == 9


def test_knapsack_matches_bruteforce_and_space_opt():
    rng = random.Random(SEED)
    for _ in range(60):
        n = rng.randint(1, 12)
        w = [rng.randint(1, 10) for _ in range(n)]
        v = [float(rng.randint(1, 20)) for _ in range(n)]
        cap = rng.randint(0, 30)
        best, chosen, _ = knapsack_full_table(w, v, cap)
        assert best == knapsack_bruteforce(w, v, cap)         # optimal
        assert best == knapsack_space_optimised(w, v, cap)    # O(W) agrees
        assert sum(w[i] for i in chosen) <= cap               # feasible
        assert abs(sum(v[i] for i in chosen) - best) < 1e-9


def test_knapsack_rejects_bad_input():
    with pytest.raises(ValueError):
        knapsack_full_table([1, 2], [1.0], 5)        # mismatched lengths
    with pytest.raises(ValueError):
        knapsack_full_table([1], [1.0], -1)          # negative capacity


# ------------------------------------------------------------------ greedy
def test_activity_selection_classic():
    acts = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9),
            (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
    chosen = select_activities(acts)
    # known optimum size for this CLRS instance is 4
    assert len(chosen) == 4
    # verify pairwise compatibility
    iv = sorted(acts[i] for i in chosen)
    assert all(iv[k][1] <= iv[k + 1][0] for k in range(len(iv) - 1))


def test_greedy_matches_bruteforce_optimum():
    rng = random.Random(SEED)
    for _ in range(80):
        n = rng.randint(1, 12)
        acts = []
        for _ in range(n):
            s = rng.randint(0, 20)
            acts.append((s, s + rng.randint(1, 10)))
        assert len(select_activities(acts)) == select_activities_bruteforce(acts)


# -------------------------------------------------------------- backtracking
def _cycle_graph(n: int) -> Graph:
    g = Graph()
    for i in range(n):
        g.add_edge(i, (i + 1) % n, 1.0)
    return g


def test_colouring_even_cycle_two_colourable():
    g = _cycle_graph(6)                     # even cycle -> 2-colourable
    res = colour_pruned(g, 2)
    assert res.found
    assert verify_colouring(g, res.colours)


def test_colouring_odd_cycle_needs_three():
    g = _cycle_graph(5)                     # odd cycle -> NOT 2-colourable
    assert not colour_pruned(g, 2).found
    res3 = colour_pruned(g, 3)
    assert res3.found and verify_colouring(g, res3.colours)


def test_pruned_and_naive_agree_but_pruned_explores_less():
    g = _cycle_graph(5)
    pruned = colour_pruned(g, 3)
    naive = colour_naive(g, 3)
    assert pruned.found == naive.found
    # pruning must visit strictly fewer search nodes on a constrained graph
    assert pruned.nodes_explored < naive.nodes_explored
