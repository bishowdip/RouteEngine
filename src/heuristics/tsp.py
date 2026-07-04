"""Travelling Salesman Problem instance + exact solvers -- ST5003CEM Task 4.

Route-engine framing: a delivery driver must visit a chosen set of Kathmandu
stops and return to the depot, minimising total travel distance -- a TSP tour
over a subset of intersections, reusing the geographic coordinates from Task 2.

WHY TSP IS NP-HARD (reduction intuition, not "it's slow"):
The decision form of TSP ("is there a tour of length <= L?") is NP-complete by
reduction FROM the Hamiltonian-cycle problem (itself NP-complete). Given a graph
G in which we ask whether a Hamiltonian cycle exists, build a complete graph on
the same vertices with edge weight 1 for edges present in G and 2 for absent
edges, and set L = n. A tour of total length exactly n exists iff it uses only
weight-1 edges, i.e. iff G has a Hamiltonian cycle. The construction is
polynomial, so a polynomial TSP solver would solve Hamiltonian cycle in
polynomial time. Since no polynomial algorithm is known for an NP-complete
problem, TSP is NP-hard; the optimisation version is at least as hard.

Exact solvers here exist only to obtain the TRUE optimum on small instances so
the heuristics can be scored by real optimality gap:
  * brute force  : O(n!)        -- only for n <= ~10
  * Held-Karp DP : O(n^2 * 2^n) -- exact up to ~15-16 cities
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple


class TSPInstance:
    """A symmetric TSP over points with a precomputed distance matrix."""

    def __init__(self, points: Sequence[Tuple[float, float]],
                 labels: Optional[Sequence[int]] = None) -> None:
        if len(points) < 2:
            raise ValueError("TSP needs at least 2 points")
        self.points = list(points)
        self.n = len(points)
        self.labels = list(labels) if labels is not None else list(range(self.n))
        self.dist = self._build_distance_matrix()

    def _build_distance_matrix(self) -> List[List[float]]:
        n = self.n
        d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = self.points[i]
            for j in range(i + 1, n):
                xj, yj = self.points[j]
                # Euclidean on projected coords; fine for a city-scale subset.
                dij = math.hypot(xi - xj, yi - yj)
                d[i][j] = d[j][i] = dij
        return d

    def tour_length(self, tour: Sequence[int]) -> float:
        """Closed-tour length: sum of consecutive legs plus the return leg."""
        if len(tour) != self.n or set(tour) != set(range(self.n)):
            raise ValueError("tour must be a permutation of all cities")
        total = 0.0
        for i in range(len(tour)):
            total += self.dist[tour[i]][tour[(i + 1) % len(tour)]]
        return total


def brute_force(inst: TSPInstance) -> Tuple[List[int], float]:
    """Exact optimum by trying all (n-1)!/2 distinct tours. n <= ~10 only."""
    if inst.n > 11:
        raise ValueError(f"brute force refused for n={inst.n} (> 11); use Held-Karp")
    best_tour: List[int] = list(range(inst.n))
    best_len = inst.tour_length(best_tour)
    # fix city 0 to remove rotational symmetry
    for perm in itertools.permutations(range(1, inst.n)):
        tour = [0, *perm]
        length = inst.tour_length(tour)
        if length < best_len:
            best_len, best_tour = length, tour
    return best_tour, best_len


def held_karp(inst: TSPInstance) -> Tuple[List[int], float]:
    """Exact O(n^2 * 2^n) dynamic programming over visited-subsets. n <= ~16."""
    n = inst.n
    if n > 16:
        raise ValueError(f"Held-Karp refused for n={n} (> 16): 2^n too large")
    dist = inst.dist
    # dp[(mask, j)] = min cost to start at 0, visit exactly 'mask', end at j
    dp: Dict[Tuple[int, int], float] = {}
    parent: Dict[Tuple[int, int], int] = {}
    for j in range(1, n):
        dp[(1 << j, j)] = dist[0][j]
        parent[(1 << j, j)] = 0

    for mask in range(1 << n):
        for j in range(1, n):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            if prev_mask == 0:
                continue
            best = math.inf
            best_k = -1
            for k in range(1, n):
                if k == j or not (prev_mask & (1 << k)):
                    continue
                cand = dp.get((prev_mask, k), math.inf) + dist[k][j]
                if cand < best:
                    best, best_k = cand, k
            if best_k != -1:
                dp[(mask, j)] = best
                parent[(mask, j)] = best_k

    full = (1 << n) - 2  # all cities except 0
    best, last = math.inf, -1
    for j in range(1, n):
        cand = dp.get((full, j), math.inf) + dist[j][0]
        if cand < best:
            best, last = cand, j

    # reconstruct
    tour = [0]
    mask, j = full, last
    stack = []
    while j != 0:
        stack.append(j)
        nj = parent[(mask, j)]
        mask ^= (1 << j)
        j = nj
    tour.extend(reversed(stack))
    return tour, best
