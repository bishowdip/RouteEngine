"""Cheapest-insertion construction heuristic for TSP.

Start from a 2-city sub-tour and repeatedly insert the city whose insertion
*increases* the tour length the least, between the pair of adjacent tour cities
that minimises the increase. Insertion heuristics generally beat nearest
neighbour because they keep the tour globally compact rather than chasing the
nearest unvisited city.

Complexity: O(n^3) in this straightforward form (n insertions, each scanning all
remaining cities against all tour edges). A good, still-fast construction to seed
local search.
"""

from __future__ import annotations

from typing import List, Tuple

from src.heuristics.tsp import TSPInstance


def cheapest_insertion(inst: TSPInstance, start: int = 0) -> Tuple[List[int], float]:
    n = inst.n
    dist = inst.dist
    if n == 1:
        return [start], 0.0

    # seed with start and its nearest city
    nearest = min((c for c in range(n) if c != start), key=lambda c: dist[start][c])
    tour = [start, nearest]
    in_tour = {start, nearest}

    while len(tour) < n:
        best_city, best_pos, best_delta = None, None, float("inf")
        for c in range(n):
            if c in in_tour:
                continue
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                delta = dist[a][c] + dist[c][b] - dist[a][b]   # insertion cost
                if delta < best_delta:
                    best_delta, best_city, best_pos = delta, c, i + 1
        tour.insert(best_pos, best_city)
        in_tour.add(best_city)

    return tour, inst.tour_length(tour)
