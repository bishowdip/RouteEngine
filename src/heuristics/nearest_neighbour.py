"""Nearest-neighbour greedy construction heuristic for TSP -- Task 4.

From a start city, repeatedly travel to the closest unvisited city; close the
tour at the end. O(n^2). Fast and simple but myopic -- early greedy choices can
force an expensive final return leg, so it typically lands 15-25% above optimum.
A natural contrast to the local-search heuristics (hill climbing / SA).
"""

from __future__ import annotations

from typing import List, Tuple

from src.heuristics.tsp import TSPInstance


def nearest_neighbour(inst: TSPInstance, start: int = 0) -> Tuple[List[int], float]:
    n = inst.n
    dist = inst.dist
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda c: dist[cur][c])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour, inst.tour_length(tour)


def best_nearest_neighbour(inst: TSPInstance) -> Tuple[List[int], float]:
    """Try every start city, keep the best tour -- a cheap quality boost."""
    best_tour, best_len = None, float("inf")
    for s in range(inst.n):
        tour, length = nearest_neighbour(inst, s)
        if length < best_len:
            best_tour, best_len = tour, length
    return best_tour, best_len
