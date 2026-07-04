"""2-opt hill-climbing local search for TSP -- ST5003CEM Task 4.

Start from an initial tour (default: nearest-neighbour) and repeatedly apply the
best improving 2-opt move -- reverse a tour segment whenever doing so shortens
the tour -- until no improving move remains (a local optimum).

A 2-opt move replaces edges (a,b) and (c,d) with (a,c) and (b,d) by reversing
the segment between them. One full neighbourhood scan is O(n^2); the number of
improving passes is small in practice.

Limitation (the point of contrasting it with SA): pure hill climbing only ever
descends, so it gets trapped in the first local optimum it reaches and cannot
escape. Simulated annealing relaxes exactly this.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.heuristics.nearest_neighbour import nearest_neighbour
from src.heuristics.tsp import TSPInstance


def two_opt(inst: TSPInstance, initial: Optional[List[int]] = None,
            max_passes: int = 1000) -> Tuple[List[int], float]:
    dist = inst.dist
    tour = list(initial) if initial is not None else nearest_neighbour(inst)[0]
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                if d == a:
                    continue
                # change in length if we reverse tour[i+1 .. j]
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                if delta < -1e-12:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
                    b = tour[i + 1]   # b changed after the reversal
    return tour, inst.tour_length(tour)
