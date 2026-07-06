"""GRASP -- Greedy Randomized Adaptive Search Procedure for TSP.

(Feo & Resende, 1995.) Each iteration has two phases:
  1. Construction -- a *randomized* greedy build: at each step pick the next city
     uniformly from a restricted candidate list (RCL) of the alpha-fraction best
     (nearest) candidates, rather than always the single nearest. This injects
     diversity so different runs explore different basins.
  2. Local search -- refine the constructed tour to a local optimum with 2-opt.
The best tour over all iterations is returned.

GRASP trades runtime for quality and, unlike a single greedy run, is not stuck
with one deterministic construction. Reproducible via the seed; quality improves
with more iterations and a well-chosen alpha.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from src.heuristics.hill_climbing import two_opt
from src.heuristics.tsp import TSPInstance


def _randomized_greedy(inst: TSPInstance, rng: random.Random, alpha: float) -> List[int]:
    n = inst.n
    dist = inst.dist
    start = rng.randrange(n)
    tour = [start]
    unvisited = set(range(n)) - {start}
    cur = start
    while unvisited:
        candidates = sorted(unvisited, key=lambda c: dist[cur][c])
        # restricted candidate list: the best alpha-fraction (at least one)
        k = max(1, int(len(candidates) * alpha))
        nxt = rng.choice(candidates[:k])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def grasp(
    inst: TSPInstance, *, iterations: int = 50, alpha: float = 0.3, seed: int = 42,
) -> Tuple[List[int], float]:
    """Return (best_tour, best_length) over ``iterations`` GRASP rounds."""
    rng = random.Random(seed)
    best_tour: List[int] = list(range(inst.n))
    best_len = inst.tour_length(best_tour)
    for _ in range(iterations):
        constructed = _randomized_greedy(inst, rng, alpha)
        tour, length = two_opt(inst, constructed)   # local-search refinement
        if length < best_len:
            best_tour, best_len = tour, length
    return best_tour, best_len
