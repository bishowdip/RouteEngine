"""Simulated annealing for TSP -- ST5003CEM Task 4.

Kirkpatrick, Gelatt & Vecchi (1983). Local search that probabilistically accepts
*worsening* moves to escape local optima, with the acceptance probability cooled
over time so the search settles into a good basin.

Acceptance rule for a candidate with cost change delta at temperature T:
    delta <= 0           -> always accept (improving / sideways)
    delta > 0            -> accept with probability exp(-delta / T)
Temperature decays geometrically: T <- alpha * T (0 < alpha < 1). High T early =
near-random exploration; low T late = near-greedy refinement.

Neighbour move: a random 2-opt segment reversal (same neighbourhood as the hill
climber, so the comparison is apples-to-apples).

The cooling schedule (T0, alpha, iterations) is reported with the results, and a
convergence curve (best cost vs iteration) is plotted. Seeded for reproducibility
but variance over seeds is also reported.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.heuristics.nearest_neighbour import nearest_neighbour
from src.heuristics.tsp import TSPInstance


@dataclass
class SAResult:
    tour: List[int]
    length: float
    history: List[float] = field(default_factory=list)   # best-so-far per iter
    schedule: dict = field(default_factory=dict)


def simulated_annealing(
    inst: TSPInstance,
    *,
    t0: float = None,            # type: ignore[assignment]
    alpha: float = 0.9995,
    iterations: int = 20_000,
    seed: int = 42,
    initial: Optional[List[int]] = None,
) -> SAResult:
    rng = random.Random(seed)
    dist = inst.dist
    n = inst.n

    current = list(initial) if initial is not None else nearest_neighbour(inst)[0]
    current_len = inst.tour_length(current)
    best, best_len = list(current), current_len

    # Default starting temperature: scale to a typical edge length so the early
    # acceptance probability of a representative move is ~0.5-0.8.
    if t0 is None:
        avg_edge = sum(dist[i][(i + 1) % n] for i in range(n)) / n
        t0 = max(avg_edge, 1e-9)
    temp = t0

    history: List[float] = []
    for _ in range(iterations):
        # random 2-opt: pick i<j and reverse the segment between them
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        a, b = current[i], current[(i + 1) % n]
        c, d = current[j], current[(j + 1) % n]
        delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current[i + 1:j + 1] = reversed(current[i + 1:j + 1])
            current_len += delta
            if current_len < best_len:
                best_len = current_len
                best = list(current)
        history.append(best_len)
        temp *= alpha
        if temp < 1e-12:
            temp = 1e-12

    return SAResult(
        tour=best, length=best_len, history=history,
        schedule={"t0": t0, "alpha": alpha, "iterations": iterations, "seed": seed},
    )
