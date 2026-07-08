"""Or-opt local search for TSP.

Or-opt relocates a short *chain* of 1-3 consecutive cities to a better position
elsewhere in the tour, complementing 2-opt's edge reversals. Some improvements
reachable by moving a segment are missed by 2-opt alone, so running both yields
better local optima. One neighbourhood scan is O(n^2) per segment length.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.heuristics.nearest_neighbour import nearest_neighbour
from src.heuristics.tsp import TSPInstance


def or_opt(inst: TSPInstance, initial: Optional[List[int]] = None,
           max_passes: int = 1000) -> Tuple[List[int], float]:
    dist = inst.dist
    tour = list(initial) if initial is not None else nearest_neighbour(inst)[0]
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for seg_len in (1, 2, 3):
            for i in range(n):
                if i + seg_len > n:
                    continue
                seg = tour[i:i + seg_len]
                rest = tour[:i] + tour[i + seg_len:]
                p_prev = tour[i - 1]
                p_next = tour[(i + seg_len) % n]
                removed = (dist[p_prev][seg[0]] + dist[seg[-1]][p_next]
                           - dist[p_prev][p_next])
                # try reinserting the segment between every adjacent pair of rest
                for j in range(len(rest)):
                    a = rest[j]
                    b = rest[(j + 1) % len(rest)]
                    added = (dist[a][seg[0]] + dist[seg[-1]][b] - dist[a][b])
                    if added - removed < -1e-9:
                        tour = rest[:j + 1] + seg + rest[j + 1:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return tour, inst.tour_length(tour)
