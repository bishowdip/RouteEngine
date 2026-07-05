"""A* shortest path -- the heuristic-guided extension of Dijkstra.

A* orders the frontier by f(n) = g(n) + h(n), where g is the cost so far and h
is an *admissible* estimate of the remaining cost to the target (one that never
overestimates). With an admissible heuristic A* is optimal, and it typically
expands far fewer nodes than Dijkstra for point-to-point queries.

For the road network the natural heuristic is the great-circle (haversine)
straight-line distance between two intersections' coordinates: it can never
exceed the true road distance, so it is admissible. Without coordinates the
heuristic is zero and A* gracefully degrades to Dijkstra.

Complexity: same worst case as Dijkstra, O((V+E) log V), but far fewer
expansions in practice; built on the hand-written min-heap.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from src.graph.dijkstra import reconstruct_path
from src.graph.graph import Graph
from src.structures.min_heap import MinHeap

EARTH_R = 6_371_000.0  # metres


def haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def astar(
    graph: Graph, source: int, target: int,
    heuristic: Optional[Callable[[int, int], float]] = None,
) -> Tuple[float, List[int], int]:
    """Return (cost, path, expanded) from source to target.

    ``expanded`` is the number of nodes finalised -- reported so A* can be
    compared against Dijkstra on search effort.
    """
    if not graph.has_node(source) or not graph.has_node(target):
        raise KeyError("source or target not in graph")

    if heuristic is None:
        coords = graph.coords

        def heuristic(u: int, v: int) -> float:
            cu, cv = coords.get(u), coords.get(v)
            if cu is None or cv is None or cu[0] is None or cv[0] is None:
                return 0.0  # no coords -> behaves like Dijkstra
            return haversine(cu, cv)

    g_score: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    settled: set = set()
    expanded = 0

    pq = MinHeap()
    pq.insert(heuristic(source, target), source)
    while not pq.is_empty():
        _, u = pq.extract_min()
        if u in settled:
            continue
        settled.add(u)
        expanded += 1
        if u == target:
            return g_score[u], reconstruct_path(prev, source, target), expanded
        for v, w in graph.neighbours(u):
            if w < 0:
                raise ValueError("A* requires non-negative weights")
            tentative = g_score[u] + w
            if tentative < g_score.get(v, float("inf")):
                g_score[v] = tentative
                prev[v] = u
                pq.push_or_decrease(tentative + heuristic(v, target), v)
    return float("inf"), [], expanded
