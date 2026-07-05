"""Dijkstra's single-source shortest paths.

Hand-implemented for ST5003CEM Task 2 using our own binary min-heap
(``src.structures.min_heap``) as the priority queue -- the brief forbids
``heapq`` here.

Complexity with a binary heap and ``decrease_key``:
    time  : O((V + E) log V)
    space : O(V)
Correct ONLY for non-negative edge weights: the greedy "finalise the closest
unsettled node" argument relies on no later, cheaper path appearing -- which a
negative edge could create. That failure is demonstrated against Bellman-Ford
in the Task 2 report.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.graph.graph import Graph
from src.structures.min_heap import MinHeap


def dijkstra(
    graph: Graph, source: int, target: Optional[int] = None
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """Return (dist, prev) maps. If ``target`` is given, stop once settled.

    ``dist[v]``  = shortest-path cost source->v (inf if unreachable)
    ``prev[v]``  = predecessor of v on a shortest path (for reconstruction)
    """
    if not graph.has_node(source):
        raise KeyError(f"source {source!r} not in graph")

    dist: Dict[int, float] = {u: float("inf") for u in graph.nodes()}
    prev: Dict[int, Optional[int]] = {u: None for u in graph.nodes()}
    dist[source] = 0.0
    settled: set = set()

    pq = MinHeap()
    pq.insert(0.0, source)
    while not pq.is_empty():
        d, u = pq.extract_min()
        if u in settled:
            continue            # stale entry; the live one was processed already
        settled.add(u)
        if target is not None and u == target:
            break               # early exit: u is finalised
        for v, w in graph.neighbours(u):
            if w < 0:
                raise ValueError(
                    "Dijkstra requires non-negative weights; use Bellman-Ford"
                )
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                pq.push_or_decrease(nd, v)
    return dist, prev


def reconstruct_path(prev: Dict[int, Optional[int]], source: int, target: int) -> List[int]:
    """Walk predecessors from target back to source. [] if unreachable."""
    path: List[int] = []
    cur: Optional[int] = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev[cur]
    path.reverse()
    return path if path and path[0] == source else []
