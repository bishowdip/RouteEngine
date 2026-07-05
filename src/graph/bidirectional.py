"""Bidirectional Dijkstra for point-to-point shortest paths.

Runs two simultaneous Dijkstra searches -- one forward from the source, one
backward from the target -- and stops when their explored frontiers meet. By
growing two small search balls of radius ~d/2 instead of one of radius d, it
typically settles far fewer nodes than a single Dijkstra, while remaining exact.

Complexity: same O((V+E) log V) worst case, but roughly halves the explored area
on uniform graphs. Requires non-negative weights (built on the custom min-heap).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.graph.graph import Graph
from src.structures.min_heap import MinHeap


def _build_reverse(graph: Graph) -> Dict[int, List[Tuple[int, float]]]:
    """Incoming-edge adjacency for the backward search."""
    rev: Dict[int, List[Tuple[int, float]]] = {u: [] for u in graph.nodes()}
    for u, v, w in graph.edges():
        rev[v].append((u, w))
        if not graph.directed:
            rev[u].append((v, w))
    return rev


def bidirectional_dijkstra(graph: Graph, source: int, target: int
                           ) -> Tuple[float, List[int]]:
    """Return (cost, path) from source to target ([] if unreachable)."""
    if not graph.has_node(source) or not graph.has_node(target):
        raise KeyError("source or target not in graph")
    if source == target:
        return 0.0, [source]

    rev = _build_reverse(graph)
    df: Dict[int, float] = {source: 0.0}
    db: Dict[int, float] = {target: 0.0}
    pf: Dict[int, Optional[int]] = {source: None}
    pb: Dict[int, Optional[int]] = {target: None}
    settled_f: set = set()
    settled_b: set = set()
    fq, bq = MinHeap(), MinHeap()
    fq.insert(0.0, source)
    bq.insert(0.0, target)

    best = float("inf")
    meet = -1

    while not fq.is_empty() and not bq.is_empty():
        # stop once the two minimum frontiers can no longer improve 'best'
        if fq.peek()[0] + bq.peek()[0] >= best:
            break

        # ---- forward step ----
        d, u = fq.extract_min()
        if u not in settled_f:
            settled_f.add(u)
            for v, w in graph.neighbours(u):
                if w < 0:
                    raise ValueError("bidirectional Dijkstra needs non-negative weights")
                nd = d + w
                if nd < df.get(v, float("inf")):
                    df[v] = nd
                    pf[v] = u
                    fq.push_or_decrease(nd, v)
                    if v in db and nd + db[v] < best:
                        best, meet = nd + db[v], v

        # ---- backward step ----
        d, u = bq.extract_min()
        if u not in settled_b:
            settled_b.add(u)
            for v, w in rev[u]:
                nd = d + w
                if nd < db.get(v, float("inf")):
                    db[v] = nd
                    pb[v] = u
                    bq.push_or_decrease(nd, v)
                    if v in df and nd + df[v] < best:
                        best, meet = nd + df[v], v

    if meet == -1:
        return float("inf"), []

    # stitch: source..meet (forward) + meet..target (backward)
    forward: List[int] = []
    x: Optional[int] = meet
    while x is not None:
        forward.append(x)
        x = pf[x]
    forward.reverse()
    backward: List[int] = []
    x = pb[meet]
    while x is not None:
        backward.append(x)
        x = pb[x]
    return best, forward + backward
