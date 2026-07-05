"""Bellman-Ford single-source shortest paths, with negative-cycle detection.

Hand-implemented for ST5003CEM Task 2. Unlike Dijkstra it tolerates *negative
edge weights* (e.g. a road segment modelling a refund/credit, or a downhill
energy gain for an EV), and it detects negative cycles -- where "shortest path"
becomes ill-defined.

Algorithm: relax every edge V-1 times (any shortest path has <= V-1 edges); then
one extra pass -- if any edge still relaxes, a negative cycle is reachable.

Complexity:  time O(V * E), space O(V).
That O(V*E) is markedly worse than Dijkstra's O((V+E) log V); the report shows
Bellman-Ford succeeding where Dijkstra fails, *and* the runtime price paid -- so
it is kept to modest graph sizes deliberately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class NegativeCycleError(ValueError):
    """Raised (when requested) if a negative-weight cycle is reachable."""

    def __init__(self, cycle: Optional[List[int]] = None) -> None:
        self.cycle = cycle
        super().__init__("graph contains a negative-weight cycle reachable from source")


def bellman_ford(
    graph, source: int, *, raise_on_negative_cycle: bool = False
) -> Tuple[Dict[int, float], Dict[int, Optional[int]], bool]:
    """Return (dist, prev, has_negative_cycle).

    If ``raise_on_negative_cycle`` is set, raise ``NegativeCycleError`` instead
    of returning the flag (with the offending cycle attached).
    """
    if not graph.has_node(source):
        raise KeyError(f"source {source!r} not in graph")

    nodes = graph.nodes()
    dist: Dict[int, float] = {u: float("inf") for u in nodes}
    prev: Dict[int, Optional[int]] = {u: None for u in nodes}
    dist[source] = 0.0

    # Materialise the arc list once (each undirected edge relaxes both ways).
    arcs: List[Tuple[int, int, float]] = []
    for u in nodes:
        for v, w in graph.neighbours(u):
            arcs.append((u, v, w))

    # V-1 relaxation passes, with early termination if a pass changes nothing.
    for _ in range(len(nodes) - 1):
        changed = False
        for u, v, w in arcs:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                changed = True
        if not changed:
            break

    # One more pass: any further relaxation => reachable negative cycle.
    has_negative_cycle = False
    culprit: Optional[int] = None
    for u, v, w in arcs:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            has_negative_cycle = True
            culprit = v
            break

    if has_negative_cycle and raise_on_negative_cycle:
        raise NegativeCycleError(_extract_cycle(prev, culprit, len(nodes)))

    return dist, prev, has_negative_cycle


def _extract_cycle(prev: Dict[int, Optional[int]], start: Optional[int], v_count: int) -> Optional[List[int]]:
    """Recover a node on the negative cycle, then walk it out."""
    if start is None:
        return None
    x = start
    for _ in range(v_count):  # step back V times to land *inside* the cycle
        x = prev.get(x)
        if x is None:
            return None
    cycle = [x]
    cur = prev.get(x)
    while cur is not None and cur != x:
        cycle.append(cur)
        cur = prev.get(cur)
    cycle.append(x)
    cycle.reverse()
    return cycle
