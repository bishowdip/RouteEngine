"""Maximum flow / minimum cut via the Edmonds-Karp algorithm.

Route-engine use: treat edge weights as road *capacities* (vehicles/hour) and ask
how much traffic can move from a source district to a sink district, and which
roads form the bottleneck (the minimum cut). By the max-flow min-cut theorem the
two values coincide, so the saturated cut edges are exactly the capacity
limiters to upgrade.

Edmonds-Karp is Ford-Fulkerson with BFS augmenting paths (shortest residual path
each round), giving a polynomial bound of O(V * E^2) independent of capacities.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

from src.graph.graph import Graph


def _residual(graph: Graph) -> Dict[int, Dict[int, float]]:
    """Build residual capacities; weight is treated as capacity."""
    cap: Dict[int, Dict[int, float]] = {u: {} for u in graph.nodes()}
    for u, v, w in graph.edges():
        cap[u][v] = cap[u].get(v, 0.0) + w
        cap[v].setdefault(u, 0.0)                 # reverse residual arc
        if not graph.directed:
            cap[v][u] = cap[v].get(u, 0.0) + w
    return cap


def max_flow(graph: Graph, source: int, sink: int) -> Tuple[float, Dict[int, Dict[int, float]]]:
    """Return (max_flow_value, residual_capacities)."""
    if not graph.has_node(source) or not graph.has_node(sink):
        raise KeyError("source or sink not in graph")
    if source == sink:
        raise ValueError("source and sink must differ")

    cap = _residual(graph)
    flow = 0.0
    while True:
        # BFS for a shortest augmenting path in the residual graph
        parent: Dict[int, int] = {source: source}
        q = deque([source])
        while q and sink not in parent:
            u = q.popleft()
            for v, c in cap[u].items():
                if v not in parent and c > 1e-12:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            break                                  # no augmenting path -> done

        # bottleneck residual capacity along the path
        bottleneck = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            bottleneck = min(bottleneck, cap[u][v])
            v = u
        # augment
        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= bottleneck
            cap[v][u] += bottleneck
            v = u
        flow += bottleneck
    return flow, cap


def min_cut(graph: Graph, source: int, sink: int) -> Tuple[float, List[Tuple[int, int]]]:
    """Return (cut_value, cut_edges): the saturated source-side -> sink-side edges."""
    value, cap = max_flow(graph, source, sink)
    # nodes reachable from source in the residual graph form the source side
    reachable: Set[int] = set()
    q = deque([source])
    reachable.add(source)
    while q:
        u = q.popleft()
        for v, c in cap[u].items():
            if c > 1e-12 and v not in reachable:
                reachable.add(v)
                q.append(v)
    cut_edges = [(u, v) for u, v, _ in graph.edges()
                 if (u in reachable) != (v in reachable)]
    return value, cut_edges
