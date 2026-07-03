"""Prim's minimum spanning tree.

Hand-implemented for ST5003CEM Task 2 on our own min-heap. Grows a single tree
from a start node, repeatedly adding the cheapest edge crossing the cut between
in-tree and out-of-tree nodes -- the priority of each frontier node is its
cheapest known connecting edge, relaxed via ``decrease_key``.

Complexity (binary heap):  time O(E log V), space O(V).
Operates on the undirected, connected component containing ``start``; for a
road network we run it on the largest connected component.

In the route engine an MST answers: "the cheapest set of roads that still keeps
every served intersection reachable" -- e.g. a minimal maintenance/backbone
network.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.graph.graph import Graph
from src.structures.min_heap import MinHeap


def prim_mst(graph: Graph, start: Optional[int] = None) -> Tuple[List[Tuple[int, int, float]], float]:
    """Return (mst_edges, total_weight) for the component containing ``start``.

    ``mst_edges`` is a list of (u, v, weight) with V'-1 entries for a component
    of V' nodes.
    """
    if graph.directed:
        raise ValueError("Prim's MST is defined on an undirected graph")
    if graph.num_nodes == 0:
        return [], 0.0
    if start is None:
        start = graph.nodes()[0]
    if not graph.has_node(start):
        raise KeyError(f"start {start!r} not in graph")

    INF = float("inf")
    best: Dict[int, float] = {u: INF for u in graph.nodes()}
    parent: Dict[int, Optional[int]] = {u: None for u in graph.nodes()}
    in_tree: set = set()
    best[start] = 0.0

    pq = MinHeap()
    pq.insert(0.0, start)
    mst_edges: List[Tuple[int, int, float]] = []
    total = 0.0

    while not pq.is_empty():
        w, u = pq.extract_min()
        if u in in_tree:
            continue
        in_tree.add(u)
        if parent[u] is not None:
            mst_edges.append((parent[u], u, w))
            total += w
        for v, weight in graph.neighbours(u):
            if v not in in_tree and weight < best[v]:
                best[v] = weight
                parent[v] = u
                pq.push_or_decrease(weight, v)

    return mst_edges, total
