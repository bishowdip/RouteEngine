"""Kruskal's minimum spanning tree.

A second MST algorithm alongside Prim's, included so the two can be compared:
Kruskal sorts all edges and greedily adds the cheapest edge that does not form a
cycle (cycle detection by union-find). It produces the same total weight as Prim
on a connected graph, but is edge-centric rather than vertex-centric.

Complexity: O(E log E) dominated by the edge sort (union-find ops are
near-constant). Kruskal tends to suit sparse graphs and naturally yields a
minimum spanning *forest* on disconnected graphs.
"""

from __future__ import annotations

from typing import List, Tuple

from src.graph.graph import Graph
from src.structures.union_find import UnionFind


def kruskal_mst(graph: Graph) -> Tuple[List[Tuple[int, int, float]], float]:
    """Return (mst_edges, total_weight). Spanning forest if disconnected."""
    if graph.directed:
        raise ValueError("Kruskal's MST is defined on an undirected graph")
    uf = UnionFind()
    for u in graph.nodes():
        uf.add(u)
    edges = sorted(graph.edges(), key=lambda e: e[2])  # by weight ascending
    mst: List[Tuple[int, int, float]] = []
    total = 0.0
    for u, v, w in edges:
        if uf.union(u, v):          # only add if it joins two components
            mst.append((u, v, w))
            total += w
    return mst, total
