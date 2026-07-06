"""Topological sort of a directed acyclic graph (Kahn's algorithm).

Route-engine use: order tasks/segments under precedence constraints -- e.g.
sequencing one-way road works or delivery dependencies where some stops must
precede others. A valid linear order exists iff the directed graph is acyclic;
the algorithm doubles as a cycle detector.

Kahn's method: repeatedly emit a node with in-degree 0 and decrement its
successors. Complexity: O(V + E).
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from src.graph.graph import Graph


def topological_sort(graph: Graph) -> Tuple[Optional[List[int]], bool]:
    """Return (order, is_dag). order is None when a cycle exists."""
    if not graph.directed:
        raise ValueError("topological sort requires a directed graph")
    indeg = {u: 0 for u in graph.nodes()}
    for _, v, _ in graph.edges():
        indeg[v] += 1
    queue = deque(u for u in graph.nodes() if indeg[u] == 0)
    order: List[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in graph.neighbours(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != graph.num_nodes:
        return None, False          # leftover in-degree => a cycle remains
    return order, True
