"""Uninformed graph traversals: BFS, DFS, and connected components.

These underpin reachability and connectivity queries in the route engine (e.g.
"is every served intersection reachable?") and are kept iterative so they handle
the full city graph without recursion limits.

Complexity: O(V + E) time, O(V) space for each.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

from src.graph.graph import Graph


def bfs(graph: Graph, source: int) -> Dict[int, int]:
    """Breadth-first search; return node -> hop distance from source."""
    if not graph.has_node(source):
        raise KeyError(f"source {source!r} not in graph")
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v, _ in graph.neighbours(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def dfs(graph: Graph, source: int) -> List[int]:
    """Iterative depth-first search; return nodes in visitation order."""
    if not graph.has_node(source):
        raise KeyError(f"source {source!r} not in graph")
    seen, order, stack = {source}, [], [source]
    while stack:
        u = stack.pop()
        order.append(u)
        for v, _ in graph.neighbours(u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return order


def connected_components(graph: Graph) -> List[List[int]]:
    """Partition the (undirected) graph into connected components."""
    seen: set = set()
    components: List[List[int]] = []
    for start in graph.nodes():
        if start in seen:
            continue
        comp, q = [], deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v, _ in graph.neighbours(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        components.append(comp)
    return components


def is_connected(graph: Graph) -> bool:
    """True if the graph has exactly one connected component (or is empty)."""
    if graph.num_nodes == 0:
        return True
    return len(bfs(graph, graph.nodes()[0])) == graph.num_nodes
