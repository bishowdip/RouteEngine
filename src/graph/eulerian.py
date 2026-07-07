"""Eulerian path / circuit detection and construction (Hierholzer).

Route-engine use: a street-sweeper or postal round that must traverse *every
road exactly once* -- the Eulerian-trail question. An undirected connected graph
has an Eulerian circuit iff every vertex has even degree, and an Eulerian path
iff exactly zero or two vertices have odd degree.

Hierholzer's algorithm builds the trail in O(E) by following unused edges into
cycles and splicing them together.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.graph.graph import Graph


def euler_status(graph: Graph) -> str:
    """Return 'circuit', 'path', or 'none' for an undirected graph."""
    if graph.directed:
        raise ValueError("this Eulerian check assumes an undirected graph")
    odd = sum(1 for u in graph.nodes() if len(graph.neighbours(u)) % 2 == 1)
    if odd == 0:
        return "circuit"
    if odd == 2:
        return "path"
    return "none"


def eulerian_trail(graph: Graph) -> Optional[List[int]]:
    """Return a node sequence covering every edge once, or None if impossible.

    Assumes edges are connected on the non-isolated vertices.
    """
    status = euler_status(graph)
    if status == "none":
        return None

    # mutable multiset of incident edges per node, with unique edge ids
    adj: Dict[int, List[Tuple[int, int]]] = {u: [] for u in graph.nodes()}
    eid = 0
    for u, v, _ in graph.edges():
        adj[u].append((v, eid))
        adj[v].append((u, eid))
        eid += 1
    used = [False] * eid
    if eid == 0:
        return []

    # start at an odd-degree vertex for a path, else any vertex with edges
    start = next((u for u in graph.nodes() if len(adj[u]) % 2 == 1), None)
    if start is None:
        start = next(u for u in graph.nodes() if adj[u])

    stack = [start]
    trail: List[int] = []
    ptr = {u: 0 for u in graph.nodes()}
    while stack:
        u = stack[-1]
        # advance past already-used incident edges
        while ptr[u] < len(adj[u]) and used[adj[u][ptr[u]][1]]:
            ptr[u] += 1
        if ptr[u] == len(adj[u]):
            trail.append(stack.pop())
        else:
            v, e = adj[u][ptr[u]]
            used[e] = True
            stack.append(v)
    trail.reverse()
    # a valid trail uses every edge -> length E+1
    return trail if len(trail) == eid + 1 else None
