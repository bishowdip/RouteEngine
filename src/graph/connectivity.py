"""Articulation points and bridges -- network resilience analysis.

Route-engine use: find the *critical* infrastructure whose failure fragments the
city. An articulation point is an intersection whose removal disconnects the
graph; a bridge is a road whose removal does the same. These are exactly the
single points of failure a logistics network should harden or avoid depending on.

Both are found in a single DFS (Tarjan) using discovery times and low-link
values: a child subtree that cannot reach above its parent reveals a cut.
Complexity: O(V + E). Implemented iteratively for the full city graph.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from src.graph.graph import Graph


def find_bridges_and_articulation(graph: Graph
                                  ) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """Return (bridges, articulation_points) of an undirected graph."""
    if graph.directed:
        raise ValueError("articulation/bridge analysis assumes an undirected graph")

    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    bridges: List[Tuple[int, int]] = []
    articulation: Set[int] = set()
    timer = 0

    for root in graph.nodes():
        if root in disc:
            continue
        # iterative DFS; frame carries node, parent, and a neighbour iterator
        stack = [(root, -1, iter(graph.neighbours(root)))]
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        while stack:
            u, parent, it = stack[-1]
            advanced = False
            for v, _ in it:
                if v == parent:
                    continue
                if v not in disc:
                    if u == root:
                        root_children += 1
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, u, iter(graph.neighbours(v))))
                    advanced = True
                    break
                else:
                    low[u] = min(low[u], disc[v])     # back edge
            if advanced:
                continue
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[u])
                # non-root articulation: a child cannot escape above p
                if stack[-1][1] != -1 and low[u] >= disc[p]:
                    articulation.add(p)
                if low[u] > disc[p]:                  # bridge: child strictly below
                    bridges.append((min(p, u), max(p, u)))
        if root_children > 1:                         # root is a cut if >1 DFS child
            articulation.add(root)
    return bridges, articulation
