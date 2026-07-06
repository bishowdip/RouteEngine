"""Strongly connected components (Tarjan's algorithm).

Route-engine use: in a directed network of one-way streets, an SCC is a set of
intersections all mutually reachable. Identifying them reveals zones where any
point can reach any other, and condensing them yields a DAG of zones.

Tarjan's single-DFS method tracks each node's discovery index and the lowest
index reachable from its subtree; a node whose low-link equals its own index
roots an SCC popped from a stack. Complexity: O(V + E). Implemented iteratively
to handle the full city graph without recursion limits.
"""

from __future__ import annotations

from typing import Dict, List

from src.graph.graph import Graph


def strongly_connected_components(graph: Graph) -> List[List[int]]:
    """Return the SCCs of a directed graph as lists of node ids."""
    if not graph.directed:
        raise ValueError("SCCs are defined on a directed graph")

    index_of: Dict[int, int] = {}
    low: Dict[int, int] = {}
    on_stack: set = set()
    stack: List[int] = []
    sccs: List[List[int]] = []
    counter = 0

    for root in graph.nodes():
        if root in index_of:
            continue
        # iterative DFS: frame = (node, neighbour-iterator)
        work = [(root, iter(graph.neighbours(root)))]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            u, it = work[-1]
            advanced = False
            for v, _ in it:
                if v not in index_of:
                    index_of[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack.add(v)
                    work.append((v, iter(graph.neighbours(v))))
                    advanced = True
                    break
                elif v in on_stack:
                    low[u] = min(low[u], index_of[v])
            if advanced:
                continue
            # done with u: if it roots an SCC, pop the component
            if low[u] == index_of[u]:
                comp: List[int] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == u:
                        break
                sccs.append(comp)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[u])
    return sccs
