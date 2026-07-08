"""DSATUR graph-colouring heuristic (Brelaz, 1979).

A *scalable* alternative to the exact backtracking colourer: backtracking is
exponential and only feasible for tiny signal clusters, whereas DSATUR colours
the whole city graph in near-quadratic time, usually with very few colours.

Greedy rule: repeatedly colour the uncoloured vertex with the highest
*saturation degree* (the number of distinct colours already on its neighbours),
breaking ties by ordinary degree, assigning it the smallest feasible colour.
DSATUR is exact on some graph classes (e.g. bipartite) and a strong heuristic in
general -- it never uses more colours than the naive greedy and often fewer.

Complexity: O(V^2) with this straightforward selection. Returns a proper colouring
(adjacent vertices differ) and the number of colours used.
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.graph.graph import Graph


def dsatur_colouring(graph: Graph) -> Tuple[Dict[int, int], int]:
    """Return (colour map, colours used) for an undirected graph."""
    if graph.directed:
        raise ValueError("graph colouring assumes an undirected graph")
    adj = {u: [v for v, _ in graph.neighbours(u)] for u in graph.nodes()}
    colour: Dict[int, int] = {}
    saturation: Dict[int, set] = {u: set() for u in adj}      # neighbour colours
    degree = {u: len(adj[u]) for u in adj}
    uncoloured = set(adj)

    while uncoloured:
        # pick the most saturated vertex, ties broken by degree
        u = max(uncoloured, key=lambda x: (len(saturation[x]), degree[x]))
        used = saturation[u]
        c = 0
        while c in used:                                      # smallest free colour
            c += 1
        colour[u] = c
        uncoloured.discard(u)
        for v in adj[u]:
            if v in uncoloured:
                saturation[v].add(c)

    colours_used = (max(colour.values()) + 1) if colour else 0
    return colour, colours_used
