"""Backtracking graph colouring -- ST5003CEM Task 3 (backtracking component).

Route-engine framing: assign one of k phases/colours to each road junction (or
region) so that no two adjacent junctions share a colour -- e.g. traffic-signal
grouping or zoning on the city graph. Decide whether a proper k-colouring exists
and return one.

Two variants share a recursive skeleton so we can MEASURE the effect of pruning:
  * ``colour_naive``  -- assigns colours then checks validity only at the end of
    a full assignment (generate-and-test): explores ~k^V leaves.
  * ``colour_pruned`` -- checks the constraint at each assignment and backtracks
    immediately on conflict (forward checking of the just-placed vertex), cutting
    whole subtrees.

Both count the nodes expanded so the report can plot the exponential blow-up and
the pruning saving. Graph colouring is NP-complete for k >= 3; worst case is
exponential, which the plot makes concrete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.graph.graph import Graph


@dataclass
class ColouringResult:
    colours: Optional[Dict[int, int]]   # node -> colour index, or None if none
    nodes_explored: int                 # search-tree nodes visited
    found: bool


def colour_pruned(graph: Graph, k: int) -> ColouringResult:
    """Backtracking with constraint checking at each step (prunes early)."""
    nodes = graph.nodes()
    adj = {u: [v for v, _ in graph.neighbours(u)] for u in nodes}
    colour: Dict[int, int] = {}
    counter = {"n": 0}

    def consistent(u: int, c: int) -> bool:
        return all(colour.get(v) != c for v in adj[u])

    def backtrack(idx: int) -> bool:
        if idx == len(nodes):
            return True
        u = nodes[idx]
        for c in range(k):
            counter["n"] += 1
            if consistent(u, c):       # prune: only descend on a valid colour
                colour[u] = c
                if backtrack(idx + 1):
                    return True
                del colour[u]
        return False

    found = backtrack(0)
    return ColouringResult(dict(colour) if found else None, counter["n"], found)


def colour_naive(graph: Graph, k: int, node_cap: int = 12) -> ColouringResult:
    """Generate-and-test: validity checked only at a complete assignment.

    Exponential and used purely as the pruning baseline, so it refuses graphs
    above ``node_cap`` vertices to avoid blowing up the benchmark.
    """
    nodes = graph.nodes()
    if len(nodes) > node_cap:
        raise ValueError(f"naive colouring capped at {node_cap} nodes (got {len(nodes)})")
    adj = {u: [v for v, _ in graph.neighbours(u)] for u in nodes}
    assignment: Dict[int, int] = {}
    counter = {"n": 0}
    best: Dict[str, Optional[Dict[int, int]]] = {"sol": None}

    def fully_valid() -> bool:
        return all(assignment[u] != assignment[v] for u in nodes for v in adj[u])

    def backtrack(idx: int) -> bool:
        if idx == len(nodes):
            if fully_valid():               # check only at the leaf
                best["sol"] = dict(assignment)
                return True
            return False
        u = nodes[idx]
        for c in range(k):
            counter["n"] += 1
            assignment[u] = c               # assign blindly, no pruning
            if backtrack(idx + 1):
                return True
        del assignment[u]
        return False

    found = backtrack(0)
    return ColouringResult(best["sol"], counter["n"], found)


def verify_colouring(graph: Graph, colours: Dict[int, int]) -> bool:
    """True iff ``colours`` is a proper colouring of every edge."""
    for u, v, _ in graph.edges():
        if colours.get(u) == colours.get(v):
            return False
    return True
