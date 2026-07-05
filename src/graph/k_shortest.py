"""Yen's algorithm for the K shortest loopless paths.

Route-engine use: offer a driver not just the optimal route but the next-best
*alternatives* (e.g. to avoid a closure). Yen's algorithm finds the 1st shortest
path with Dijkstra, then derives each successive path by, at every node of the
previous path, forcing a detour: temporarily banning the edges that would
reproduce an already-found path and the spur node's predecessors, and splicing
the best spur onto the root.

Complexity: O(K * V * (E + V log V)) -- K Dijkstra-scale searches per candidate.
Built on the custom min-heap; supports banned nodes/edges without mutating the
shared graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.graph.graph import Graph
from src.structures.min_heap import MinHeap

Path = Tuple[float, List[int]]


def _dijkstra_constrained(
    graph: Graph, source: int, target: int,
    banned_nodes: Set[int], banned_edges: Set[Tuple[int, int]],
) -> Optional[Path]:
    """Shortest source->target path avoiding banned nodes/edges, or None."""
    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    settled: Set[int] = set()
    pq = MinHeap()
    pq.insert(0.0, source)
    while not pq.is_empty():
        d, u = pq.extract_min()
        if u in settled:
            continue
        settled.add(u)
        if u == target:
            break
        for v, w in graph.neighbours(u):
            if v in banned_nodes or (u, v) in banned_edges:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                pq.push_or_decrease(nd, v)
    if target not in dist:
        return None
    path, cur = [], target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[target], path


def k_shortest_paths(graph: Graph, source: int, target: int, k: int) -> List[Path]:
    """Return up to k shortest loopless paths, ascending by cost."""
    if not graph.has_node(source) or not graph.has_node(target):
        raise KeyError("source or target not in graph")
    first = _dijkstra_constrained(graph, source, target, set(), set())
    if first is None:
        return []
    accepted: List[Path] = [first]
    candidates: List[Path] = []

    for _ in range(1, k):
        prev_cost, prev_path = accepted[-1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]
            banned_edges: Set[Tuple[int, int]] = set()
            for _, p in accepted:
                if len(p) > i and p[: i + 1] == root_path:
                    banned_edges.add((p[i], p[i + 1]))   # don't retread a found path
            banned_nodes = set(root_path[:-1])           # keep root loopless
            spur = _dijkstra_constrained(graph, spur_node, target,
                                         banned_nodes, banned_edges)
            if spur is None:
                continue
            _, spur_path = spur
            total_path = root_path[:-1] + spur_path
            cost = _path_cost(graph, total_path)
            entry = (cost, total_path)
            if entry not in candidates and entry not in accepted:
                candidates.append(entry)
        if not candidates:
            break
        candidates.sort(key=lambda e: e[0])
        accepted.append(candidates.pop(0))
    return accepted


def _path_cost(graph: Graph, path: List[int]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        leg = next((w for v, w in graph.neighbours(a) if v == b), None)
        if leg is None:
            return float("inf")
        total += leg
    return total
