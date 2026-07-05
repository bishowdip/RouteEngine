"""Floyd-Warshall all-pairs shortest paths.

Dynamic programming over an intermediate-vertex index k: the shortest path from
i to j using only vertices {0..k} is either the best path using {0..k-1}, or the
best path i->k plus k->j. Iterating k from 0..V-1 fills the full distance matrix.

Recurrence:  D_k[i][j] = min( D_{k-1}[i][j], D_{k-1}[i][k] + D_{k-1}[k][j] )

Complexity: O(V^3) time, O(V^2) space. Only practical for small/medium graphs,
but it answers *every* origin-destination pair at once -- useful for building the
distance matrix a TSP tour needs, and it tolerates negative edges (and flags a
negative cycle if any diagonal entry goes negative).
"""

from __future__ import annotations

from typing import List, Tuple

from src.graph.graph import Graph


def floyd_warshall(graph: Graph) -> Tuple[List[List[float]], List[int], bool]:
    """Return (dist_matrix, node_order, has_negative_cycle)."""
    order = graph.nodes()
    idx = {u: i for i, u in enumerate(order)}
    n = len(order)
    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    for u, v, w in graph.edges():
        a, b = idx[u], idx[v]
        if w < dist[a][b]:
            dist[a][b] = w
        if not graph.directed and w < dist[b][a]:
            dist[b][a] = w

    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                via = dik + dk[j]
                if via < di[j]:
                    di[j] = via

    has_negative_cycle = any(dist[i][i] < 0 for i in range(n))
    return dist, order, has_negative_cycle
