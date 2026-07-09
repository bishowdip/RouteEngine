"""Johnson's all-pairs shortest paths.

Combines the two Task-2 algorithms: Bellman-Ford reweights the graph so all edges
are non-negative, then a Dijkstra is run from every vertex. This beats
Floyd-Warshall (O(V^3)) on *sparse* graphs -- exactly the city case -- at
O(V*E + V^2 log V), and still tolerates negative edges (flagging negative cycles).

Reweighting: add a virtual vertex with a 0-cost edge to every node, run
Bellman-Ford to get potentials h(v), then w'(u,v) = w(u,v) + h(u) - h(v) >= 0 by
the triangle inequality. Dijkstra distances are then corrected back by h.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.graph.bellman_ford import bellman_ford
from src.graph.dijkstra import dijkstra
from src.graph.graph import Graph


def johnson(graph: Graph) -> Optional[Dict[int, Dict[int, float]]]:
    """Return {source: {target: distance}} or None if a negative cycle exists."""
    nodes = graph.nodes()
    virtual = object()  # sentinel id not colliding with int node ids

    # augmented graph: virtual -> every node with weight 0
    aug = Graph(directed=True)
    for u, v, w in graph.edges():
        aug.add_edge(u, v, w)
        if not graph.directed:
            aug.add_edge(v, u, w)
    for u in nodes:
        aug.add_edge(virtual, u, 0.0)

    h_dist, _, has_neg = bellman_ford(aug, virtual)
    if has_neg:
        return None
    h = {u: h_dist[u] for u in nodes}

    # reweighted non-negative graph
    rew = Graph(directed=True)
    for u in nodes:
        rew.add_node(u)
    for u, v, w in graph.edges():
        rew.add_edge(u, v, w + h[u] - h[v])
        if not graph.directed:
            rew.add_edge(v, u, w + h[v] - h[u])

    all_pairs: Dict[int, Dict[int, float]] = {}
    for src in nodes:
        d, _ = dijkstra(rew, src)
        # undo the reweighting: real = reweighted - h[src] + h[tgt]
        all_pairs[src] = {
            t: (d[t] - h[src] + h[t]) if d[t] != float("inf") else float("inf")
            for t in nodes
        }
    return all_pairs
