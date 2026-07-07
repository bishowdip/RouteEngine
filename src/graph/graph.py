"""Weighted graph supporting BOTH adjacency-list and adjacency-matrix backings.

Hand-implemented for ST5003CEM Task 2. Holding both representations lets us
benchmark the classic space/lookup trade-off on the same data:

                 space        edge lookup   neighbours iter   best for
  adjacency list O(V + E)     O(deg(u))     O(deg(u))         sparse graphs
  adjacency mat  O(V^2)       O(1)          O(V)              dense graphs

A real road network is *sparse* (each intersection meets a handful of roads),
so the list backing is what the route engine uses in practice; the matrix is
kept for the dense-graph comparison in the report.

Nodes carry an optional (lat, lon) so Task 2 can render routes on a real map
and Task 4's TSP can compute geographic distances.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

Edge = Tuple[int, float]  # (neighbour, weight)


class Graph:
    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self._adj: Dict[int, List[Edge]] = {}
        self.coords: Dict[int, Tuple[float, float]] = {}  # node -> (lat, lon)

    # --------------------------------------------------------------- mutation
    def add_node(self, u: int, coord: Optional[Tuple[float, float]] = None) -> None:
        self._adj.setdefault(u, [])
        if coord is not None:
            self.coords[u] = coord

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        if weight != weight:  # NaN guard
            raise ValueError("edge weight must not be NaN")
        self.add_node(u)
        self.add_node(v)
        self._adj[u].append((v, weight))
        if not self.directed:
            self._adj[v].append((u, weight))

    def remove_edge(self, u: int, v: int) -> bool:
        """Remove edge(s) u->v (and v->u if undirected). Return True if removed."""
        if u not in self._adj:
            return False
        before = len(self._adj[u])
        self._adj[u] = [(w, wt) for w, wt in self._adj[u] if w != v]
        removed = len(self._adj[u]) != before
        if removed and not self.directed and v in self._adj:
            self._adj[v] = [(w, wt) for w, wt in self._adj[v] if w != u]
        return removed

    def remove_node(self, u: int) -> bool:
        """Remove a node and all incident edges. Return True if it existed."""
        if u not in self._adj:
            return False
        del self._adj[u]
        self.coords.pop(u, None)
        for w, nbrs in self._adj.items():
            self._adj[w] = [(x, wt) for x, wt in nbrs if x != u]
        return True

    # ---------------------------------------------------------------- queries
    @property
    def num_nodes(self) -> int:
        return len(self._adj)

    @property
    def num_edges(self) -> int:
        total = sum(len(nbrs) for nbrs in self._adj.values())
        return total if self.directed else total // 2

    def nodes(self) -> List[int]:
        return list(self._adj.keys())

    def neighbours(self, u: int) -> List[Edge]:
        if u not in self._adj:
            raise KeyError(f"node {u!r} not in graph")
        return self._adj[u]

    def has_node(self, u: int) -> bool:
        return u in self._adj

    def has_edge(self, u: int, v: int) -> bool:
        """True if an edge u->v exists. O(deg(u))."""
        if u not in self._adj:
            return False
        return any(w == v for w, _ in self._adj[u])

    def get_edge_weight(self, u: int, v: int) -> Optional[float]:
        """Weight of the lightest u->v edge, or None if absent. O(deg(u))."""
        if u not in self._adj:
            raise KeyError(f"node {u!r} not in graph")
        best: Optional[float] = None
        for w, weight in self._adj[u]:
            if w == v and (best is None or weight < best):
                best = weight
        return best

    def degree(self, u: int) -> int:
        """Number of incident edges (out-arcs for a directed graph)."""
        if u not in self._adj:
            raise KeyError(f"node {u!r} not in graph")
        return len(self._adj[u])

    def edges(self) -> Iterator[Tuple[int, int, float]]:
        """Yield each edge once (undirected) or each arc (directed)."""
        seen = set()
        for u, nbrs in self._adj.items():
            for v, w in nbrs:
                if not self.directed:
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        continue
                    seen.add(key)
                yield u, v, w

    def density(self) -> float:
        """E / E_max in [0, 1]; ~0 for a road network, ~1 for a clique."""
        v = self.num_nodes
        if v < 2:
            return 0.0
        max_e = v * (v - 1)
        if not self.directed:
            max_e //= 2
        return self.num_edges / max_e

    # ----------------------------------------------------- matrix conversion
    def to_adjacency_matrix(self) -> Tuple[List[List[float]], List[int]]:
        """Materialise the O(V^2) adjacency matrix (INF = no edge).

        Returns (matrix, index_order) where index_order maps matrix rows back to
        node ids. Used for the list-vs-matrix benchmark; not for large sparse
        graphs where the V^2 blow-up is the very point being demonstrated.
        """
        order = self.nodes()
        idx = {u: i for i, u in enumerate(order)}
        n = len(order)
        INF = float("inf")
        mat = [[INF] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 0.0
        for u, nbrs in self._adj.items():
            for v, w in nbrs:
                # keep the lightest parallel edge if any
                if w < mat[idx[u]][idx[v]]:
                    mat[idx[u]][idx[v]] = w
        return mat, order

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        return f"<Graph {kind} V={self.num_nodes} E={self.num_edges} density={self.density():.4f}>"
