"""Disjoint-set (union-find) structure.

Supports near-constant-time union and find via two optimisations:
  * union by rank  -- attach the shorter tree under the taller;
  * path compression -- flatten the find path on the way to the root.
Together these give an amortised cost of O(alpha(n)) per operation, where
alpha is the inverse Ackermann function (<= 4 for any practical n).

In the route engine this powers Kruskal's MST (tracking which intersections are
already connected) and connected-component queries.
"""

from __future__ import annotations

from typing import Dict, Hashable, List


class UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[Hashable, Hashable] = {}
        self._rank: Dict[Hashable, int] = {}
        self._sets = 0

    def add(self, x: Hashable) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._sets += 1

    def find(self, x: Hashable) -> Hashable:
        """Return the representative of x's set, compressing the path."""
        if x not in self._parent:
            raise KeyError(f"{x!r} not in any set")
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression: point every node on the path straight at the root
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> bool:
        """Merge the sets of a and b. Return False if already joined."""
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._sets -= 1
        return True

    def connected(self, a: Hashable, b: Hashable) -> bool:
        return self.find(a) == self.find(b)

    @property
    def num_sets(self) -> int:
        return self._sets

    def groups(self) -> List[List[Hashable]]:
        out: Dict[Hashable, List[Hashable]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return list(out.values())
