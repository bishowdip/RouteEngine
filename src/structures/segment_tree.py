"""Segment tree for range queries with point updates.

Route-engine use: answer range aggregate queries over an ordered array of
segment metrics -- e.g. the minimum travel time, or the sum of vehicle counts,
across a contiguous stretch of road segments -- while individual segments update
live. More flexible than a Fenwick tree (any associative combine, not just
sums), at roughly double the space.

    build      : O(n)
    update(i)  : O(log n)
    query(l,r) : O(log n)

The combine function and identity are injected, so the same tree serves sum, min
or max queries.
"""

from __future__ import annotations

from typing import Callable, List


class SegmentTree:
    def __init__(self, values: List[float],
                 combine: Callable[[float, float], float] = lambda a, b: a + b,
                 identity: float = 0.0) -> None:
        self._combine = combine
        self._identity = identity
        self._n = len(values)
        if self._n == 0:
            self._tree: List[float] = []
            return
        self._tree = [identity] * (2 * self._n)
        # leaves live in [n, 2n); internal nodes are built bottom-up
        for i, v in enumerate(values):
            self._tree[self._n + i] = v
        for i in range(self._n - 1, 0, -1):
            self._tree[i] = combine(self._tree[2 * i], self._tree[2 * i + 1])

    def update(self, i: int, value: float) -> None:
        """Set position ``i`` to ``value`` and refresh its ancestors."""
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range [0, {self._n})")
        i += self._n
        self._tree[i] = value
        i //= 2
        while i >= 1:
            self._tree[i] = self._combine(self._tree[2 * i], self._tree[2 * i + 1])
            i //= 2

    def query(self, left: int, right: int) -> float:
        """Aggregate over [left, right] inclusive."""
        if left > right or left < 0 or right >= self._n:
            raise IndexError("invalid query range")
        res = self._identity
        l, r = left + self._n, right + self._n + 1
        while l < r:
            if l & 1:
                res = self._combine(res, self._tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = self._combine(res, self._tree[r])
            l //= 2
            r //= 2
        return res
