"""Fenwick tree (binary indexed tree) for prefix sums with point updates.

Route-engine use: maintain cumulative counts that change online -- e.g. live
vehicle/traffic counts per road segment, where we frequently bump one segment
and query the total over a range of segments.

A Fenwick tree supports both in O(log n) by letting index i cover a range whose
length is the lowest set bit of i. Far cheaper than recomputing prefix sums
(O(n)) after every update, and simpler than a segment tree when only sums are
needed.

    update(i, delta) : O(log n)
    prefix_sum(i)    : O(log n)   sum of [0, i]
    range_sum(l, r)  : O(log n)
"""

from __future__ import annotations

from typing import List


class FenwickTree:
    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("size must be non-negative")
        self._n = n
        self._tree = [0] * (n + 1)         # 1-indexed internally

    @classmethod
    def from_values(cls, values: List[int]) -> "FenwickTree":
        ft = cls(len(values))
        for i, v in enumerate(values):
            ft.update(i, v)
        return ft

    def update(self, i: int, delta: int) -> None:
        """Add ``delta`` at position ``i`` (0-indexed)."""
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range [0, {self._n})")
        i += 1
        while i <= self._n:
            self._tree[i] += delta
            i += i & (-i)                  # move to the next covering node

    def prefix_sum(self, i: int) -> int:
        """Sum of positions [0, i] inclusive (0-indexed)."""
        if i < 0:
            return 0
        i = min(i, self._n - 1) + 1
        total = 0
        while i > 0:
            total += self._tree[i]
            i -= i & (-i)                  # strip the lowest set bit
        return total

    def range_sum(self, left: int, right: int) -> int:
        """Sum of positions [left, right] inclusive."""
        if left > right:
            return 0
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
