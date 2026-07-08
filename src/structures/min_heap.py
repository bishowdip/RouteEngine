"""Array-backed binary min-heap / priority queue.

Hand-implemented for ST5003CEM Task 1, and reused as the priority queue that
drives Dijkstra and Prim in Task 2 (the brief forbids using ``heapq`` there).

Complexity:
    peek        : O(1)
    insert      : O(log n)   sift-up
    extract_min : O(log n)   sift-down
    build_heap  : O(n)       Floyd's bottom-up heapify (NOT n log n)
    decrease_key: O(log n)   via an index map, needed for Dijkstra/Prim

Items are (priority, payload) pairs. A position index maps each payload to its
slot in the array so ``decrease_key`` can find it in O(1) before sifting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class MinHeap:
    def __init__(self, items: Optional[List[Tuple[Any, Any]]] = None) -> None:
        self._heap: List[Tuple[Any, Any]] = []
        self._pos: Dict[Any, int] = {}  # payload -> index, for decrease_key
        if items:
            self._heap = list(items)
            for i, (_, payload) in enumerate(self._heap):
                self._pos[payload] = i
            self._build_heap()

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return not self._heap

    def __contains__(self, payload: Any) -> bool:
        return payload in self._pos

    # ----------------------------------------------------------------- helpers
    def _swap(self, i: int, j: int) -> None:
        h = self._heap
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][1]] = i
        self._pos[h[j][1]] = j

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[i][0] < self._heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        n = len(self._heap)
        while True:
            left, right, smallest = 2 * i + 1, 2 * i + 2, i
            if left < n and self._heap[left][0] < self._heap[smallest][0]:
                smallest = left
            if right < n and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _build_heap(self) -> None:
        """Floyd's heapify: sift down each internal node, last to first. O(n)."""
        for i in range(len(self._heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    # ------------------------------------------------------------------- public
    def peek(self) -> Tuple[Any, Any]:
        if not self._heap:
            raise IndexError("peek from an empty heap")
        return self._heap[0]

    def insert(self, priority: Any, payload: Any) -> None:
        if payload in self._pos:
            raise KeyError(f"payload {payload!r} already in heap; use decrease_key")
        self._heap.append((priority, payload))
        i = len(self._heap) - 1
        self._pos[payload] = i
        self._sift_up(i)

    def extract_min(self) -> Tuple[Any, Any]:
        if not self._heap:
            raise IndexError("extract_min from an empty heap")
        top = self._heap[0]
        last = self._heap.pop()
        del self._pos[top[1]]
        if self._heap:  # move last item to root and sift down
            self._heap[0] = last
            self._pos[last[1]] = 0
            self._sift_down(0)
        return top

    def decrease_key(self, payload: Any, new_priority: Any) -> None:
        """Lower the priority of an existing payload, then re-heapify upward.

        Core of an efficient Dijkstra/Prim: when a shorter path to a frontier
        node is found we relax its key in place rather than pushing a duplicate.
        """
        i = self._pos.get(payload)
        if i is None:
            raise KeyError(f"payload {payload!r} not in heap")
        if new_priority > self._heap[i][0]:
            raise ValueError("decrease_key cannot raise a priority")
        self._heap[i] = (new_priority, payload)
        self._sift_up(i)

    def push_or_decrease(self, priority: Any, payload: Any) -> None:
        """Convenience for graph algorithms: insert, or decrease if cheaper."""
        if payload in self._pos:
            if priority < self._heap[self._pos[payload]][0]:
                self.decrease_key(payload, priority)
        else:
            self.insert(priority, payload)

    def replace_min(self, priority: Any, payload: Any) -> Tuple[Any, Any]:
        """Pop the min and push a new item in one sift-down. O(log n).

        Cheaper than extract_min + insert (one re-heapify instead of two) -- a
        common pattern when streaming items through a fixed-size frontier.
        """
        if not self._heap:
            raise IndexError("replace_min on an empty heap")
        if payload in self._pos:
            raise KeyError(f"payload {payload!r} already in heap")
        top = self._heap[0]
        del self._pos[top[1]]
        self._heap[0] = (priority, payload)
        self._pos[payload] = 0
        self._sift_down(0)
        return top


def heapsort(values: List[Any]) -> List[Any]:
    """Sort ascending by repeatedly extracting the min. O(n log n).

    Uses the min-heap itself as the sorting engine: build_heap is O(n), then n
    extract-min calls are O(n log n) overall.
    """
    heap = MinHeap([(v, i) for i, v in enumerate(values)])
    return [heap.extract_min()[0] for _ in range(len(values))]
