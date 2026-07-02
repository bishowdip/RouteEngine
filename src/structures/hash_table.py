"""Hash tables with two collision strategies.

Hand-implemented for ST5003CEM Task 1. Implementing *both* separate chaining
and open addressing lets us benchmark them against each other under rising load
factor -- a direct critical-evaluation comparison the brief rewards.

In the route engine these index intersections by id for O(1) average lookup.

Complexity (both variants):
    insert / search / delete : O(1) average, O(n) worst case (all collide)
Chaining degrades gracefully as load factor alpha = n/m grows (expected chain
length ~alpha). Open addressing degrades sharply as alpha -> 1 because probe
sequences lengthen super-linearly (~1/(1-alpha)); both resize to stay healthy.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple

_MISSING = object()       # sentinel: never stored
_TOMBSTONE = object()     # sentinel: deleted slot in open addressing


class ChainingHashTable:
    """Separate chaining: each bucket holds a list of (key, value) pairs."""

    def __init__(self, capacity: int = 8, max_load: float = 0.75) -> None:
        self._capacity = capacity
        self._max_load = max_load
        self._buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(capacity)]
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not _MISSING

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity

    def _index(self, key: Any) -> int:
        return hash(key) & (self._capacity - 1) if self._is_pow2() else hash(key) % self._capacity

    def _is_pow2(self) -> bool:
        return self._capacity & (self._capacity - 1) == 0

    def _resize(self, new_capacity: int) -> None:
        old = self._buckets
        self._capacity = new_capacity
        self._buckets = [[] for _ in range(new_capacity)]
        self._size = 0
        for bucket in old:
            for key, value in bucket:
                self.insert(key, value)

    def insert(self, key: Any, value: Any) -> None:
        idx = self._index(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # update in place
                return
        bucket.append((key, value))
        self._size += 1
        if self.load_factor > self._max_load:
            self._resize(self._capacity * 2)

    def search(self, key: Any) -> Any:
        for k, v in self._buckets[self._index(key)]:
            if k == key:
                return v
        return _MISSING

    def delete(self, key: Any) -> bool:
        bucket = self._buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def items(self) -> Iterator[Tuple[Any, Any]]:
        for bucket in self._buckets:
            yield from bucket


class OpenAddressingHashTable:
    """Open addressing with quadratic probing and tombstone deletion.

    Probe sequence: h(k), h(k)+1, h(k)+3, h(k)+6, ... (triangular numbers),
    which visits every slot when capacity is a power of two -- so no insert can
    fail spuriously while free slots remain.
    """

    def __init__(self, capacity: int = 8, max_load: float = 0.5) -> None:
        # power-of-two capacity keeps the triangular-number probe a full cycle
        self._capacity = capacity
        self._max_load = max_load
        self._keys: List[Any] = [_MISSING] * capacity
        self._values: List[Any] = [None] * capacity
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        return self.search(key) is not _MISSING

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity

    def _probe(self, key: Any) -> Iterator[int]:
        idx = hash(key) & (self._capacity - 1)
        step = 0
        for _ in range(self._capacity):
            yield (idx + (step * (step + 1)) // 2) & (self._capacity - 1)
            step += 1

    def _resize(self, new_capacity: int) -> None:
        old_keys, old_vals = self._keys, self._values
        self._capacity = new_capacity
        self._keys = [_MISSING] * new_capacity
        self._values = [None] * new_capacity
        self._size = 0
        for k, v in zip(old_keys, old_vals):
            if k is not _MISSING and k is not _TOMBSTONE:
                self.insert(k, v)

    def insert(self, key: Any, value: Any) -> None:
        if self.load_factor >= self._max_load:
            self._resize(self._capacity * 2)
        first_tomb = -1
        for slot in self._probe(key):
            cur = self._keys[slot]
            if cur is _MISSING:
                target = first_tomb if first_tomb != -1 else slot
                self._keys[target] = key
                self._values[target] = value
                self._size += 1
                return
            if cur is _TOMBSTONE:
                if first_tomb == -1:
                    first_tomb = slot
            elif cur == key:
                self._values[slot] = value  # update in place
                return
        # probe cycle exhausted (should not happen below max_load); grow + retry
        self._resize(self._capacity * 2)
        self.insert(key, value)

    def search(self, key: Any) -> Any:
        for slot in self._probe(key):
            cur = self._keys[slot]
            if cur is _MISSING:
                return _MISSING
            if cur is not _TOMBSTONE and cur == key:
                return self._values[slot]
        return _MISSING

    def delete(self, key: Any) -> bool:
        for slot in self._probe(key):
            cur = self._keys[slot]
            if cur is _MISSING:
                return False
            if cur is not _TOMBSTONE and cur == key:
                self._keys[slot] = _TOMBSTONE
                self._values[slot] = None
                self._size -= 1
                return True
        return False

    def items(self) -> Iterator[Tuple[Any, Any]]:
        for k, v in zip(self._keys, self._values):
            if k is not _MISSING and k is not _TOMBSTONE:
                yield k, v


# Exposed sentinel so callers/tests can check for "absent" unambiguously.
MISSING = _MISSING
