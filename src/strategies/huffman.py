"""Huffman coding -- a second greedy algorithm with an optimality guarantee.

Builds an optimal prefix-free binary code by repeatedly merging the two
lowest-frequency nodes (using the hand-written min-heap as the priority queue)
until a single tree remains. The greedy choice -- always merge the two rarest
symbols -- is provably optimal via an exchange argument: in some optimal tree the
two least-frequent symbols are siblings at maximum depth, so merging them first
loses nothing.

Complexity: O(n log n) for n distinct symbols (n-1 merges, each O(log n) heap
ops). In the route engine this compresses repetitive log/telemetry streams.
"""

from __future__ import annotations

from typing import Dict

from src.structures.min_heap import MinHeap


class _Node:
    __slots__ = ("freq", "symbol", "left", "right")

    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right


def build_codes(frequencies: Dict[str, int]) -> Dict[str, str]:
    """Return symbol -> binary code string for the given frequency table."""
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        # single symbol still needs one bit
        return {next(iter(frequencies)): "0"}

    # MinHeap stores (priority, payload). Priority is (freq, counter) so equal
    # frequencies break ties deterministically; payload is an integer node id
    # into a side table (heap payloads must be hashable, trees are not).
    heap = MinHeap()
    nodes: Dict[int, _Node] = {}
    counter = 0
    for sym, freq in frequencies.items():
        nodes[counter] = _Node(freq, sym)
        heap.insert((freq, counter), counter)
        counter += 1

    while len(heap) > 1:
        _, a_id = heap.extract_min()
        _, b_id = heap.extract_min()
        a, b = nodes[a_id], nodes[b_id]
        merged = _Node(a.freq + b.freq, None, a, b)
        nodes[counter] = merged
        heap.insert((merged.freq, counter), counter)
        counter += 1

    _, root_id = heap.extract_min()
    codes: Dict[str, str] = {}

    def walk(node: _Node, prefix: str) -> None:
        if node.symbol is not None:
            codes[node.symbol] = prefix
            return
        if node.left:
            walk(node.left, prefix + "0")
        if node.right:
            walk(node.right, prefix + "1")

    walk(nodes[root_id], "")
    return codes


def encode(text: str, codes: Dict[str, str]) -> str:
    return "".join(codes[ch] for ch in text)


def average_code_length(frequencies: Dict[str, int], codes: Dict[str, str]) -> float:
    total = sum(frequencies.values())
    if total == 0:
        return 0.0
    return sum(frequencies[s] * len(codes[s]) for s in frequencies) / total


def is_prefix_free(codes: Dict[str, str]) -> bool:
    """No code word may be a prefix of another (the defining property)."""
    words = sorted(codes.values())
    for i in range(len(words) - 1):
        if words[i + 1].startswith(words[i]):
            return False
    return True
