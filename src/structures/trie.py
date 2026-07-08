"""Prefix tree (trie) for string keys.

Route-engine use: fast street-/place-name autocomplete -- given a typed prefix,
return all matching names. A trie stores shared prefixes once, giving lookup and
prefix queries that depend on the key length, not the number of stored keys.

Complexity (L = key length, P = prefix length, k = matches):
    insert / search : O(L)
    starts_with     : O(P + total length of matches)
Space: O(total characters across all keys) in the worst case.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class _TrieNode:
    __slots__ = ("children", "is_end", "value")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.is_end = False
        self.value: Optional[object] = None


class Trie:
    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, key: str, value: object = None) -> None:
        node = self._root
        for ch in key:
            node = node.children.setdefault(ch, _TrieNode())
        if not node.is_end:
            self._size += 1
        node.is_end = True
        node.value = value

    def __contains__(self, key: str) -> bool:
        node = self._find(key)
        return node is not None and node.is_end

    def get(self, key: str) -> Optional[object]:
        node = self._find(key)
        return node.value if node and node.is_end else None

    def _find(self, key: str) -> Optional[_TrieNode]:
        node = self._root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def delete(self, key: str) -> bool:
        """Remove ``key`` if present, pruning now-empty nodes. O(L)."""
        path = [self._root]
        for ch in key:
            nxt = path[-1].children.get(ch)
            if nxt is None:
                return False
            path.append(nxt)
        leaf = path[-1]
        if not leaf.is_end:
            return False
        leaf.is_end = False
        leaf.value = None
        self._size -= 1
        # prune childless, non-terminal nodes bottom-up
        for depth in range(len(key) - 1, -1, -1):
            node = path[depth + 1]
            if node.children or node.is_end:
                break
            del path[depth].children[key[depth]]
        return True

    def starts_with(self, prefix: str) -> List[str]:
        """All stored keys beginning with ``prefix`` (autocomplete)."""
        node = self._find(prefix)
        if node is None:
            return []
        out: List[str] = []
        self._collect(node, prefix, out)
        return out

    def _collect(self, node: _TrieNode, prefix: str, out: List[str]) -> None:
        if node.is_end:
            out.append(prefix)
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, out)
