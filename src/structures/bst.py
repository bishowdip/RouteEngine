"""Binary Search Tree.

Hand-implemented for ST5003CEM Task 1. Supports both a recursive and an
iterative insert: the iterative path is essential for the *sorted-input
degradation* benchmark, where a recursive insert on 10,000 ordered keys would
blow the Python recursion limit. Keys map to a payload (e.g. an intersection
id -> node metadata) so the tree doubles as the city's intersection index.

Average-case complexity (random insertion order, height ~ log n):
    insert / search / delete : O(log n)
Worst case (sorted input, height = n -> degenerate "linked list"):
    insert / search / delete : O(n)

The worst case is exactly what we demonstrate empirically against the AVL tree.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional


class BSTNode:
    __slots__ = ("key", "value", "left", "right")

    def __init__(self, key: Any, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.left: Optional["BSTNode"] = None
        self.right: Optional["BSTNode"] = None


class BinarySearchTree:
    """Unbalanced binary search tree keyed on a comparable ``key``."""

    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        # Direct traversal, not search(): a key may legitimately store None.
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    # ------------------------------------------------------------------ insert
    def insert(self, key: Any, value: Any = None) -> None:
        """Iterative insert -- the safe default.

        O(h) where h is the tree height. Uses no recursion, so it survives
        adversarial sorted input that would otherwise raise ``RecursionError``.
        """
        if self.root is None:
            self.root = BSTNode(key, value)
            self._size = 1
            return
        cur = self.root
        while True:
            if key == cur.key:
                cur.value = value  # update existing key, no duplicate insert
                return
            if key < cur.key:
                if cur.left is None:
                    cur.left = BSTNode(key, value)
                    self._size += 1
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = BSTNode(key, value)
                    self._size += 1
                    return
                cur = cur.right

    def insert_recursive(self, key: Any, value: Any = None) -> None:
        """Recursive insert -- used only for the random-order benchmark.

        Kept to satisfy the brief's "provide both" requirement and to contrast
        recursive vs iterative style. NOT safe for sorted bulk input.
        """
        self.root, inserted = self._insert_rec(self.root, key, value)
        if inserted:
            self._size += 1

    def _insert_rec(self, node: Optional[BSTNode], key: Any, value: Any):
        if node is None:
            return BSTNode(key, value), True
        if key == node.key:
            node.value = value
            return node, False
        if key < node.key:
            node.left, inserted = self._insert_rec(node.left, key, value)
        else:
            node.right, inserted = self._insert_rec(node.right, key, value)
        return node, inserted

    # ------------------------------------------------------------------ search
    def search(self, key: Any) -> Optional[Any]:
        """Return the stored value for ``key`` or ``None`` if absent. O(h)."""
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return cur.value
            cur = cur.left if key < cur.key else cur.right
        return None

    # ------------------------------------------------------------------ delete
    def delete(self, key: Any) -> bool:
        """Delete ``key``; return True if it existed. O(h).

        Standard three-case delete (leaf / single child / two children).
        For the two-child case we splice in the in-order successor.
        """
        parent: Optional[BSTNode] = None
        cur = self.root
        while cur is not None and cur.key != key:
            parent = cur
            cur = cur.left if key < cur.key else cur.right
        if cur is None:
            return False  # not found

        # Node with two children: replace with in-order successor, then delete
        # the successor (which has at most one child) from the right subtree.
        if cur.left is not None and cur.right is not None:
            succ_parent = cur
            succ = cur.right
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            cur.key, cur.value = succ.key, succ.value
            parent, cur = succ_parent, succ  # fall through to <=1 child case

        child = cur.left if cur.left is not None else cur.right
        if parent is None:
            self.root = child
        elif parent.left is cur:
            parent.left = child
        else:
            parent.right = child
        self._size -= 1
        return True

    # ----------------------------------------------------------------- metrics
    def height(self) -> int:
        """Height in edges; empty tree -1, single node 0.

        Computed iteratively (explicit stack of (node, depth)) so it survives a
        fully degenerate tree of 10k sorted keys without a RecursionError.
        """
        if self.root is None:
            return -1
        max_depth = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
            if node.left is not None:
                stack.append((node.left, depth + 1))
            if node.right is not None:
                stack.append((node.right, depth + 1))
        return max_depth

    def in_order(self) -> Iterator[Any]:
        """Yield keys in sorted order (iterative to avoid deep recursion)."""
        stack, cur = [], self.root
        while stack or cur is not None:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            yield cur.key
            cur = cur.right
