"""Self-balancing AVL tree.

Hand-implemented for ST5003CEM Task 1. An AVL tree keeps every node's
balance factor (height of left subtree minus height of right subtree) in
{-1, 0, +1} via the four rotation cases (LL, RR, LR, RL). This guarantees
height <= 1.44 * log2(n+2), so insert/search/delete are *worst-case* O(log n)
even on sorted input -- the property we contrast against the plain BST.

All four operations: O(log n) worst case, O(log n) average.
Space: O(n).

The insert/delete here are recursive but the recursion depth is bounded by the
tree height (~O(log n)), so unlike the BST there is no RecursionError risk even
on 10,000 sorted keys.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional


class AVLNode:
    __slots__ = ("key", "value", "left", "right", "height")

    def __init__(self, key: Any, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.left: Optional["AVLNode"] = None
        self.right: Optional["AVLNode"] = None
        self.height = 0  # height in edges of subtree rooted here; leaf == 0


class AVLTree:
    def __init__(self) -> None:
        self.root: Optional[AVLNode] = None
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

    # --------------------------------------------------------------- internals
    @staticmethod
    def _h(node: Optional[AVLNode]) -> int:
        return -1 if node is None else node.height

    def _update_height(self, node: AVLNode) -> None:
        node.height = 1 + max(self._h(node.left), self._h(node.right))

    def _balance(self, node: Optional[AVLNode]) -> int:
        if node is None:
            return 0
        return self._h(node.left) - self._h(node.right)

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """LL fix: pivot ``y`` down-right, its left child ``x`` up."""
        x = y.left
        assert x is not None
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """RR fix: pivot ``x`` down-left, its right child ``y`` up."""
        y = x.right
        assert y is not None
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node: AVLNode) -> AVLNode:
        self._update_height(node)
        bf = self._balance(node)
        if bf > 1:  # left-heavy
            if self._balance(node.left) < 0:  # LR
                node.left = self._rotate_left(node.left)  # type: ignore[arg-type]
            return self._rotate_right(node)  # LL (or finished LR)
        if bf < -1:  # right-heavy
            if self._balance(node.right) > 0:  # RL
                node.right = self._rotate_right(node.right)  # type: ignore[arg-type]
            return self._rotate_left(node)  # RR (or finished RL)
        return node

    # ------------------------------------------------------------------ insert
    def insert(self, key: Any, value: Any = None) -> None:
        self.root, inserted = self._insert(self.root, key, value)
        if inserted:
            self._size += 1

    def _insert(self, node: Optional[AVLNode], key: Any, value: Any):
        if node is None:
            return AVLNode(key, value), True
        if key == node.key:
            node.value = value
            return node, False
        if key < node.key:
            node.left, inserted = self._insert(node.left, key, value)
        else:
            node.right, inserted = self._insert(node.right, key, value)
        return self._rebalance(node), inserted

    # ------------------------------------------------------------------ search
    def search(self, key: Any) -> Optional[Any]:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return cur.value
            cur = cur.left if key < cur.key else cur.right
        return None

    # ------------------------------------------------------------------ delete
    def delete(self, key: Any) -> bool:
        self.root, deleted = self._delete(self.root, key)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node: Optional[AVLNode], key: Any):
        if node is None:
            return None, False
        if key < node.key:
            node.left, deleted = self._delete(node.left, key)
        elif key > node.key:
            node.right, deleted = self._delete(node.right, key)
        else:
            deleted = True
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            # two children: copy in-order successor up, delete it below
            succ = node.right
            while succ.left is not None:
                succ = succ.left
            node.key, node.value = succ.key, succ.value
            node.right, _ = self._delete(node.right, succ.key)
        return self._rebalance(node), deleted

    # ----------------------------------------------------------------- metrics
    def height(self) -> int:
        return self._h(self.root)

    def is_balanced(self) -> bool:
        """Assert the AVL invariant holds at every node (used in tests)."""
        def check(node: Optional[AVLNode]) -> bool:
            if node is None:
                return True
            if abs(self._balance(node)) > 1:
                return False
            return check(node.left) and check(node.right)

        return check(self.root)

    def in_order(self) -> Iterator[Any]:
        stack, cur = [], self.root
        while stack or cur is not None:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            yield cur.key
            cur = cur.right
