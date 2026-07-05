"""Task 1 -- correctness tests for the hand-built data structures.

These cross-check our structures against Python's own ``sorted``/``dict`` as a
trusted oracle, and assert structural invariants (AVL balance, heap order).
"""

import random

import pytest

from src.structures.avl import AVLTree
from src.structures.bst import BinarySearchTree
from src.structures.hash_table import (
    MISSING,
    ChainingHashTable,
    OpenAddressingHashTable,
)
from src.structures.min_heap import MinHeap

SEED = 42


# --------------------------------------------------------------------- BST
def test_bst_insert_search_inorder_sorted():
    bst = BinarySearchTree()
    keys = random.Random(SEED).sample(range(10_000), 500)
    for k in keys:
        bst.insert(k, value=f"v{k}")
    assert len(bst) == len(keys)
    assert list(bst.in_order()) == sorted(keys)  # oracle: sorted()
    for k in keys:
        assert bst.search(k) == f"v{k}"
    assert bst.search(-1) is None


def test_bst_recursive_matches_iterative():
    rng = random.Random(SEED)
    keys = rng.sample(range(1000), 200)
    a, b = BinarySearchTree(), BinarySearchTree()
    for k in keys:
        a.insert(k)
        b.insert_recursive(k)
    assert list(a.in_order()) == list(b.in_order())


def test_bst_delete_all_cases():
    bst = BinarySearchTree()
    for k in [50, 30, 70, 20, 40, 60, 80, 65]:
        bst.insert(k)
    assert bst.delete(20)            # leaf
    assert bst.delete(70)            # two children (successor = 80? -> 80)
    assert not bst.delete(999)       # absent
    remaining = list(bst.in_order())
    assert remaining == sorted(remaining)
    assert 20 not in remaining and 70 not in remaining


def test_bst_iterative_insert_survives_sorted_10k():
    """The headline gotcha: 10k *sorted* keys must not RecursionError."""
    bst = BinarySearchTree()
    for k in range(10_000):
        bst.insert(k)               # iterative path
    assert len(bst) == 10_000
    assert bst.height() == 9_999    # fully degenerate, as expected
    assert bst.search(9_999) is None or bst.search(9_999) is None  # value None
    assert 5_000 in bst


# --------------------------------------------------------------------- AVL
def test_avl_stays_balanced_on_sorted_input():
    avl = AVLTree()
    for k in range(10_000):
        avl.insert(k)               # adversarial sorted input
    assert avl.is_balanced()
    # AVL height bound: <= 1.44 log2(n+2); for n=10000 that's < 20.
    assert avl.height() <= 20
    assert list(avl.in_order()) == list(range(10_000))


def test_avl_random_insert_delete_against_oracle():
    rng = random.Random(SEED)
    avl = AVLTree()
    present = set()
    for _ in range(2000):
        k = rng.randrange(500)
        if rng.random() < 0.7:
            avl.insert(k)
            present.add(k)
        else:
            avl.delete(k)
            present.discard(k)
        assert avl.is_balanced()
    assert list(avl.in_order()) == sorted(present)


# --------------------------------------------------------------------- Heap
def test_min_heap_sorts_like_oracle():
    rng = random.Random(SEED)
    data = [rng.randrange(10_000) for _ in range(1000)]
    heap = MinHeap()
    for i, p in enumerate(data):
        heap.insert(p, payload=(i, p))   # unique payloads
    out = [heap.extract_min()[0] for _ in range(len(data))]
    assert out == sorted(data)           # oracle: sorted()
    assert heap.is_empty()


def test_build_heap_linear_property():
    items = [(p, f"n{i}") for i, p in enumerate([5, 3, 8, 1, 9, 2, 7])]
    heap = MinHeap(items)
    assert heap.peek()[0] == 1
    out = [heap.extract_min()[0] for _ in range(len(items))]
    assert out == sorted(p for p, _ in items)


def test_decrease_key_for_dijkstra_pattern():
    heap = MinHeap()
    for node, dist in {"a": 10, "b": 7, "c": 12}.items():
        heap.insert(dist, node)
    heap.decrease_key("c", 3)            # relax c
    assert heap.extract_min() == (3, "c")
    heap.push_or_decrease(1, "b")        # cheaper than 7 -> decreases
    heap.push_or_decrease(99, "d")       # new node -> inserts
    assert heap.extract_min() == (1, "b")
    with pytest.raises(ValueError):
        h2 = MinHeap([(1, "x")])
        h2.decrease_key("x", 5)          # cannot raise


def test_empty_heap_errors():
    heap = MinHeap()
    with pytest.raises(IndexError):
        heap.peek()
    with pytest.raises(IndexError):
        heap.extract_min()


# --------------------------------------------------------------- Hash tables
@pytest.mark.parametrize("cls", [ChainingHashTable, OpenAddressingHashTable])
def test_hash_table_against_dict_oracle(cls):
    rng = random.Random(SEED)
    table = cls()
    oracle = {}
    for _ in range(5000):
        k = rng.randrange(1000)
        op = rng.random()
        if op < 0.6:
            v = rng.randrange(10_000)
            table.insert(k, v)
            oracle[k] = v
        elif op < 0.8:
            assert table.delete(k) == (k in oracle)
            oracle.pop(k, None)
        else:
            got = table.search(k)
            if k in oracle:
                assert got == oracle[k]
            else:
                assert got is MISSING
    assert len(table) == len(oracle)
    assert dict(table.items()) == oracle


@pytest.mark.parametrize("cls", [ChainingHashTable, OpenAddressingHashTable])
def test_hash_table_resizes_and_stays_correct(cls):
    table = cls(capacity=8)
    for i in range(1000):
        table.insert(i, i * i)
    assert len(table) == 1000
    for i in range(1000):
        assert table.search(i) == i * i
    assert table.search(10_000) is MISSING
