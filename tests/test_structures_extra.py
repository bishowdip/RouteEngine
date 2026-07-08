"""Tests for the additional data-structure operations."""

import random

import pytest

from src.structures.bst import BinarySearchTree
from src.structures.hash_table import ChainingHashTable, OpenAddressingHashTable
from src.structures.min_heap import MinHeap, heapsort

SEED = 8


# --------------------------------------------------------------------- BST ops
def test_bst_min_max_successor():
    bst = BinarySearchTree()
    for k in [50, 30, 70, 20, 40, 60, 80]:
        bst.insert(k)
    assert bst.min_key() == 20
    assert bst.max_key() == 80
    assert bst.successor(50) == 60
    assert bst.successor(40) == 50
    assert bst.successor(80) is None        # nothing larger
    assert bst.successor(10) == 20


def test_bst_min_max_empty_raises():
    bst = BinarySearchTree()
    with pytest.raises(ValueError):
        bst.min_key()
    with pytest.raises(ValueError):
        bst.max_key()


def test_bst_level_order_root_first():
    bst = BinarySearchTree()
    for k in [50, 30, 70, 20, 40]:
        bst.insert(k)
    levels = list(bst.level_order())
    assert levels[0] == 50                  # root first
    assert set(levels) == {50, 30, 70, 20, 40}


# -------------------------------------------------------------------- heap ops
def test_heapsort_matches_sorted():
    rng = random.Random(SEED)
    for _ in range(30):
        data = [rng.randrange(1000) for _ in range(rng.randint(0, 50))]
        assert heapsort(data) == sorted(data)


def test_replace_min_keeps_heap_property():
    heap = MinHeap()
    for p in [5, 3, 8, 1]:
        heap.insert(p, f"n{p}")
    popped = heap.replace_min(10, "new")
    assert popped[0] == 1                    # old min returned
    # remaining come out sorted, including the replacement
    out = [heap.extract_min()[0] for _ in range(len(heap))]
    assert out == sorted(out)
    assert 10 in out


# ------------------------------------------------------------- hash iteration
@pytest.mark.parametrize("cls", [ChainingHashTable, OpenAddressingHashTable])
def test_hash_iteration_and_get(cls):
    table = cls()
    for i in range(50):
        table.insert(i, i * 10)
    assert sorted(table.keys()) == list(range(50))
    assert sorted(table.values()) == [i * 10 for i in range(50)]
    assert sorted(iter(table)) == list(range(50))
    assert table.get(7) == 70
    assert table.get(999, default=-1) == -1
