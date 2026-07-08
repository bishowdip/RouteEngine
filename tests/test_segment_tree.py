"""Tests for the segment tree (sum and min range queries)."""

import random

import pytest

from src.structures.segment_tree import SegmentTree

SEED = 6


def test_sum_queries_match_naive():
    rng = random.Random(SEED)
    arr = [rng.randint(-10, 10) for _ in range(40)]
    st = SegmentTree(arr)
    for _ in range(200):
        l = rng.randrange(len(arr))
        r = rng.randrange(l, len(arr))
        assert st.query(l, r) == sum(arr[l:r + 1])


def test_min_queries_with_updates():
    arr = [5, 2, 7, 1, 9, 3]
    st = SegmentTree(arr, combine=min, identity=float("inf"))
    assert st.query(0, 5) == 1
    assert st.query(0, 2) == 2
    st.update(3, 10)                 # the former minimum is replaced
    arr[3] = 10
    assert st.query(0, 5) == 2
    assert st.query(2, 4) == 7


def test_point_updates_track_naive_sum():
    rng = random.Random(SEED)
    arr = [0] * 30
    st = SegmentTree(arr)
    for _ in range(200):
        i = rng.randrange(30)
        v = rng.randint(-5, 5)
        arr[i] = v
        st.update(i, v)
        l = rng.randrange(30)
        r = rng.randrange(l, 30)
        assert st.query(l, r) == sum(arr[l:r + 1])


def test_invalid_ranges_raise():
    st = SegmentTree([1, 2, 3])
    with pytest.raises(IndexError):
        st.query(0, 3)
    with pytest.raises(IndexError):
        st.update(5, 0)
