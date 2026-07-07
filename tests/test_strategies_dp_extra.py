"""Tests for coin-change DP and the Fenwick tree."""

import random

import pytest

from src.strategies.coin_change import count_ways, fewest_coins
from src.structures.fenwick import FenwickTree

SEED = 4


# ----------------------------------------------------------------- coin change
def test_fewest_coins_classic():
    n, used = fewest_coins([1, 5, 10, 25], 63)
    assert n == 6                                   # 25+25+10+1+1+1
    assert sum(used) == 63 and len(used) == 6


def test_fewest_coins_unreachable():
    n, used = fewest_coins([5, 10], 3)
    assert n == -1 and used is None


def test_fewest_coins_matches_bruteforce():
    rng = random.Random(SEED)
    for _ in range(40):
        coins = rng.sample(range(1, 12), rng.randint(1, 4))
        amount = rng.randint(0, 30)
        n, used = fewest_coins(coins, amount)
        # brute-force min coins via BFS over reachable amounts
        from collections import deque
        seen = {0}
        q = deque([(0, 0)])
        best = -1
        while q:
            a, depth = q.popleft()
            if a == amount:
                best = depth
                break
            for c in coins:
                if a + c <= amount and a + c not in seen:
                    seen.add(a + c)
                    q.append((a + c, depth + 1))
        assert n == best
        if n != -1:
            assert sum(used) == amount


def test_count_ways_classic():
    # ways to make 5 from {1,2,5}: 5, 1+2+2, 1+1+1+2, 1+1+1+1+1, 2+... = 4
    assert count_ways([1, 2, 5], 5) == 4
    assert count_ways([2], 3) == 0
    assert count_ways([1, 2, 5], 0) == 1


def test_coin_change_rejects_negative():
    with pytest.raises(ValueError):
        fewest_coins([1], -1)


# -------------------------------------------------------------------- Fenwick
def test_fenwick_prefix_and_range_sums():
    vals = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2]
    ft = FenwickTree.from_values(vals)
    pref = 0
    for i, v in enumerate(vals):
        pref += v
        assert ft.prefix_sum(i) == pref
    assert ft.range_sum(2, 5) == sum(vals[2:6])
    assert ft.range_sum(0, 9) == sum(vals)


def test_fenwick_point_updates_match_naive():
    rng = random.Random(SEED)
    n = 50
    arr = [0] * n
    ft = FenwickTree(n)
    for _ in range(300):
        i = rng.randrange(n)
        d = rng.randint(-5, 5)
        arr[i] += d
        ft.update(i, d)
        l = rng.randrange(n)
        r = rng.randrange(l, n)
        assert ft.range_sum(l, r) == sum(arr[l:r + 1])


def test_fenwick_index_bounds():
    ft = FenwickTree(5)
    with pytest.raises(IndexError):
        ft.update(5, 1)
