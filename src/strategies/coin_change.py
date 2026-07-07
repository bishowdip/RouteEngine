"""Coin-change dynamic programming.

Route-engine use: making exact change for a fare/toll with the fewest coins, and
counting how many distinct ways an amount can be paid. Two classic DP variants:

  * fewest-coins:  C(a) = 0 if a == 0 else 1 + min over coins c<=a of C(a-c)
  * count-ways:    number of multisets of coins summing to the amount

Complexity: O(amount * |coins|) time, O(amount) space for both. A clean contrast
between an optimisation DP (min) and a counting DP (sum) on the same structure.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def fewest_coins(coins: List[int], amount: int) -> Tuple[int, Optional[List[int]]]:
    """Return (min_coins, coins_used) or (-1, None) if amount is unreachable."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    INF = float("inf")
    best = [0] + [INF] * amount
    pick = [-1] * (amount + 1)              # which coin was used to reach a
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
                pick[a] = c
    if best[amount] == INF:
        return -1, None
    used: List[int] = []
    a = amount
    while a > 0:
        used.append(pick[a])
        a -= pick[a]
    return int(best[amount]), used


def count_ways(coins: List[int], amount: int) -> int:
    """Number of distinct multisets of coins summing to ``amount``."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    ways = [1] + [0] * amount
    for c in coins:                         # outer loop over coins => combinations
        for a in range(c, amount + 1):
            ways[a] += ways[a - c]
    return ways[amount]
