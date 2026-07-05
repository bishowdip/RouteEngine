"""0/1 Knapsack via dynamic programming -- ST5003CEM Task 3 (DP component).

Route-engine framing: a delivery vehicle with a fixed weight capacity W must
choose which parcels (each with a weight and a value/priority) to load to
maximise delivered value -- the classic 0/1 knapsack.

Recurrence (items 1..n, capacity w):
    K(i, w) = K(i-1, w)                                  if weight[i] > w
            = max( K(i-1, w),                            (skip item i)
                   value[i] + K(i-1, w - weight[i]) )    (take item i)
    base:  K(0, w) = 0  for all w   (no items -> no value)

Complexity:
    full table   : time O(nW), space O(nW)
    rolling rows : time O(nW), space O(W)   -- iterate w downward to reuse row
Note O(nW) is *pseudo-polynomial*: W is a magnitude, not the input length, so the
cost grows with the numeric capacity, not just the item count.
"""

from __future__ import annotations

from typing import List, Tuple


def knapsack_full_table(weights: List[int], values: List[float], capacity: int
                        ) -> Tuple[float, List[int], List[List[float]]]:
    """Return (best_value, chosen_item_indices, dp_table).

    The full table is returned so the report can show it and reconstruct the
    chosen set by back-tracking through the decisions.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if len(weights) != len(values):
        raise ValueError("weights and values must align")
    if any(w < 0 for w in weights):
        raise ValueError("item weights must be non-negative")

    n = len(weights)
    # dp[i][w] = best value using the first i items within capacity w
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = weights[i - 1], values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]                       # skip item i
            if wi <= w:
                take = vi + dp[i - 1][w - wi]
                if take > dp[i][w]:
                    dp[i][w] = take                       # take item i

    # Reconstruct chosen items by walking the decisions backward.
    chosen: List[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:                      # item i was taken
            chosen.append(i - 1)
            w -= weights[i - 1]
    chosen.reverse()
    return dp[n][capacity], chosen, dp


def knapsack_space_optimised(weights: List[int], values: List[float], capacity: int) -> float:
    """O(W) space variant: a single rolling row, iterating w *downward*.

    Iterating capacity high->low guarantees each item is used at most once (0/1):
    dp[w] still holds the previous row's value for the smaller capacity we read.
    Returns the optimum value only (no reconstruction in O(W) space).
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    dp = [0.0] * (capacity + 1)
    for wi, vi in zip(weights, values):
        for w in range(capacity, wi - 1, -1):
            cand = vi + dp[w - wi]
            if cand > dp[w]:
                dp[w] = cand
    return dp[capacity]


def knapsack_bruteforce(weights: List[int], values: List[float], capacity: int) -> float:
    """O(2^n) reference used only to validate the DP on small instances."""
    n = len(weights)
    best = 0.0
    for mask in range(1 << n):
        w = sum(weights[i] for i in range(n) if mask & (1 << i))
        if w <= capacity:
            v = sum(values[i] for i in range(n) if mask & (1 << i))
            best = max(best, v)
    return best
