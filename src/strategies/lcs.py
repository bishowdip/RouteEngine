"""Longest Common Subsequence -- a second dynamic-programming example.

Given sequences X (len m) and Y (len n), find the longest subsequence common to
both (not necessarily contiguous).

Recurrence:
    L(i, j) = 0                              if i == 0 or j == 0
            = L(i-1, j-1) + 1                if X[i] == Y[j]
            = max( L(i-1, j), L(i, j-1) )    otherwise

Complexity: O(mn) time, O(mn) space (O(min(m,n)) if only the length is needed).
LCS underpins diffing and similarity scoring -- e.g. comparing two recorded GPS
trajectories through the city by their shared ordered waypoints.
"""

from __future__ import annotations

from typing import Sequence, Tuple


def lcs(x: Sequence, y: Sequence) -> Tuple[int, list]:
    """Return (length, one_longest_common_subsequence)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # reconstruct by walking back from dp[m][n]
    seq: list = []
    i, j = m, n
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            seq.append(x[i - 1])
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    seq.reverse()
    return dp[m][n], seq


def lcs_length_optimised(x: Sequence, y: Sequence) -> int:
    """O(min(m,n)) space length-only variant (two rolling rows)."""
    if len(x) < len(y):
        x, y = y, x
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(y)]
