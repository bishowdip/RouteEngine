"""Levenshtein edit distance -- dynamic programming.

Route-engine use: fuzzy matching of mistyped place names (pairs naturally with
the trie autocomplete) -- the minimum number of single-character insertions,
deletions or substitutions to turn one string into another.

Recurrence (i, j over prefixes of a, b):
    D(i, 0) = i,   D(0, j) = j
    D(i, j) = D(i-1, j-1)                                  if a[i] == b[j]
            = 1 + min( D(i-1, j),    # delete
                       D(i, j-1),    # insert
                       D(i-1, j-1) ) # substitute          otherwise

Complexity: O(mn) time, O(min(m, n)) space with two rolling rows.
"""

from __future__ import annotations

from typing import List, Sequence


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (O(min(m,n)) space)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[len(b)]


def closest_match(query: str, candidates: List[str]) -> str:
    """Return the candidate with the smallest edit distance to ``query``."""
    if not candidates:
        raise ValueError("no candidates to match against")
    return min(candidates, key=lambda c: edit_distance(query, c))
