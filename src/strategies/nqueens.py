"""N-Queens -- a second backtracking example with constraint pruning.

Place n queens on an n x n board so none attack another. The backtracking
search places one queen per row and prunes any column/diagonal already attacked,
counting search nodes so the pruning benefit can be measured (mirroring the
graph-colouring study).

Worst case is exponential, but column/diagonal pruning makes moderate n
tractable. The solution count grows quickly (1, 0, 0, 2, 10, 4, 40, 92, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NQueensResult:
    solution: Optional[List[int]]   # solution[r] = column of the queen in row r
    nodes_explored: int
    found: bool


def solve_nqueens(n: int) -> NQueensResult:
    """Find one valid placement (or report none) with pruning."""
    cols: set = set()
    diag: set = set()         # r - c constant
    anti: set = set()         # r + c constant
    placement: List[int] = []
    counter = {"n": 0}

    def backtrack(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            counter["n"] += 1
            if col in cols or (row - col) in diag or (row + col) in anti:
                continue                      # prune: square is attacked
            cols.add(col); diag.add(row - col); anti.add(row + col)
            placement.append(col)
            if backtrack(row + 1):
                return True
            placement.pop()
            cols.discard(col); diag.discard(row - col); anti.discard(row + col)
        return False

    found = backtrack(0)
    return NQueensResult(list(placement) if found else None, counter["n"], found)


def count_solutions(n: int) -> int:
    """Count all distinct solutions (used to validate against known values)."""
    cols: set = set()
    diag: set = set()
    anti: set = set()
    total = 0

    def backtrack(row: int) -> None:
        nonlocal total
        if row == n:
            total += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag or (row + col) in anti:
                continue
            cols.add(col); diag.add(row - col); anti.add(row + col)
            backtrack(row + 1)
            cols.discard(col); diag.discard(row - col); anti.discard(row + col)

    backtrack(0)
    return total


def is_valid_solution(placement: List[int]) -> bool:
    """True if no two queens in ``placement`` attack each other."""
    n = len(placement)
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            c1, c2 = placement[r1], placement[r2]
            if c1 == c2 or abs(c1 - c2) == abs(r1 - r2):
                return False
    return True
