"""Tests for the additional algorithmic-strategy examples."""

import random

from src.strategies.huffman import (
    average_code_length,
    build_codes,
    encode,
    is_prefix_free,
)
from src.strategies.lcs import lcs, lcs_length_optimised
from src.strategies.nqueens import (
    count_solutions,
    is_valid_solution,
    solve_nqueens,
)

SEED = 9


# ------------------------------------------------------------------- Huffman
def test_huffman_prefix_free_and_optimal_length():
    freq = {"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5}  # CLRS example
    codes = build_codes(freq)
    assert is_prefix_free(codes)
    # CLRS optimal average length for this table is 2.24 bits/symbol
    assert abs(average_code_length(freq, codes) - 2.24) < 1e-9
    # frequent symbols get codes no longer than rare ones
    assert len(codes["a"]) <= len(codes["f"])


def test_huffman_roundtrip_decode():
    freq = {"x": 5, "y": 2, "z": 1}
    codes = build_codes(freq)
    bits = encode("xxyz", codes)
    # decode by walking the prefix-free code
    inv = {v: k for k, v in codes.items()}
    decoded, buf = "", ""
    for b in bits:
        buf += b
        if buf in inv:
            decoded += inv[buf]
            buf = ""
    assert decoded == "xxyz"


def test_huffman_single_symbol():
    codes = build_codes({"a": 10})
    assert codes == {"a": "0"}


# ----------------------------------------------------------------------- LCS
def test_lcs_classic():
    length, seq = lcs("ABCBDAB", "BDCAB")
    assert length == 4
    assert "".join(seq) in {"BCAB", "BDAB"}


def test_lcs_against_optimised_length():
    rng = random.Random(SEED)
    alphabet = "ABCD"
    for _ in range(50):
        x = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 15)))
        y = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 15)))
        length, seq = lcs(x, y)
        assert length == lcs_length_optimised(x, y)
        assert len(seq) == length


# ------------------------------------------------------------------- N-Queens
def test_nqueens_finds_valid_solution():
    for n in [1, 4, 5, 6, 8]:
        res = solve_nqueens(n)
        assert res.found
        assert is_valid_solution(res.solution)


def test_nqueens_no_solution_small():
    for n in [2, 3]:
        assert not solve_nqueens(n).found


def test_nqueens_solution_counts_match_known():
    # OEIS A000170: solutions for n = 1..8
    known = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92}
    for n, count in known.items():
        assert count_solutions(n) == count
