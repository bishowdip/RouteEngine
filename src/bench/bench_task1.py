"""Task 1 benchmarks: data-structure scaling, measured vs theoretical.

Run:  python -m src.bench.bench_task1
Generates into figures/:
  * task1_build_bst_vs_avl.png  -- sorted-input degradation (the headline)
  * task1_search.png            -- search cost vs n, measured vs O(log n)/O(n)
  * task1_hash_load_factor.png  -- chaining vs open addressing as alpha rises
and prints the timing tables used in the report.
"""

from __future__ import annotations

import random

from src.bench.harness import measure
from src.bench.plots import plot_measured_vs_theory
from src.structures.avl import AVLTree
from src.structures.bst import BinarySearchTree
from src.structures.hash_table import ChainingHashTable, OpenAddressingHashTable

SEED = 42
SIZES = [100, 250, 500, 1000, 2000, 5000, 10_000]


def _build_bst(keys):
    def run():
        t = BinarySearchTree()
        for k in keys:
            t.insert(k)  # iterative -> safe on sorted input
    return run


def _build_avl(keys):
    def run():
        t = AVLTree()
        for k in keys:
            t.insert(k)
    return run


def bench_build():
    """Total build time vs n for random and sorted insertion order."""
    rng = random.Random(SEED)
    bst_rand, bst_sort, avl_rand, avl_sort = ([] for _ in range(4))
    for n in SIZES:
        rand_keys = rng.sample(range(n * 10), n)
        sort_keys = list(range(n))
        bst_rand.append(measure(_build_bst(rand_keys), label="BST rand build", n=n, repeats=5).mean * 1e3)
        bst_sort.append(measure(_build_bst(sort_keys), label="BST sorted build", n=n, repeats=5).mean * 1e3)
        avl_rand.append(measure(_build_avl(rand_keys), label="AVL rand build", n=n, repeats=5).mean * 1e3)
        avl_sort.append(measure(_build_avl(sort_keys), label="AVL sorted build", n=n, repeats=5).mean * 1e3)
        print(f"n={n:<6} BSTrand={bst_rand[-1]:.3f}  BSTsort={bst_sort[-1]:.3f}  "
              f"AVLrand={avl_rand[-1]:.3f}  AVLsort={avl_sort[-1]:.3f} ms")

    plot_measured_vs_theory(
        {
            "BST sorted input": {"n": SIZES, "ms": bst_sort},
            "AVL sorted input": {"n": SIZES, "ms": avl_sort},
            "BST random input": {"n": SIZES, "ms": bst_rand},
        },
        title="Task 1: build time vs n -- BST degrades to O(n^2) on sorted input,\nAVL stays O(n log n)",
        filename="task1_build_bst_vs_avl.png",
        theory_for={
            "BST sorted input": "O(n^2)",
            "AVL sorted input": "O(n log n)",
            "BST random input": "O(n log n)",
        },
    )


def bench_search():
    """Average successful-search cost vs n; the per-op complexity contrast."""
    rng = random.Random(SEED)
    bst_rand_t, bst_sort_t, avl_t = [], [], []
    for n in SIZES:
        rand_keys = rng.sample(range(n * 10), n)
        sort_keys = list(range(n))
        probes = [rng.choice(rand_keys) for _ in range(2000)]
        probes_sorted = [rng.choice(sort_keys) for _ in range(2000)]

        bst_r = BinarySearchTree()
        for k in rand_keys:
            bst_r.insert(k)
        bst_s = BinarySearchTree()
        for k in sort_keys:
            bst_s.insert(k)
        avl = AVLTree()
        for k in rand_keys:
            avl.insert(k)

        bst_rand_t.append(measure(lambda: [bst_r.search(k) for k in probes],
                                  label="BST rand search", n=n, repeats=5).mean * 1e3)
        bst_sort_t.append(measure(lambda: [bst_s.search(k) for k in probes_sorted],
                                  label="BST sorted search", n=n, repeats=5).mean * 1e3)
        avl_t.append(measure(lambda: [avl.search(k) for k in probes],
                             label="AVL search", n=n, repeats=5).mean * 1e3)
        print(f"n={n:<6} BSTrand={bst_rand_t[-1]:.3f}  BSTsort={bst_sort_t[-1]:.3f}  AVL={avl_t[-1]:.3f} ms / 2000 probes")

    plot_measured_vs_theory(
        {
            "BST sorted (degenerate)": {"n": SIZES, "ms": bst_sort_t},
            "BST random": {"n": SIZES, "ms": bst_rand_t},
            "AVL": {"n": SIZES, "ms": avl_t},
        },
        title="Task 1: search cost vs n (2000 probes) -- O(n) degenerate BST vs O(log n)",
        filename="task1_search.png",
        theory_for={
            "BST sorted (degenerate)": "O(n)",
            "BST random": "O(log n)",
            "AVL": "O(log n)",
        },
    )


def bench_hash_load_factor():
    """Average insert+search time as the table fills -- chaining vs probing.

    We disable resizing (huge fixed capacity) so the load factor actually rises
    and the two strategies' divergence is visible.
    """
    capacity = 1 << 14  # 16384 fixed slots
    target_alphas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    rng = random.Random(SEED)
    keys = rng.sample(range(10_000_000), int(capacity * 0.96))

    chain_t, open_t, alphas = [], [], []
    for alpha in target_alphas:
        count = int(capacity * alpha)
        subset = keys[:count]
        # open addressing must stay below 1.0; cap just under capacity
        chain = ChainingHashTable(capacity=capacity, max_load=10.0)  # no resize
        opn = OpenAddressingHashTable(capacity=capacity, max_load=0.99)  # minimal resize
        for k in subset:
            chain.insert(k, k)
            opn.insert(k, k)
        probes = [rng.choice(subset) for _ in range(5000)]
        chain_t.append(measure(lambda: [chain.search(k) for k in probes],
                               label="chain search", n=count, repeats=5).mean * 1e3)
        open_t.append(measure(lambda: [opn.search(k) for k in probes],
                              label="open search", n=count, repeats=5).mean * 1e3)
        alphas.append(alpha)
        print(f"alpha={alpha:<5} chaining={chain_t[-1]:.3f}  open_addr={open_t[-1]:.3f} ms / 5000 probes")

    plot_measured_vs_theory(
        {
            "separate chaining": {"n": alphas, "ms": chain_t},
            "open addressing (quadratic)": {"n": alphas, "ms": open_t},
        },
        title="Task 1: hash-table search time vs load factor alpha",
        filename="task1_hash_load_factor.png",
        theory_for={},  # empirical only; no clean single-term overlay
        xlabel="load factor alpha = n / m",
    )


def main():
    print("=== Task 1: build time (random vs sorted) ===")
    bench_build()
    print("\n=== Task 1: search time ===")
    bench_search()
    print("\n=== Task 1: hash-table load factor ===")
    bench_hash_load_factor()


if __name__ == "__main__":
    main()
