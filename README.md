# RouteEngine — ST5003CEM Advanced Algorithms Coursework

A unified **city route-planning and logistics engine for Kathmandu**. Every
coursework task is a component of one coherent system rather than a standalone
script. All data structures and algorithms are **hand-implemented** in `src/`;
third-party libraries are used only for data loading, visualisation, plotting,
and *validating* the results against a reference oracle.

## Tasks

| Task | Topic | Module(s) |
|---|---|---|
| 1 | Advanced data structures — BST, AVL, min-heap, hash table, union-find, Fenwick, segment tree, trie | `src/structures/` |
| 2 | Graph algorithms — Dijkstra, A\*, bidirectional, Bellman-Ford, Floyd-Warshall, Johnson, Prim, Kruskal, SCC, topological sort, max-flow, Eulerian, k-shortest | `src/graph/` |
| 3 | Algorithm strategies — DP (knapsack, LCS, coin-change, edit distance), greedy (Huffman), backtracking (N-Queens, colouring) | `src/strategies/` |
| 4 | NP-hard TSP + heuristics — nearest-neighbour, insertion, 2-opt, or-opt, simulated annealing, GRASP | `src/heuristics/` |
| 5 | Concurrency — threading vs multiprocessing for multi-source shortest paths | `src/concurrency/` |

## Library boundary (academic-integrity note)

`networkx` is used **only** to (a) load/convert the OpenStreetMap graph and
(b) act as a reference oracle to validate the hand-written algorithms — never as
the solution itself. `heapq` is **not** used for the priority queue; that is our
own `src/structures/min_heap.py`.
