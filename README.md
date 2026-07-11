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

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python 3.11+ is required (developed on 3.13). `make install` runs the same
`pip install`.

## Command-line interface

```bash
route-engine info                      # graph size and connectivity summary
route-engine route <src> <dst>         # shortest path (Dijkstra / A*)
route-engine mst                       # minimum spanning tree weight
route-engine gui                       # launch the desktop app
```

The engine can also be run as a module without installing: `python -m src …`.

## Desktop GUI

An interactive Tkinter route-planner with click-to-route, alternative routes,
an MST overlay, zoom/pan and an ordered shortest-path panel:

```bash
make gui           # or: python -m src.gui   /   route-engine gui
```

## Tests

```bash
make test          # or: python -m pytest -q
```

## Library boundary (academic-integrity note)

`networkx` is used **only** to (a) load/convert the OpenStreetMap graph and
(b) act as a reference oracle to validate the hand-written algorithms — never as
the solution itself. `heapq` is **not** used for the priority queue; that is our
own `src/structures/min_heap.py`.
