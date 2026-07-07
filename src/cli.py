"""Command-line interface for the route engine.

Examples:
    python -m src.cli info
    python -m src.cli route --source 0 --target 500
    python -m src.cli route --source 0 --target 500 --alternatives 3
    python -m src.cli mst --limit 1000

Loads the Kathmandu graph (cached OSM data, or a synthetic fallback) and answers
queries using the hand-written algorithms. A small, practical front-end that
demonstrates the components working together.
"""

from __future__ import annotations

import argparse
import sys

from src.data.loader import load_city_graph
from src.graph.astar import astar
from src.graph.dijkstra import dijkstra, reconstruct_path
from src.graph.k_shortest import k_shortest_paths
from src.graph.prim import prim_mst


def _load(limit):
    graph, source = load_city_graph(max_nodes=limit)
    print(f"loaded {graph} from {source} data")
    return graph


def cmd_info(args):
    g = _load(args.limit)
    print(f"nodes={g.num_nodes} edges={g.num_edges} density={g.density():.5f}")
    print(f"directed={g.directed}")


def cmd_route(args):
    g = _load(args.limit)
    if not g.has_node(args.source) or not g.has_node(args.target):
        print("error: source/target not in graph", file=sys.stderr)
        return 1
    dist, prev = dijkstra(g, args.source, target=args.target)
    if dist[args.target] == float("inf"):
        print("no route between the given nodes")
        return 0
    path = reconstruct_path(prev, args.source, args.target)
    cost_a, path_a, expanded = astar(g, args.source, args.target)
    print(f"Dijkstra: cost={dist[args.target]:.1f} m, {len(path)} hops")
    print(f"A*:       cost={cost_a:.1f} m, {len(path_a)} hops, {expanded} nodes expanded")
    if args.alternatives > 1:
        print(f"top {args.alternatives} alternative routes:")
        for i, (c, p) in enumerate(k_shortest_paths(g, args.source, args.target,
                                                     args.alternatives), 1):
            print(f"  {i}. cost={c:.1f} m, {len(p)} hops")
    return 0


def cmd_mst(args):
    g = _load(args.limit)
    edges, total = prim_mst(g)
    print(f"minimum spanning tree: {len(edges)} edges, total weight {total:.1f} m")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="route-engine", description=__doc__)
    p.add_argument("--limit", type=int, default=2000,
                   help="max nodes to load (subgraph size)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print graph statistics").set_defaults(func=cmd_info)

    r = sub.add_parser("route", help="shortest route between two nodes")
    r.add_argument("--source", type=int, required=True)
    r.add_argument("--target", type=int, required=True)
    r.add_argument("--alternatives", type=int, default=1,
                   help="also list this many alternative routes")
    r.set_defaults(func=cmd_route)

    m = sub.add_parser("mst", help="minimum spanning backbone")
    m.set_defaults(func=cmd_mst)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
