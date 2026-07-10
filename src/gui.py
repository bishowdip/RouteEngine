"""Tkinter desktop GUI for the route engine.

An interactive map of the Kathmandu road graph: click two intersections to pick a
source and target, then compute the shortest route with the hand-written Dijkstra
or A*, list Yen's alternative routes, or draw the minimum spanning backbone. The
whole engine (Tasks 1-5 plus the extended library) drives a single visual front
end. Tkinter ships with Python, so there is no extra dependency.

Run:  python -m src.gui            (or  python -m src.gui --limit 1500)
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional

from src.data.loader import load_city_graph
from src.graph.astar import astar
from src.graph.dijkstra import dijkstra, reconstruct_path
from src.graph.k_shortest import k_shortest_paths
from src.graph.prim import prim_mst
from src.gui_support import format_distance, nearest_node, project_nodes

BG = "#0f1419"
EDGE = "#2b3640"
NODE = "#5a6b78"
ROUTE = "#e23c43"
SOURCE = "#39d353"
TARGET = "#f5a623"
ALT_COLOURS = ["#e23c43", "#f5a623", "#3b82f6", "#a855f7", "#14b8a6"]


class RouteEngineGUI:
    """The main application window."""

    def __init__(self, root: tk.Tk, graph, source_kind: str) -> None:
        self.root = root
        self.graph = graph
        self.source_kind = source_kind
        self.positions: Dict[int, tuple] = {}
        self.source: Optional[int] = None
        self.target: Optional[int] = None

        root.title("Kathmandu Route Engine")
        root.configure(bg=BG)
        root.geometry("1100x720")

        self._build_controls()
        self._build_canvas()
        self._build_statusbar()
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)
        self._set_status(f"Loaded {graph.num_nodes} intersections, "
                         f"{graph.num_edges} roads ({source_kind} data). "
                         f"Click the map to choose a start, then a destination.")

    # ------------------------------------------------------------------ layout
    def _build_controls(self) -> None:
        bar = tk.Frame(self.root, bg="#161b22", padx=10, pady=8)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Algorithm:", bg="#161b22", fg="#c9d1d9").pack(side=tk.LEFT)
        self.algo = tk.StringVar(value="Dijkstra")
        for name in ("Dijkstra", "A*"):
            ttk.Radiobutton(bar, text=name, value=name, variable=self.algo,
                            command=self._recompute).pack(side=tk.LEFT, padx=4)

        ttk.Button(bar, text="Find route", command=self._recompute).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(bar, text="3 alternatives", command=self._show_alternatives).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Spanning tree", command=self._show_mst).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=4)

        self.info = tk.Label(bar, text="", bg="#161b22", fg="#8b949e")
        self.info.pack(side=tk.RIGHT)

    def _build_canvas(self) -> None:
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_statusbar(self) -> None:
        self.status = tk.Label(self.root, text="", bg="#161b22", fg="#c9d1d9",
                               anchor="w", padx=10, pady=4)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------ drawing
    def _reproject(self) -> None:
        w = self.canvas.winfo_width() or 1000
        h = self.canvas.winfo_height() or 600
        self.positions = project_nodes(self.graph.coords, w, h)

    def _draw_base_graph(self) -> None:
        self.canvas.delete("all")
        pos = self.positions
        # edges first so nodes sit on top
        for u, v, _ in self.graph.edges():
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                self.canvas.create_line(x1, y1, x2, y2, fill=EDGE, width=1)
        for n, (x, y) in pos.items():
            self.canvas.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5,
                                    fill=NODE, outline="")
        self._draw_markers()

    def _draw_markers(self) -> None:
        self.canvas.delete("marker")
        for node, colour, label in ((self.source, SOURCE, "S"), (self.target, TARGET, "T")):
            if node is not None and node in self.positions:
                x, y = self.positions[node]
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=colour,
                                        outline="white", width=2, tags="marker")
                self.canvas.create_text(x, y - 14, text=label, fill="white",
                                        font=("Helvetica", 10, "bold"), tags="marker")

    def _draw_path(self, path: List[int], colour: str, width: int = 4,
                   tag: str = "route") -> None:
        pos = self.positions
        coords: list = []
        for n in path:
            if n in pos:
                coords.extend(pos[n])
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=colour, width=width,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=tag)

    # ------------------------------------------------------------------ events
    def _on_resize(self, _event) -> None:
        self._reproject()
        self._draw_base_graph()
        self._recompute(silent=True)

    def _on_click(self, event) -> None:
        if not self.positions:
            return
        node = nearest_node(self.positions, event.x, event.y, max_dist=20)
        if node is None:
            return
        # first click sets source; second sets target; third restarts
        if self.source is None or (self.source is not None and self.target is not None):
            self.source, self.target = node, None
            self.canvas.delete("route")
            self._set_status(f"Start = intersection {node}. Now click a destination.")
        else:
            self.target = node
            self._recompute()
        self._draw_markers()

    # ------------------------------------------------------------------ actions
    def _recompute(self, silent: bool = False) -> None:
        self.canvas.delete("route")
        if self.source is None or self.target is None:
            return
        algo = self.algo.get()
        t0 = time.perf_counter()
        if algo == "A*":
            cost, path, expanded = astar(self.graph, self.source, self.target)
            extra = f", {expanded} nodes expanded"
        else:
            dist, prev = dijkstra(self.graph, self.source, target=self.target)
            cost = dist[self.target]
            path = reconstruct_path(prev, self.source, self.target)
            extra = ""
        elapsed = (time.perf_counter() - t0) * 1e3

        if not path or cost == float("inf"):
            self._set_status(f"No route between {self.source} and {self.target}.")
            return
        self._draw_path(path, ROUTE)
        self._draw_markers()
        self.info.config(text=f"{algo}: {format_distance(cost)} · {len(path)} hops")
        if not silent:
            self._set_status(f"{algo} route {self.source} -> {self.target}: "
                             f"{format_distance(cost)}, {len(path)} hops, "
                             f"{elapsed:.1f} ms{extra}.")

    def _show_alternatives(self) -> None:
        if self.source is None or self.target is None:
            self._set_status("Pick a start and destination first.")
            return
        self.canvas.delete("route")
        t0 = time.perf_counter()
        paths = k_shortest_paths(self.graph, self.source, self.target, 3)
        elapsed = (time.perf_counter() - t0) * 1e3
        if not paths:
            self._set_status("No route to offer alternatives for.")
            return
        # draw worst first so the best (red) ends up on top
        for i, (cost, path) in reversed(list(enumerate(paths))):
            self._draw_path(path, ALT_COLOURS[i % len(ALT_COLOURS)],
                            width=5 - i, tag="route")
        self._draw_markers()
        summary = "  ".join(f"#{i+1} {format_distance(c)}" for i, (c, _) in enumerate(paths))
        self._set_status(f"{len(paths)} alternative routes ({elapsed:.1f} ms):  {summary}")

    def _show_mst(self) -> None:
        self.canvas.delete("route")
        t0 = time.perf_counter()
        edges, total = prim_mst(self.graph)
        elapsed = (time.perf_counter() - t0) * 1e3
        pos = self.positions
        for u, v, _ in edges:
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                self.canvas.create_line(x1, y1, x2, y2, fill="#39d353", width=2,
                                        tags="route")
        self._set_status(f"Minimum spanning backbone: {len(edges)} roads, "
                         f"total {format_distance(total)} ({elapsed:.1f} ms).")

    def _clear(self) -> None:
        self.source = self.target = None
        self.canvas.delete("route")
        self.info.config(text="")
        self._draw_markers()
        self._set_status("Cleared. Click the map to choose a new start.")

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)


def launch(limit: int = 1500) -> None:
    """Load the graph and open the GUI window."""
    print(f"loading graph (limit {limit} nodes)...")
    graph, source_kind = load_city_graph(max_nodes=limit)
    root = tk.Tk()
    app = RouteEngineGUI(root, graph, source_kind)
    root.after(80, lambda: (app._reproject(), app._draw_base_graph()))
    root.mainloop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Route engine desktop GUI")
    parser.add_argument("--limit", type=int, default=1500,
                        help="max intersections to load (smaller = snappier)")
    args = parser.parse_args(argv)
    launch(args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
