"""Tkinter desktop GUI for the route engine, with one tab per coursework task.

Tabs:
  1. Data structures  -- build a BST and an AVL from the same keys and draw both,
     so the BST visibly degrades on sorted input while the AVL stays balanced.
  2. Routing          -- the Kathmandu map: click two junctions to route with
     Dijkstra or A*, list alternative routes, or draw the spanning tree.
  3. Strategies       -- a 0/1 knapsack solver (dynamic programming) and a graph
     colouring demo (backtracking / DSATUR) drawn on a sample of the city.
  4. TSP              -- pick stops and compare a nearest-neighbour tour with a
     simulated-annealing tour, both drawn on the map.
  5. Concurrency      -- time the sequential and threaded versions live (showing
     the GIL), alongside the measured multiprocessing speed-up.

Tkinter ships with Python, so there is no extra dependency.
Run:  python -m src.gui   (or  python -m src.gui --limit 1500)
"""

from __future__ import annotations

import argparse
import os
import random
import threading
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
from src.structures.avl import AVLTree
from src.structures.bst import BinarySearchTree

BG = "#0f1419"
PANEL = "#161b22"
EDGE = "#2b3640"
NODE = "#5a6b78"
ROUTE = "#e23c43"
SOURCE = "#39d353"
TARGET = "#f5a623"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
ALT_COLOURS = ["#e23c43", "#f5a623", "#3b82f6", "#a855f7", "#14b8a6"]
PALETTE = ["#e23c43", "#f5a623", "#3b82f6", "#39d353", "#a855f7",
           "#14b8a6", "#ec4899", "#eab308", "#f97316", "#22d3ee"]


# ----------------------------------------------------------- tree layout helper
def tree_positions(root):
    """In-order column / depth layout for a binary tree. {node: (col, depth)}."""
    pos: Dict[object, tuple] = {}
    counter = [0]

    def walk(node, depth):
        if node is None:
            return
        walk(node.left, depth + 1)
        pos[id(node)] = (counter[0], depth)
        counter[0] += 1
        walk(node.right, depth + 1)

    walk(root, 0)
    return pos, counter[0]


# =============================================================== Task 2: routing
class RouteEngineGUI:
    """The routing map (Task 2). Parent may be the root window or a tab frame."""

    def __init__(self, parent, graph, source_kind: str) -> None:
        self.root = parent
        self.graph = graph
        self.source_kind = source_kind
        self.positions: Dict[int, tuple] = {}
        self.source: Optional[int] = None
        self.target: Optional[int] = None

        self._build_controls()
        self._build_canvas()
        self._build_statusbar()
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)
        self._set_status(f"Loaded {graph.num_nodes} junctions and {graph.num_edges} "
                         f"roads ({source_kind} data). Click a start, then a destination.")

    def _build_controls(self) -> None:
        bar = tk.Frame(self.root, bg=PANEL, padx=10, pady=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar, text="Algorithm:", bg=PANEL, fg=TEXT).pack(side=tk.LEFT)
        self.algo = tk.StringVar(value="Dijkstra")
        for name in ("Dijkstra", "A*"):
            ttk.Radiobutton(bar, text=name, value=name, variable=self.algo,
                            command=self._recompute).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Find route", command=self._recompute).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(bar, text="3 alternatives", command=self._show_alternatives).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Spanning tree", command=self._show_mst).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=4)
        self.info = tk.Label(bar, text="", bg=PANEL, fg=MUTED)
        self.info.pack(side=tk.RIGHT)

    def _build_canvas(self) -> None:
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_statusbar(self) -> None:
        self.status = tk.Label(self.root, text="", bg=PANEL, fg=TEXT, anchor="w", padx=10, pady=4)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _reproject(self) -> None:
        w = self.canvas.winfo_width() or 1000
        h = self.canvas.winfo_height() or 600
        self.positions = project_nodes(self.graph.coords, w, h)

    def _draw_base_graph(self) -> None:
        self.canvas.delete("all")
        pos = self.positions
        for u, v, _ in self.graph.edges():
            if u in pos and v in pos:
                x1, y1 = pos[u]; x2, y2 = pos[v]
                self.canvas.create_line(x1, y1, x2, y2, fill=EDGE, width=1)
        for n, (x, y) in pos.items():
            self.canvas.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill=NODE, outline="")
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

    def _draw_path(self, path, colour, width=4, tag="route") -> None:
        coords = []
        for n in path:
            if n in self.positions:
                coords.extend(self.positions[n])
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=colour, width=width,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=tag)

    def _on_resize(self, _e) -> None:
        self._reproject(); self._draw_base_graph(); self._recompute(silent=True)

    def _on_click(self, event) -> None:
        if not self.positions:
            return
        node = nearest_node(self.positions, event.x, event.y, max_dist=20)
        if node is None:
            return
        if self.source is None or (self.source is not None and self.target is not None):
            self.source, self.target = node, None
            self.canvas.delete("route")
            self._set_status(f"Start = junction {node}. Now click a destination.")
        else:
            self.target = node
            self._recompute()
        self._draw_markers()

    def _recompute(self, silent=False) -> None:
        self.canvas.delete("route")
        if self.source is None or self.target is None:
            return
        algo = self.algo.get()
        t0 = time.perf_counter()
        if algo == "A*":
            cost, path, expanded = astar(self.graph, self.source, self.target)
            extra = f", {expanded} junctions explored"
        else:
            dist, prev = dijkstra(self.graph, self.source, target=self.target)
            cost = dist[self.target]; path = reconstruct_path(prev, self.source, self.target); extra = ""
        ms = (time.perf_counter() - t0) * 1e3
        if not path or cost == float("inf"):
            self._set_status(f"No route between {self.source} and {self.target}."); return
        self._draw_path(path, ROUTE); self._draw_markers()
        self.info.config(text=f"{algo}: {format_distance(cost)} - {len(path)} hops")
        if not silent:
            self._set_status(f"{algo} route {self.source} to {self.target}: "
                             f"{format_distance(cost)}, {len(path)} hops, {ms:.1f} ms{extra}.")

    def _show_alternatives(self) -> None:
        if self.source is None or self.target is None:
            self._set_status("Pick a start and destination first."); return
        self.canvas.delete("route")
        t0 = time.perf_counter()
        paths = k_shortest_paths(self.graph, self.source, self.target, 3)
        ms = (time.perf_counter() - t0) * 1e3
        if not paths:
            self._set_status("No route to offer alternatives for."); return
        for i, (cost, path) in reversed(list(enumerate(paths))):
            self._draw_path(path, ALT_COLOURS[i % len(ALT_COLOURS)], width=5 - i)
        self._draw_markers()
        summary = "   ".join(f"#{i+1} {format_distance(c)}" for i, (c, _) in enumerate(paths))
        self._set_status(f"{len(paths)} alternative routes ({ms:.1f} ms):   {summary}")

    def _show_mst(self) -> None:
        self.canvas.delete("route")
        t0 = time.perf_counter()
        edges, total = prim_mst(self.graph)
        ms = (time.perf_counter() - t0) * 1e3
        for u, v, _ in edges:
            if u in self.positions and v in self.positions:
                x1, y1 = self.positions[u]; x2, y2 = self.positions[v]
                self.canvas.create_line(x1, y1, x2, y2, fill=SOURCE, width=2, tags="route")
        self._set_status(f"Minimum spanning backbone: {len(edges)} roads, "
                         f"total {format_distance(total)} ({ms:.1f} ms).")

    def _clear(self) -> None:
        self.source = self.target = None
        self.canvas.delete("route"); self.info.config(text="")
        self._draw_markers(); self._set_status("Cleared. Click the map to choose a new start.")

    def _set_status(self, text) -> None:
        self.status.config(text=text)


# ======================================================= Task 1: data structures
class StructuresTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        bar = tk.Frame(self, bg=PANEL, padx=10, pady=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar, text="Keys:", bg=PANEL, fg=TEXT).pack(side=tk.LEFT)
        self.entry = tk.Entry(bar, width=46)
        self.entry.insert(0, "50,30,70,20,40,60,80,10,25,35,45")
        self.entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Build", command=self.build).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Random 15", command=lambda: self.fill(False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Sorted 15", command=lambda: self.fill(True)).pack(side=tk.LEFT, padx=4)
        self.note = tk.Label(self, text="The BST and AVL are built from the same keys. "
                             "Try the Sorted button to see the BST collapse while the AVL stays balanced.",
                             bg=BG, fg=MUTED, anchor="w", padx=10, pady=6)
        self.note.pack(side=tk.TOP, fill=tk.X)
        wrap = tk.Frame(self, bg=BG); wrap.pack(fill=tk.BOTH, expand=True)
        self.bst_canvas = self._titled(wrap, "Binary Search Tree")
        self.avl_canvas = self._titled(wrap, "AVL Tree (self-balancing)")
        self.after(120, self.build)

    def _titled(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._label = tk.Label(f, text=title, bg=PANEL, fg=TEXT, pady=4)
        self._label.pack(side=tk.TOP, fill=tk.X)
        c = tk.Canvas(f, bg=BG, highlightthickness=0)
        c.pack(fill=tk.BOTH, expand=True)
        c.title_label = self._label
        c.base_title = title
        return c

    def fill(self, ordered):
        rng = random.Random()
        keys = sorted(rng.sample(range(10, 99), 15))
        if not ordered:
            rng.shuffle(keys)
        self.entry.delete(0, tk.END); self.entry.insert(0, ",".join(map(str, keys)))
        self.build()

    def build(self):
        try:
            keys = [int(x) for x in self.entry.get().replace(" ", "").split(",") if x != ""]
        except ValueError:
            self.note.config(text="Please enter whole numbers separated by commas."); return
        keys = keys[:31]
        bst, avl = BinarySearchTree(), AVLTree()
        for k in keys:
            bst.insert(k); avl.insert(k)
        self._draw(self.bst_canvas, bst.root, bst.height())
        self._draw(self.avl_canvas, avl.root, avl.height())

    def _draw(self, canvas, root, height):
        canvas.delete("all")
        canvas.title_label.config(text=f"{canvas.base_title}   (height = {height})")
        if root is None:
            return
        pos, n = tree_positions(root)
        w = canvas.winfo_width() or 500
        h = canvas.winfo_height() or 500
        margin, r = 28, 15
        max_depth = max((d for _, d in pos.values()), default=0)
        col_w = (w - 2 * margin) / max(n - 1, 1)
        row_h = (h - 2 * margin) / max(max_depth, 1) if max_depth else 0
        xy = {nid: (margin + col * col_w, margin + depth * row_h) for nid, (col, depth) in pos.items()}

        def edges(node):
            if node is None:
                return
            for child in (node.left, node.right):
                if child is not None:
                    x1, y1 = xy[id(node)]; x2, y2 = xy[id(child)]
                    canvas.create_line(x1, y1, x2, y2, fill=EDGE, width=2)
                    edges(child)
        edges(root)

        def nodes(node):
            if node is None:
                return
            x, y = xy[id(node)]
            canvas.create_oval(x - r, y - r, x + r, y + r, fill="#1f6feb", outline="white", width=2)
            canvas.create_text(x, y, text=str(node.key), fill="white", font=("Helvetica", 9, "bold"))
            nodes(node.left); nodes(node.right)
        nodes(root)


# ============================================================ Task 3: strategies
class StrategiesTab(ttk.Frame):
    def __init__(self, parent, graph):
        super().__init__(parent)
        self.graph = graph
        # --- knapsack panel ---
        kp = tk.Frame(self, bg=PANEL, padx=10, pady=8); kp.pack(side=tk.TOP, fill=tk.X)
        tk.Label(kp, text="Knapsack (DP)  items weight:value", bg=PANEL, fg=TEXT).pack(side=tk.LEFT)
        self.items = tk.Entry(kp, width=34); self.items.insert(0, "1:1, 3:4, 4:5, 5:7")
        self.items.pack(side=tk.LEFT, padx=6)
        tk.Label(kp, text="capacity", bg=PANEL, fg=TEXT).pack(side=tk.LEFT)
        self.cap = tk.Entry(kp, width=5); self.cap.insert(0, "7"); self.cap.pack(side=tk.LEFT, padx=4)
        ttk.Button(kp, text="Solve", command=self.solve_knapsack).pack(side=tk.LEFT, padx=4)
        self.kresult = tk.Label(self, text="", bg=BG, fg="#39d353", anchor="w", padx=10, pady=6)
        self.kresult.pack(side=tk.TOP, fill=tk.X)
        # --- colouring panel ---
        cp = tk.Frame(self, bg=PANEL, padx=10, pady=8); cp.pack(side=tk.TOP, fill=tk.X)
        tk.Label(cp, text="Graph colouring (signal phases)", bg=PANEL, fg=TEXT).pack(side=tk.LEFT)
        ttk.Button(cp, text="Colour the junctions", command=self.colour).pack(side=tk.LEFT, padx=6)
        self.cresult = tk.Label(cp, text="", bg=PANEL, fg=MUTED); self.cresult.pack(side=tk.LEFT, padx=8)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.after(120, self.solve_knapsack)

    def solve_knapsack(self):
        from src.strategies.knapsack import knapsack_full_table
        try:
            ws, vs = [], []
            for part in self.items.get().split(","):
                w, v = part.strip().split(":"); ws.append(int(w)); vs.append(float(v))
            cap = int(self.cap.get())
            best, chosen, _ = knapsack_full_table(ws, vs, cap)
            picked = ", ".join(f"({ws[i]}kg, value {vs[i]:g})" for i in chosen)
            self.kresult.config(text=f"Best value = {best:g}   using items: {picked or 'none'} "
                                f"(total weight {sum(ws[i] for i in chosen)} <= {cap})")
        except Exception:
            self.kresult.config(text="Enter items as weight:value pairs, e.g. 1:1, 3:4, 4:5")

    def colour(self):
        from src.data.loader import connected_subgraph
        from src.strategies.dsatur import dsatur_colouring
        sub = connected_subgraph(self.graph, min(self.graph.num_nodes, 220))
        colours, k = dsatur_colouring(sub)
        ok = all(colours[u] != colours[v] for u, v, _ in sub.edges())
        self.cresult.config(text=f"{k} colours used on {sub.num_nodes} junctions; "
                            f"every neighbour differs: {ok}")
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 800; h = self.canvas.winfo_height() or 400
        pos = project_nodes(sub.coords, w, h)
        for u, v, _ in sub.edges():
            if u in pos and v in pos:
                x1, y1 = pos[u]; x2, y2 = pos[v]
                self.canvas.create_line(x1, y1, x2, y2, fill=EDGE, width=1)
        for n, (x, y) in pos.items():
            col = PALETTE[colours.get(n, 0) % len(PALETTE)]
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=col, outline="")


# =================================================================== Task 4: TSP
class TspTab(ttk.Frame):
    def __init__(self, parent, graph):
        super().__init__(parent)
        self.graph = graph
        bar = tk.Frame(self, bg=PANEL, padx=10, pady=8); bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar, text="Stops:", bg=PANEL, fg=TEXT).pack(side=tk.LEFT)
        self.n = tk.IntVar(value=18)
        tk.Spinbox(bar, from_=5, to=40, textvariable=self.n, width=5).pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Plan delivery tour", command=self.plan).pack(side=tk.LEFT, padx=4)
        tk.Label(bar, text="  red = nearest-neighbour, green = simulated annealing",
                 bg=PANEL, fg=MUTED).pack(side=tk.LEFT)
        self.result = tk.Label(self, text="", bg=BG, fg=TEXT, anchor="w", padx=10, pady=6)
        self.result.pack(side=tk.TOP, fill=tk.X)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True)

    def plan(self):
        from src.heuristics.nearest_neighbour import nearest_neighbour
        from src.heuristics.simulated_annealing import simulated_annealing
        from src.heuristics.tsp import TSPInstance
        w = self.canvas.winfo_width() or 900; h = self.canvas.winfo_height() or 500
        allpos = project_nodes(self.graph.coords, w, h)
        usable = [n for n in self.graph.nodes() if n in allpos]
        if len(usable) < self.n.get():
            self.result.config(text="Not enough junctions with coordinates."); return
        chosen = random.sample(usable, self.n.get())
        pts = [allpos[n] for n in chosen]
        inst = TSPInstance(pts)
        nn_tour, nn_len = nearest_neighbour(inst)
        sa = simulated_annealing(inst, iterations=6000)
        self.canvas.delete("all")
        for n, (x, y) in allpos.items():
            self.canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill=EDGE, outline="")
        self._draw_tour(pts, nn_tour, ROUTE, 2)
        self._draw_tour(pts, sa.tour, SOURCE, 3)
        for x, y in pts:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="white", outline="")
        gap = (nn_len - sa.length) / sa.length * 100 if sa.length else 0
        self.result.config(text=f"{self.n.get()} stops.  Nearest-neighbour tour = {nn_len:.0f}px,  "
                           f"simulated annealing = {sa.length:.0f}px  "
                           f"({gap:.1f}% shorter).")

    def _draw_tour(self, pts, tour, colour, width):
        coords = []
        for i in tour:
            coords.extend(pts[i])
        coords.extend(pts[tour[0]])  # close the loop
        self.canvas.create_line(*coords, fill=colour, width=width)


# =========================================================== Task 5: concurrency
class ConcurrencyTab(ttk.Frame):
    def __init__(self, parent, graph):
        super().__init__(parent)
        self.graph = graph
        bar = tk.Frame(self, bg=PANEL, padx=10, pady=8); bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Run live test (sequential vs threads)", command=self.run).pack(side=tk.LEFT)
        tk.Label(bar, text="  threads cannot speed up CPU-bound Python work (the GIL).",
                 bg=PANEL, fg=MUTED).pack(side=tk.LEFT, padx=8)
        self.result = tk.Label(self, text="Press the button to time the same Dijkstra work "
                               "run sequentially and across 4 threads.", bg=BG, fg=TEXT,
                               anchor="w", justify="left", padx=12, pady=10)
        self.result.pack(side=tk.TOP, fill=tk.X)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, height=180)
        self.canvas.pack(side=tk.TOP, fill=tk.X)
        # measured multiprocessing chart from the report (if present)
        self._img = None
        path = os.path.join(os.path.dirname(__file__), "..", "figures", "task5_speedup.png")
        path = os.path.abspath(path)
        holder = tk.Frame(self, bg=BG); holder.pack(fill=tk.BOTH, expand=True)
        if os.path.exists(path):
            try:
                self._img = tk.PhotoImage(file=path)
                # subsample to fit
                factor = max(1, self._img.width() // 700)
                self._img = self._img.subsample(factor, factor)
                tk.Label(holder, text="Measured speed-up including multiprocessing "
                         "(reaches about 4.4x on 8 cores):", bg=BG, fg=MUTED).pack(anchor="w", padx=12)
                tk.Label(holder, image=self._img, bg=BG).pack(anchor="w", padx=12, pady=6)
            except Exception:
                pass

    def run(self):
        self.result.config(text="Running...")
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        from src.concurrency.parallel_shortest_paths import (
            sequential_multi_source, threaded_multi_source, results_equal)
        from src.data.loader import connected_subgraph
        sub = connected_subgraph(self.graph, min(self.graph.num_nodes, 1500))
        sources = sub.nodes()[:40]
        t0 = time.perf_counter(); seq = sequential_multi_source(sub, sources, reduce="farness")
        t_seq = time.perf_counter() - t0
        t0 = time.perf_counter(); thr = threaded_multi_source(sub, sources, n_workers=4, reduce="farness")
        t_thr = time.perf_counter() - t0
        same = results_equal(seq, thr)
        speed = t_seq / t_thr if t_thr else 0
        msg = (f"Workload: Dijkstra from {len(sources)} sources on {sub.num_nodes} junctions.\n"
               f"Sequential: {t_seq*1e3:.0f} ms        4 threads: {t_thr*1e3:.0f} ms "
               f"(speed-up {speed:.2f}x)\n"
               f"Threaded answers identical to sequential: {same}  ->  no race conditions.\n"
               f"The threads give almost no speed-up because of the GIL; real parallelism "
               f"needs separate processes (see the measured chart below).")
        self.after(0, lambda: self._show(msg, t_seq, t_thr))

    def _show(self, msg, t_seq, t_thr):
        self.result.config(text=msg)
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 800
        base = max(t_seq, t_thr, 1e-9)
        for i, (label, t) in enumerate([("sequential", t_seq), ("4 threads", t_thr)]):
            y = 40 + i * 60
            length = int((t / base) * (w - 220))
            self.canvas.create_rectangle(160, y, 160 + length, y + 34, fill="#1f6feb", outline="")
            self.canvas.create_text(150, y + 17, text=label, fill=TEXT, anchor="e")
            self.canvas.create_text(170 + length, y + 17, text=f"{t*1e3:.0f} ms", fill=TEXT, anchor="w")


# ===================================================================== launch
def launch(limit: int = 1500) -> None:
    print(f"loading graph (limit {limit} junctions)...")
    graph, kind = load_city_graph(max_nodes=limit)
    root = tk.Tk()
    root.title("Kathmandu Route Engine")
    root.configure(bg=BG)
    root.geometry("1180x760")
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True)

    routing_frame = ttk.Frame(nb)
    routing = RouteEngineGUI(routing_frame, graph, kind)
    nb.add(StructuresTab(nb), text="1. Data structures")
    nb.add(routing_frame, text="2. Routing")
    nb.add(StrategiesTab(nb, graph), text="3. Strategies")
    nb.add(TspTab(nb, graph), text="4. TSP tour")
    nb.add(ConcurrencyTab(nb, graph), text="5. Concurrency")
    nb.select(1)  # open on the map

    root.after(80, lambda: (routing._reproject(), routing._draw_base_graph()))
    root.mainloop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Route engine desktop GUI")
    parser.add_argument("--limit", type=int, default=1500,
                        help="max junctions to load (smaller is snappier)")
    args = parser.parse_args(argv)
    launch(args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
