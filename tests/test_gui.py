"""Tests for the GUI: pure geometry helpers plus a guarded widget smoke test."""

import pytest

from src.data.loader import synthetic_city_graph
from src.gui_support import format_distance, nearest_node, project_nodes


# ----------------------------------------------------------- pure geometry
def test_project_nodes_within_canvas():
    coords = {0: (27.70, 85.30), 1: (27.71, 85.31), 2: (27.69, 85.32)}
    pos = project_nodes(coords, 800, 600, padding=20)
    assert set(pos) == {0, 1, 2}
    for x, y in pos.values():
        assert 0 <= x <= 800 and 0 <= y <= 600


def test_project_nodes_north_is_up():
    coords = {0: (27.70, 85.30), 1: (27.72, 85.30)}   # node 1 is further north
    pos = project_nodes(coords, 400, 400)
    assert pos[1][1] < pos[0][1]                       # smaller y = higher on screen


def test_project_skips_missing_coords():
    coords = {0: (27.7, 85.3), 1: (None, None), 2: None}
    pos = project_nodes(coords, 400, 400)
    assert set(pos) == {0}


def test_nearest_node_picks_closest():
    pos = {0: (10.0, 10.0), 1: (100.0, 100.0), 2: (50.0, 50.0)}
    assert nearest_node(pos, 12, 11) == 0
    assert nearest_node(pos, 95, 98) == 1
    assert nearest_node(pos, 200, 200, max_dist=20) is None   # too far
    assert nearest_node({}, 0, 0) is None


def test_format_distance():
    assert format_distance(450) == "450 m"
    assert format_distance(1500) == "1.50 km"
    assert format_distance(float("inf")) == "unreachable"


# --------------------------------------------------- widget smoke test (guarded)
def _make_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tkinter")
    root.withdraw()        # don't actually show the window
    return root


def test_gui_builds_and_computes_route():
    root = _make_root()
    try:
        from src.gui import RouteEngineGUI
        g = synthetic_city_graph(120, seed=1)
        app = RouteEngineGUI(root, g, "synthetic")
        app._reproject()
        app._draw_base_graph()
        assert app.positions                          # nodes projected
        app.source, app.target = g.nodes()[0], g.nodes()[-1]
        app._recompute()
        # a route should have been drawn (canvas items tagged 'route')
        assert app.canvas.find_withtag("route")
        assert "route" in app.status.cget("text").lower() or \
               "hops" in app.info.cget("text").lower()
    finally:
        root.destroy()


def test_gui_mst_overlay():
    root = _make_root()
    try:
        from src.gui import RouteEngineGUI
        g = synthetic_city_graph(100, seed=2)
        app = RouteEngineGUI(root, g, "synthetic")
        app._reproject()
        app._draw_base_graph()
        app._show_mst()
        assert app.canvas.find_withtag("route")       # MST edges drawn
        assert "backbone" in app.status.cget("text").lower()
    finally:
        root.destroy()
