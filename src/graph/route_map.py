"""Render a computed shortest path on a real (folium) map -- Task 2 visual.

Library use here is pure visualisation: folium draws the route our own Dijkstra
computed. Produces an HTML map into figures/ that the report screenshots.
"""

from __future__ import annotations

import os
from typing import List

from src.bench.plots import ensure_fig_dir
from src.graph.graph import Graph


def render_route(graph: Graph, path: List[int], filename: str = "task2_route_map.html",
                 zoom: int = 14) -> str:
    """Draw ``path`` (a list of node ids) over an OpenStreetMap tile layer."""
    import folium

    coords = [graph.coords[n] for n in path if n in graph.coords and graph.coords[n][0] is not None]
    if not coords:
        raise ValueError("path has no usable coordinates to plot")

    centre = coords[len(coords) // 2]
    fmap = folium.Map(location=list(centre), zoom_start=zoom, tiles="OpenStreetMap")
    folium.PolyLine(coords, weight=5, opacity=0.8, color="crimson",
                    tooltip="shortest path (our Dijkstra)").add_to(fmap)
    folium.Marker(list(coords[0]), tooltip="source",
                  icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(list(coords[-1]), tooltip="target",
                  icon=folium.Icon(color="red")).add_to(fmap)

    out = os.path.join(ensure_fig_dir(), filename)
    fmap.save(out)
    print(f"  saved {out}")
    return out
