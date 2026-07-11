"""Pure (Tk-free) geometry helpers for the GUI.

Kept separate from the widget code so the projection and hit-testing logic can be
unit-tested headlessly, without a display. The GUI in ``src.gui`` imports these.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

EARTH_R = 6_371_000.0  # metres

Coord = Tuple[float, float]            # (lat, lon)
Point = Tuple[float, float]            # (x, y) in canvas pixels


def project_nodes(coords: Dict[int, Coord], width: int, height: int,
                  padding: int = 24) -> Dict[int, Point]:
    """Project (lat, lon) nodes to canvas pixels, preserving aspect ratio.

    Uses an equirectangular projection to metres (so the city is not distorted),
    then fits the bounding box into the canvas with a uniform scale. North is up
    (canvas y is flipped). Nodes without coordinates are skipped.
    """
    pts = {n: c for n, c in coords.items()
           if c is not None and c[0] is not None and c[1] is not None}
    if not pts:
        return {}
    lat0 = sum(c[0] for c in pts.values()) / len(pts)
    cos0 = math.cos(math.radians(lat0))
    raw = {n: (EARTH_R * math.radians(lon) * cos0, EARTH_R * math.radians(lat))
           for n, (lat, lon) in pts.items()}
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span_x = (maxx - minx) or 1.0
    span_y = (maxy - miny) or 1.0
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
    # centre the drawing within the canvas
    off_x = (width - span_x * scale) / 2
    off_y = (height - span_y * scale) / 2
    out: Dict[int, Point] = {}
    for n, (x, y) in raw.items():
        cx = off_x + (x - minx) * scale
        cy = height - off_y - (y - miny) * scale   # flip: north up
        out[n] = (cx, cy)
    return out


def nearest_node(positions: Dict[int, Point], x: float, y: float,
                 max_dist: Optional[float] = None) -> Optional[int]:
    """Return the node whose canvas point is closest to (x, y).

    If ``max_dist`` is given, return None when nothing is within that radius.
    """
    best, best_d2 = None, float("inf")
    for n, (px, py) in positions.items():
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best_d2:
            best_d2, best = d2, n
    if best is None:
        return None
    if max_dist is not None and best_d2 > max_dist * max_dist:
        return None
    return best


def format_distance(metres: float) -> str:
    """Human-readable distance label."""
    if metres == float("inf"):
        return "unreachable"
    if metres >= 1000:
        return f"{metres / 1000:.2f} km"
    return f"{metres:.0f} m"
