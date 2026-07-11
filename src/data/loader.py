"""Dataset loader: real Kathmandu road graph (OSM) with a synthetic fallback.

Hand-built conversion from OSMnx to our own ``Graph``. ``networkx``/``osmnx`` are
used ONLY to fetch and parse the OpenStreetMap data -- never to run the
algorithms. The downloaded graph is cached to ``data/`` so later runs work
offline and are reproducible.

If the download is unavailable (no network), ``load_city_graph`` falls back to a
clearly-labelled synthetic grid-plus-shortcuts city, so the whole pipeline still
runs end to end. Figures must state which source was used.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional, Tuple

from src.graph.graph import Graph

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEFAULT_PLACE = "Kathmandu, Nepal"


# --------------------------------------------------------------- OSM loading
def _cache_path(place: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in place.lower())
    return os.path.abspath(os.path.join(DATA_DIR, f"{slug}.graphml"))


def _osm_to_graph(nx_graph) -> Graph:
    """Convert an OSMnx MultiDiGraph to our undirected weighted Graph.

    Node ids are reindexed to a compact 0..V-1 range; edge weight is segment
    length in metres; (lat, lon) is preserved for map rendering and TSP.
    """
    g = Graph(directed=False)
    id_map = {osmid: i for i, osmid in enumerate(nx_graph.nodes())}
    for osmid, data in nx_graph.nodes(data=True):
        g.add_node(id_map[osmid], coord=(data.get("y"), data.get("x")))  # y=lat, x=lon
    seen = set()
    for u, v, data in nx_graph.edges(data=True):
        a, b = id_map[u], id_map[v]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        length = float(data.get("length", 1.0))
        g.add_edge(a, b, weight=length)
    return g


def load_city_graph(
    place: str = DEFAULT_PLACE,
    *,
    max_nodes: Optional[int] = None,
    use_cache: bool = True,
    allow_synthetic_fallback: bool = True,
    seed: int = 42,
) -> Tuple[Graph, str]:
    """Return (graph, source_label) where source_label is 'osm' or 'synthetic'.

    ``max_nodes`` trims to a connected subgraph of that size (for benchmarks).
    """
    os.makedirs(os.path.abspath(DATA_DIR), exist_ok=True)
    try:
        import osmnx as ox  # pulls in networkx as a dependency

        cache = _cache_path(place)
        if use_cache and os.path.exists(cache):
            nxg = ox.load_graphml(cache)
        else:
            nxg = ox.graph_from_place(place, network_type="drive")
            ox.save_graphml(nxg, cache)
        g = _osm_to_graph(nxg)
        g = largest_connected_component(g)
        if max_nodes is not None and g.num_nodes > max_nodes:
            g = connected_subgraph(g, max_nodes, seed=seed)
        return g, "osm"
    except Exception as exc:  # network down, package issue, etc.
        if not allow_synthetic_fallback:
            raise
        print(f"[loader] OSM unavailable ({type(exc).__name__}: {exc}); "
              f"using SYNTHETIC city graph.")
        n = max_nodes or 1000
        return synthetic_city_graph(n, seed=seed), "synthetic"


# ------------------------------------------------------- graph utilities
def largest_connected_component(graph: Graph) -> Graph:
    """Extract the biggest connected component (BFS over the undirected graph)."""
    if graph.num_nodes == 0:
        return graph
    unvisited = set(graph.nodes())
    best: list = []
    while unvisited:
        root = next(iter(unvisited))
        comp, q = [], deque([root])
        unvisited.discard(root)
        while q:
            u = q.popleft()
            comp.append(u)
            for v, _ in graph.neighbours(u):
                if v in unvisited:
                    unvisited.discard(v)
                    q.append(v)
        if len(comp) > len(best):
            best = comp
    return _induced_subgraph(graph, set(best))


def connected_subgraph(graph: Graph, n: int, *, seed: int = 42) -> Graph:
    """BFS outward from a random seed node until ``n`` nodes are collected."""
    if graph.num_nodes <= n:
        return graph
    rng = random.Random(seed)
    root = rng.choice(graph.nodes())
    chosen, q = {root}, deque([root])
    while q and len(chosen) < n:
        u = q.popleft()
        for v, _ in graph.neighbours(u):
            if v not in chosen:
                chosen.add(v)
                q.append(v)
                if len(chosen) >= n:
                    break
    return _induced_subgraph(graph, chosen)


def _induced_subgraph(graph: Graph, keep: set) -> Graph:
    """Subgraph induced on ``keep``, reindexed to a compact 0..k-1 range."""
    remap = {old: i for i, old in enumerate(sorted(keep))}
    sub = Graph(directed=graph.directed)
    for old, new in remap.items():
        sub.add_node(new, coord=graph.coords.get(old))
    for u, v, w in graph.edges():
        if u in keep and v in keep:
            sub.add_edge(remap[u], remap[v], weight=w)
    return sub


# ----------------------------------------------------------- synthetic data
def synthetic_city_graph(n: int, *, seed: int = 42, shortcut_prob: float = 0.08) -> Graph:
    """A grid road network plus random shortcuts -- sparse, like a real city.

    Nodes laid on a near-square lattice with Euclidean-ish edge weights; a few
    random long edges model arterial roads. Coordinates are synthetic but valid
    so map/TSP code still runs. Clearly labelled 'synthetic' by the loader.
    """
    rng = random.Random(seed)
    side = max(2, int(round(n ** 0.5)))
    g = Graph(directed=False)
    pos = {}
    node = 0
    for r in range(side):
        for c in range(side):
            if node >= n:
                break
            # fake lat/lon around Kathmandu centre so folium still renders
            lat = 27.70 + r * 0.001
            lon = 85.30 + c * 0.001
            g.add_node(node, coord=(lat, lon))
            pos[node] = (r, c)
            node += 1
    coord_to_id = {v: k for k, v in pos.items()}

    def w(a, b):
        (r1, c1), (r2, c2) = pos[a], pos[b]
        return (((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5) * 100 + rng.uniform(0, 5)

    for nd, (r, c) in pos.items():
        for dr, dc in ((1, 0), (0, 1)):
            nb = coord_to_id.get((r + dr, c + dc))
            if nb is not None:
                g.add_edge(nd, nb, weight=w(nd, nb))
    nodes = g.nodes()
    for _ in range(int(len(nodes) * shortcut_prob)):
        a, b = rng.choice(nodes), rng.choice(nodes)
        if a != b:
            g.add_edge(a, b, weight=w(a, b))
    return largest_connected_component(g)


def as_travel_time_graph(graph: Graph, default_kmph: float = 30.0) -> Graph:
    """Re-weight a length-weighted (metres) graph by travel time in seconds.

    travel_time = length_m / speed_m_per_s, with a single default driving speed
    (Kathmandu traffic averages ~20-30 km/h). This lets the same Dijkstra answer
    *fastest* routes rather than *shortest*, as the brief's dataset note allows.
    Coordinates are preserved so map/TSP code still works.
    """
    if default_kmph <= 0:
        raise ValueError("default speed must be positive")
    speed_mps = default_kmph * 1000.0 / 3600.0
    out = Graph(directed=graph.directed)
    for u in graph.nodes():
        out.add_node(u, coord=graph.coords.get(u))
    for u, v, length in graph.edges():
        out.add_edge(u, v, weight=length / speed_mps)
    return out


def random_dense_graph(n: int, *, density: float = 0.5, seed: int = 42,
                       max_weight: float = 100.0) -> Graph:
    """Synthetic dense graph for the list-vs-matrix / Dijkstra-density tests."""
    rng = random.Random(seed)
    g = Graph(directed=False)
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                g.add_edge(i, j, weight=rng.uniform(1, max_weight))
    return g
