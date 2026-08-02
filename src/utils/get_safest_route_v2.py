"""
Safety-first route selection pipeline.

Priority order: SAFETY dominates, TIME/DISTANCE only breaks ties among
already-safe candidates. This requires generating a diverse candidate pool
FIRST (not just K-shortest-by-distance), since a safety-first ranking is
only as good as the candidates it's allowed to choose from.

Pipeline
--------
1. generate_diverse_routes()   -> wide, geometrically distinct candidate pool
2. score_routes_for_safety()   -> R_route_mean / R_route_max per candidate
                                   (reuses formulas from utils.risk_scoring)
3. rank_by_safety()            -> sort candidates by safety score alone
4. select_safest_then_fastest()-> take top-N safest, then pick fastest among them
   OR
   select_lexicographic()      -> soft version: safety dominates via a tiny
                                   epsilon weight on time, no hard cutoff
"""

import copy
from typing import List, Dict, Any, Optional

import networkx as nx
import osmnx as ox
import pandas as pd

import os 
import sys

cwd = os.getcwd()
candidates = [
    cwd,
    os.path.join(cwd, "src"),
    os.path.join(cwd, ".."),
    os.path.join(cwd, "..", "src"),
]
for candidate in candidates:
    candidate = os.path.abspath(candidate)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from utils.get_safest_route_v1 import safest_route


def generate_diverse_routes(
    G: nx.MultiDiGraph,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n_routes: int = 20,
    penalty_factor: float = 1.7,
    weight: str = "length",
) -> List[Dict[str, Any]]:
    """
    Generate a diverse set of candidate routes between two points using
    penalized iterative re-routing: after finding the shortest path, the
    edges it used are penalized (weight multiplied by `penalty_factor`) in
    a working copy of the graph, then the shortest path is recomputed.
    Repeating this forces each new route to genuinely avoid streets already
    used by previous routes, instead of returning near-identical variants
    that only differ by one block (as plain K-shortest-paths tends to do).

    Parameters
    ----------
    G : networkx MultiDiGraph (from OSMnx)
    lat1, lon1, lat2, lon2 : start/end coordinates
    n_routes : how many diverse candidates to attempt to generate
    penalty_factor : multiplier applied to an edge's weight each time it's
        used by a previously found route (>1.0; higher = more aggressive
        avoidance of previously-used streets)
    weight : edge attribute to route on (e.g. "length" or "travel_time")

    Returns
    -------
    List of dicts: [{"nodes": [...], "coords": [(lat, lon), ...], "cost": float}, ...]
    Deduplicated — identical node sequences are only kept once, so you may
    get fewer than n_routes if the network doesn't support that many
    genuinely distinct paths.
    """
    orig = ox.distance.nearest_nodes(G, X=lon1, Y=lat1)
    dest = ox.distance.nearest_nodes(G, X=lon2, Y=lat2)

    # Working copy — we mutate edge weights here, never on the original graph
    G_work = copy.deepcopy(G)

    routes = []
    seen_node_sets = set()

    for _ in range(n_routes):
        try:
            node_path = nx.shortest_path(G_work, orig, dest, weight=weight)
        except nx.NetworkXNoPath:
            break

        key = tuple(node_path)
        if key in seen_node_sets:
            # Penalize harder and try once more; if still duplicate, stop
            _penalize_path(G_work, node_path, weight, penalty_factor)
            try:
                node_path = nx.shortest_path(G_work, orig, dest, weight=weight)
            except nx.NetworkXNoPath:
                break
            key = tuple(node_path)
            if key in seen_node_sets:
                break

        seen_node_sets.add(key)

        cost = nx.path_weight(G, node_path, weight=weight)  # cost from ORIGINAL graph
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_path]
        routes.append({"nodes": node_path, "coords": coords, "cost": cost})

        # Penalize this path's edges in the working graph so the next
        # iteration is pushed toward a genuinely different route
        _penalize_path(G_work, node_path, weight, penalty_factor)

    return routes


def _penalize_path(G_work, node_path: list, weight: str, factor: float):
    """
    Multiply the weight of every edge used in node_path, in place.

    Handles both graph types:
      - MultiDiGraph/MultiGraph: get_edge_data(u, v) -> {edge_key: {attr: val, ...}, ...}
      - DiGraph/Graph:           get_edge_data(u, v) -> {attr: val, ...}  (no edge_key layer)

    OSMnx graphs are MultiDiGraph by default, but if G was converted (e.g.
    via nx.DiGraph(G) or ox.utils_graph.get_digraph) it becomes a plain
    DiGraph, which changes the shape of get_edge_data()'s return value.
    """
    is_multigraph = G_work.is_multigraph()
    for u, v in zip(node_path[:-1], node_path[1:]):
        edge_data = G_work.get_edge_data(u, v)
        if edge_data is None:
            continue
        if is_multigraph:
            for k in edge_data:
                edge_data[k][weight] = edge_data[k].get(weight, 1) * factor
        else:
            edge_data[weight] = edge_data.get(weight, 1) * factor