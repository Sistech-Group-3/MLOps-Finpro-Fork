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
                                   (reuses formulas from utils.get_safest_route_v1 --
                                   static temporal aggregation, decaying spatial kernel)
3. rank_by_safety()            -> sort candidates by safety score alone
4. select_safest_then_fastest()-> take top-N safest, then pick fastest among them
   OR
   select_lexicographic()      -> soft version: safety dominates via a tiny
                                   epsilon weight on time, no hard cutoff
"""

import copy
from math import radians, sin, cos, asin, sqrt
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

_EARTH_RADIUS_M = 6_371_000.0


# -- 1. Diverse candidate generation -------------------------------------------

def generate_diverse_routes(
    G: nx.MultiDiGraph,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    n_routes: int = 20,
    penalty_factor: float = 1.7,
    weight: str = "length",
    snap: Optional[callable] = None,
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
    snap   : optional callable snap(lat, lon) -> node id, to avoid the
        repeated spatial-index build inside osmnx.nearest_nodes.

    Returns
    -------
    List of dicts: [{"nodes": [...], "coords": [(lat, lon), ...], "cost": float}, ...]
    Deduplicated — identical node sequences are only kept once, so you may
    get fewer than n_routes if the network doesn't support that many
    genuinely distinct paths.
    """
    if snap is not None:
        orig = snap(lat1, lon1)
        dest = snap(lat2, lon2)
    else:
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


# -- helpers: bridge OSMnx coords -> v1's (lat, lon, seg_len) point format ----

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lon points."""
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmbd = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmbd / 2) ** 2
    return 2 * _EARTH_RADIUS_M * asin(sqrt(a))


def _route_to_points(coords: List[tuple]) -> List[tuple]:
    """
    Convert a list of (lat, lon) route coords into the (lat, lon, seg_len)
    format utils.get_safest_route_v1.safest_route expects. seg_len for
    point i is the haversine distance from point i-1 to point i (0 for the
    first point) -- this is what route_scores() uses as the weight in its
    length-weighted mean, so it needs to reflect actual segment length,
    not just point count.
    """
    points = []
    for i, (lat, lon) in enumerate(coords):
        if i == 0:
            seg_len = 0.0
        else:
            prev_lat, prev_lon = coords[i - 1]
            seg_len = _haversine_m(prev_lat, prev_lon, lat, lon)
        points.append((lat, lon, seg_len))
    return points


# -- 2. Safety scoring ----------------------------------------------------------

def score_routes_for_safety(
    routes: List[Dict[str, Any]],
    crime_df: pd.DataFrame,
    t_query: pd.Timestamp,
    bw_space: float = 300.0,
) -> List[Dict[str, Any]]:
    """
    Score each candidate route for safety, reusing utils.get_safest_route_v1's
    formulas (static temporal aggregation + decaying spatial kernel).

    This step does NOT pick a winner -- that's rank_by_safety's / the
    select_* functions' job. alpha/beta are fixed to (1.0, 0.0) purely so
    we can reuse safest_route()'s per-route all_scores output cheaply in
    one call; its own combined_score/winner pick is discarded here in
    favor of returning R_route_mean / R_route_max directly per candidate.

    Returns
    -------
    The same list of route dicts passed in, each augmented with
    "R_route_mean", "R_route_max", and "point_risks".
    """
    if not routes:
        return []

    v1_routes = [_route_to_points(r["coords"]) for r in routes]

    result = safest_route(
        routes=v1_routes,
        crime_df=crime_df,
        t_query=t_query,
        bw_space=bw_space,
        alpha=1.0,
        beta=0.0,
    )

    scored = []
    for route, stats in zip(routes, result["all_scores"]):
        scored.append({
            **route,
            "R_route_mean": stats["R_route_mean"],
            "R_route_max": stats["R_route_max"],
            "point_risks": stats["point_risks"],
        })
    return scored


# -- 3. Ranking -------------------------------------------------------------------

def rank_by_safety(
    scored_routes: List[Dict[str, Any]],
    by: str = "R_route_mean",
) -> List[Dict[str, Any]]:
    """
    Sort candidates by safety score alone, safest first.

    by : "R_route_mean" (default -- overall danger along the route) or
         "R_route_max" (rank by worst single moment of danger instead).
    """
    if by not in ("R_route_mean", "R_route_max"):
        raise ValueError('by must be "R_route_mean" or "R_route_max"')
    return sorted(scored_routes, key=lambda r: r[by])


# -- 4. Final selection -------------------------------------------------------------

def select_safest_then_fastest(
    scored_routes: List[Dict[str, Any]],
    top_n: int = 3,
    by: str = "R_route_mean",
) -> Dict[str, Any]:
    """
    Hard-cutoff selection: take the top_n safest candidates, then pick the
    fastest ("cost") among just that safe pool. Time/distance can never
    override safety here -- it only decides among routes already judged
    acceptably safe.
    """
    if not scored_routes:
        raise ValueError("scored_routes must be non-empty")
    ranked = rank_by_safety(scored_routes, by=by)
    safest_pool = ranked[:max(1, top_n)]
    return min(safest_pool, key=lambda r: r["cost"])


def select_lexicographic(
    scored_routes: List[Dict[str, Any]],
    epsilon: float = 1e-4,
    by: str = "R_route_mean",
) -> Dict[str, Any]:
    """
    Soft version of safety-first selection: minimize (safety_score +
    epsilon * cost) instead of a hard top-N cutoff. As long as epsilon is
    small relative to the spread of `by` across candidates, cost only
    breaks ties between routes that are (near-)equally safe -- it can't
    outweigh a genuine difference in safety score. No candidates are
    discarded up front, unlike select_safest_then_fastest.
    """
    if not scored_routes:
        raise ValueError("scored_routes must be non-empty")
    return min(scored_routes, key=lambda r: r[by] + epsilon * r["cost"])