import pandas as pd
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import LineString


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon points."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def _edge_linestring(G, u, v):
    """
    Return a shapely LineString following the ACTUAL edge geometry between
    two adjacent nodes u -> v, not just a straight line. OSMnx stores real
    curved-street geometry in the 'geometry' attribute when the underlying
    OSM way has intermediate shape points; falls back to a straight 2-point
    line only when no such geometry exists (i.e. the edge really is straight).
    """
    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        raise ValueError(f"No edge between nodes {u} and {v} in G")

    if G.is_multigraph():
        # Multiple parallel edges possible -- take the shortest one
        data = min(edge_data.values(), key=lambda d: d.get("length", float("inf")))
    else:
        data = edge_data

    if "geometry" in data:
        return data["geometry"], data.get("length")

    # No stored geometry -- edge is a straight line between the two nodes
    line = LineString([
        (G.nodes[u]["x"], G.nodes[u]["y"]),
        (G.nodes[v]["x"], G.nodes[v]["y"]),
    ])
    length = data.get("length")
    return line, length


def _route_dict_to_points(route: dict, G, densify_every_m: float = 50.0) -> list:
    """
    Convert a {'nodes': [...], 'coords': [...], 'cost': float} route dict
    into (lat, lon, seg_len) tuples that follow the REAL walk-network
    geometry -- not straight lines between intersections.

    For every consecutive node pair (u, v) in route['nodes'], this pulls the
    actual edge LineString from G (curved where the street curves), then
    samples points along it every `densify_every_m` meters. Only when the
    walk path actually passes near a lat/lon does it count toward the
    crime-record weighting in your risk-scoring pipeline -- straight-line
    shortcuts through blocks are eliminated.

    Parameters
    ----------
    route : dict with a 'nodes' key (list of graph node IDs) -- REQUIRED.
            If your route dicts only have 'coords' (no 'nodes'), you cannot
            recover real edge geometry and must fall back to the simpler
            straight-line version between those points.
    G : the OSMnx graph (loaded via joblib) the route was computed on.
    densify_every_m : max spacing between sampled points along each edge.
            Smaller = more precise (catches more nearby crime records) but
            more points to score. 30-50m is reasonable for a walk network.

    Returns
    -------
    List of (lat, lon, seg_len) tuples, seg_len = distance from the
    previous point along the actual walked path.
    """
    if "nodes" not in route:
        raise ValueError(
            "route dict has no 'nodes' key -- cannot recover real edge "
            "geometry. Re-run your routing step so it returns node IDs, "
            "not just coordinates."
        )

    node_path = route["nodes"]
    points = [(G.nodes[node_path[0]]["y"], G.nodes[node_path[0]]["x"], 0.0)]

    for u, v in zip(node_path[:-1], node_path[1:]):
        line, length = _edge_linestring(G, u, v)
        if length is None:
            # Fallback: compute length from the geometry itself
            coords = list(line.coords)
            length = sum(
                _haversine_m(coords[i][1], coords[i][0], coords[i + 1][1], coords[i + 1][0])
                for i in range(len(coords) - 1)
            )

        n_segments = max(1, int(length // densify_every_m))
        seg_len = length / n_segments

        # Skip fraction 0.0 -- that's the node we already added
        for i in range(1, n_segments + 1):
            frac = i / n_segments
            pt = line.interpolate(frac, normalized=True)
            points.append((pt.y, pt.x, seg_len))  # (lat, lon, seg_len)

    return points


def routes_converter(routes: list, G, densify_every_m: float = 50.0) -> list:
    """
    Convert a list of route dicts (each with 'nodes', 'coords', 'cost')
    into the (lat, lon, seg_len) format select_safest_route expects,
    following real walk-network geometry instead of straight lines.

    Parameters
    ----------
    routes : list of route dicts from generate_diverse_routes / get_k_shortest_paths
    G : the OSMnx graph the routes were computed on (load via joblib)
    densify_every_m : point spacing along each edge (see _route_dict_to_points)
    """
    routes_converted = []
    for i, route in enumerate(routes):
        pts = set((round(lat, 5), round(lon, 5)) for lat, lon in route["coords"])
        print(f"Route {i}: {len(pts)} unique intersection points, "
              f"cost={route['cost']:.1f}m")
        converted = _route_dict_to_points(route, G, densify_every_m)
        print(f"  -> densified to {len(converted)} points "
              f"(every ~{densify_every_m}m along the actual walk path)")
        routes_converted.append(converted)

    return routes_converted