import folium

def plot_routes_folium(
    routes: list[dict[str, any]],
    highlight_index: int = None,
    save_path: str = "routes_map.html",
) -> folium.Map:
    """
    Plot one or more routes on an interactive Folium map.
 
    Parameters
    ----------
    routes : list of route dicts, each with a 'coords' key:
             [{'coords': [(lat, lon), ...], 'label': 'route_0'}, ...]
    highlight_index : index of the route to draw in red/bold (e.g. the
             safest-selected one). All others are drawn thin and gray.
    save_path : where to save the resulting HTML file.
 
    Returns
    -------
    The folium.Map object (also saved to save_path as HTML).
    """
    if not routes:
        raise ValueError("routes list is empty")
 
    # Center the map on the midpoint of the first route
    first_coords = routes[0]["coords"]
    mid = first_coords[len(first_coords) // 2]
    m = folium.Map(location=mid, zoom_start=15, tiles="cartodbpositron")
 
    for i, route in enumerate(routes):
        is_highlighted = (i == highlight_index)
        folium.PolyLine(
            locations=route["coords"],
            color="red" if is_highlighted else "gray",
            weight=6 if is_highlighted else 3,
            opacity=0.9 if is_highlighted else 0.5,
            tooltip=route.get("label", f"route_{i}"),
        ).add_to(m)
 
    # Start/end markers, taken from the first route
    folium.Marker(
        first_coords[0], popup="Start",
        icon=folium.Icon(color="green"),
    ).add_to(m)
    folium.Marker(
        first_coords[-1], popup="End",
        icon=folium.Icon(color="red"),
    ).add_to(m)
 
    m.save(save_path)
    print(f"Saved to {save_path}")
    return m


def result_to_folium_routes(result: dict) -> list:
    """
    Convert the output of select_safest_route() into the list-of-dicts
    format plot_routes_folium() expects:
        [{'label': ..., 'coords': [(lat, lon), ...]}, ...]

    select_safest_route's routes are stored as (lat, lon, seg_len) tuples
    (from routes_converter/_route_dict_to_points) -- this strips seg_len
    and keeps just (lat, lon) for plotting, and labels the safest one.
    """
    safest_idx = result["safest_route_index"]

    routes_for_plot = []
    for s in result["all_scores"]:
        route_points = s["point_risks"]  # not used here, just for reference
        idx = s["route_index"]
        # Pull the actual (lat, lon, seg_len) points back out of result
        # NOTE: select_safest_route doesn't return the raw route coords in
        # all_scores by default -- if yours doesn't either, use the
        # `routes_converted` list you passed in instead (see usage below).
        coords = [(pt[0], pt[1]) for pt in s.get("route", [])]
        label = "safest" if idx == safest_idx else f"route_{idx}"
        routes_for_plot.append({"label": label, "coords": coords})

    return routes_for_plot, safest_idx


def visualize_safest_result(result: dict, routes_converted: list, save_path: str = "routes_map.html"):
    """
    Visualize the full candidate pool + highlighted safest route, using the
    ORIGINAL routes_converted list (list of list of (lat, lon, seg_len))
    that was passed into select_safest_route -- this is the reliable
    source of route geometry, since it's guaranteed to be in the same
    order as result['all_scores'].

    Parameters
    ----------
    result : output of select_safest_route()
    routes_converted : the exact list of (lat, lon, seg_len) routes you
                        passed into select_safest_route() as `routes`
    save_path : where to save the interactive HTML map
    """
    safest_idx = result["safest_route_index"]

    routes_for_plot = []
    for i, route_points in enumerate(routes_converted):
        coords = [(lat, lon) for lat, lon, _ in route_points]
        label = "safest" if i == safest_idx else f"route_{i}"
        routes_for_plot.append({"label": label, "coords": coords})

    return plot_routes_folium(routes_for_plot, highlight_index=safest_idx, save_path=save_path)
