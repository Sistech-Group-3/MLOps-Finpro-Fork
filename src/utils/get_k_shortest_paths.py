import osmnx as ox
import networkx as nx
from itertools import islice

def get_k_shortest_paths(G: nx.digraph, lat1, lon1, lat2, lon2, k=3, weight="length", snap=None):
	"""
	Find the k shortest simple paths between two lat/lon points on graph G,
	ranked from shortest to longest by total edge weight.

	Parameters:
		G      : networkx MultiDiGraph (from OSMnx)
		lat1, lon1 : coordinates of start point
		lat2, lon2 : coordinates of end point
		k      : number of shortest paths to return
		weight : edge attribute to optimize on (e.g. "length" or "travel_time")
		snap   : optional callable snap(lat, lon) -> node id. When provided it
		         is used to map coordinates to graph nodes, avoiding the
		         repeated spatial-index build inside osmnx.nearest_nodes.

	Returns:
		List of dicts, each with:
			- 'nodes': list of node IDs in the path
			- 'coords': list of (lat, lon) tuples along the path
			- 'cost': total path cost (sum of `weight` over edges)
		Ordered from lowest cost to highest.
	"""
	# Snap input coordinates to nearest graph nodes (OSMnx uses x=lon, y=lat)
	if snap is not None:
		orig_node = snap(lat1, lon1)
		dest_node = snap(lat2, lon2)
	else:
		orig_node = ox.distance.nearest_nodes(G, X=lon1, Y=lat1)
		dest_node = ox.distance.nearest_nodes(G, X=lon2, Y=lat2)

	paths = []
	try:
		path_generator = nx.shortest_simple_paths(G, orig_node, dest_node, weight)
		for node_path in islice(path_generator, k):
			coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in node_path]
			cost = sum(
				G[u][v][weight]
				for u, v in zip(node_path[:-1], node_path[1:])
			)
			paths.append({
				"nodes": node_path,
				"coords": coords,
				"cost": cost
			})
	except nx.NetworkXNoPath:
		print("No path exists between these two points.")

	return paths