"""Route-resource cache.

The walk graph and crime records are expensive to load (and the graph's
node spatial index is expensive to rebuild), so this module loads them
once into a process-wide singleton and reuses them for every request.
It also caches the route-independent pieces of the risk-scoring step that
``safest_route`` would otherwise recompute per call: the crime BallTree,
the temporal frequencies, and the spatial rescale bounds.

Local dev and production behave identically: nothing is re-read between
requests. On process restart the artifacts are re-read from disk -- point
``C.WORKSPACE / "graph"`` and ``C.EVENTS_FILE`` at a Railway persistent
volume (or object storage) so restarts skip re-downloading and reload
prepared files instead of raw inputs.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .. import config as C
from src.utils.get_safest_route_v1 import precompute

DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"


class RouteCache:
    """Lazily-loaded, process-wide cache of routing resources (thread-safe)."""

    def __init__(self, graph_file=None, events_file=None, bw_space: float = 50.0):
        default_graph = os.environ.get(
            "RISK_GRAPH_FILE", C.WORKSPACE / "graph" / "chicago_walk_graph.joblib")
        self.graph_file = str(graph_file or default_graph)
        self.events_file = events_file or C.EVENTS_FILE
        self.bw_space = float(bw_space)
        self._lock = threading.Lock()
        self._loaded = False
        self._graph = None
        self._df = None
        self._node_tree = None
        self._node_ids = None
        self._tree = None
        self._freq = None
        self._s_bounds = None

    # -- read-only accessors ---------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def graph(self):
        return self._graph

    @property
    def df(self):
        """Prepared crime records: Latitude, Longitude, Date, FBI Code, Crime_Score."""
        return self._df

    @property
    def tree(self):
        return self._tree

    @property
    def freq(self):
        return self._freq

    @property
    def s_bounds(self):
        return self._s_bounds

    # -- loading ---------------------------------------------------------------

    def load(self) -> bool:
        """Load every resource once. Safe to call from every request."""
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            try:
                import joblib
                graph = joblib.load(self.graph_file)
                df = self._load_crime_df()
                node_tree, node_ids = self._build_node_index(graph)
                tree, freq, s_bounds = precompute(df, self.bw_space)
                self._graph, self._df = graph, df
                self._node_tree, self._node_ids = node_tree, node_ids
                self._tree, self._freq, self._s_bounds = tree, freq, s_bounds
                self._loaded = True
                print(f"[routes] cache ready: {len(node_ids):,} graph nodes, "
                      f"{len(df):,} crime records")
                return True
            except FileNotFoundError:
                print("[routes] chicago_walk_graph.joblib not found; /route/* "
                      "will return 503. Run `src/notebooks/chicago_map.ipynb` "
                      "to download it.")
            except Exception as exc:
                print(f"[routes] route resources unavailable: {exc!r}")
        return False

    def _load_crime_df(self) -> pd.DataFrame:
        events = pd.read_parquet(self.events_file)
        if "Crime_Score" in events.columns:
            pass
        elif "Severity_Score" in events.columns:
            events = events.rename(columns={"Severity_Score": "Crime_Score"})
        else:
            raise ValueError("no severity/score column available for routing")
        if "Date" not in events.columns and "Parsed_Date" in events.columns:
            events = events.rename(columns={"Parsed_Date": "Date"})
        if "FBI Code" not in events.columns and "Primary Type" in events.columns:
            events["FBI Code"] = events["Primary Type"].astype(str)
        events["Date"] = pd.to_datetime(
            events["Date"], format=DATE_FORMAT, errors="coerce")
        cols = ["Latitude", "Longitude", "Date", "FBI Code", "Crime_Score"]
        events = events[[c for c in cols if c in events.columns]]
        events = events.drop_duplicates().dropna(
            subset=["Latitude", "Longitude", "Date"])
        return events

    @staticmethod
    def _build_node_index(graph):
        """Haversine BallTree over graph nodes -> (tree, node_ids).

        Same metric/coordinates as ``osmnx.distance.nearest_nodes``, so snaps
        are identical, but the index is built once instead of every request.
        """
        node_ids = list(graph.nodes)
        coords = np.radians(
            [[graph.nodes[n]["y"], graph.nodes[n]["x"]] for n in node_ids]
        ).astype(np.float64)
        return BallTree(coords, metric="haversine"), node_ids

    # -- snapping ----------------------------------------------------------------

    def snap(self, lat: float, lon: float):
        """Nearest graph node id to (lat, lon)."""
        d_rad, idx = self._node_tree.query(np.radians([[lat, lon]]), k=1)
        return self._node_ids[idx[0][0]]
