"""
Dynamic, non-retrained risk-scoring and route-selection system.

Computes crime risk scores for candidate routes dynamically at query time
using recency-weighted and spatially-weighted crime statistics.  All weights
are recomputed live — the underlying ML model is never retrained.

Formulas
--------
1.  w_recency(c, t_query) = exp(-ln(2) * age / half_life)
purpose: Determines how much a single historical crime record should "count" right now, based purely on how old it is relative to the moment someone is actually requesting a route — not relative to whenever the model happened to be trained.

2.  K_space(dist_ic)       = exp(-dist_ic^2 / (2 * bw_space^2))
purpose: Determines how much a crime record should count based purely on how physically close it is to the point on the route you're evaluating — replacing rigid zone/boundary lookups with smooth distance-based relevance.

3.  w_c(t_query)           = w_recency * K_space
purpose: Merges "how recent" and "how close" into a single importance score per record, so that a record only strongly influences the result if it satisfies both conditions — recent AND nearby.

4.  freq_hour/month        = recency-weighted crime frequency by hour/month
    Temporal_Modifier      = freq_hour * freq_month * freq_weekend  (rescaled -> [0.5, 1.5])
purpose: Replaces a simple historical count ("how many crimes happened at 9am, ever") with a recency-adjusted proportion ("how many crimes happened at 9am, weighted so recent ones count more"), so the pattern reflects current conditions rather than being diluted equally by decades-old data.

5.  Spatial_Modifier_raw   = sum(severity_c * w_c) / sum(w_c)  (rescaled -> [0.5, 1.5])
purpose: Computes a localized, personalized crime-severity estimate for a specific route point, built from nearby-and-recent records only, instead of relying on a coarse administrative zone average that ignores exact position and record age.

6.  R_i(t_query)           = Crime_Score_i x Temporal_Modifier x Spatial_Modifier
purpose: Combines "how bad the crime type historically was" with "is this a risky time" and "is this a risky place," all multiplied together, into one single risk number for a specific point on the route, at a specific query time.

7.  R_route_mean           = sum(R_i * len_i) / sum(len_i)
    R_route_max            = max(R_i)
purpose: ollapses many individual point risk scores along an entire route into one or two summary numbers representing the route's overall danger level.    

8.  Safest_Route           = argmin[alpha * R_mean + beta * R_max]
purpose; Picks the single best (safest) route out of your candidate set, at the current query time, by balancing "overall average danger" against "worst single moment of danger," according to how much you personally care about each.

NOTE ON WHY ROUTES CAN LOOK IDENTICAL
--------------------------------------
If your candidate routes are geographically close together (sharing the
same start/end, only diverging by a block or two) AND your crime dataset
is sparse relative to `bw_space`, the spatial-radius query for every point
on every route can pull in the SAME set of nearby crime records with
nearly identical weights. That produces near-identical Spatial_Modifier
values across routes -- this is expected behavior, not a bug, when routes
are that close together. Temporal_Modifier is ALWAYS identical across
routes by design (it depends only on t_query, not on location).

If you're seeing identical results with real, well-separated routes and a
large crime dataset, check:
  1. That `k` in _query_radius doesn't exceed len(crime_df) (fixed below).
  2. That your routes list actually contains different lat/lon points
     (print len(set(...)) per route to confirm they're not duplicates).
  3. That bw_space isn't so large it blurs out all spatial distinction
     (try a smaller bw_space, e.g. 100-150m, for city-block-level routes).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from typing import Any

_EARTH_RADIUS_M = 6_371_000.0


# -- Helpers ------------------------------------------------------------------

def _validate(routes, crime_df, t_query):
	if not routes:
		raise ValueError("routes must be a non-empty list")
	required = {"Latitude", "Longitude", "Date", "FBI Code", "Crime_Score"}
	missing = required - set(crime_df.columns)
	if missing:
		raise ValueError(f"crime_df missing columns: {missing}")
	if not isinstance(t_query, pd.Timestamp):
		raise ValueError("t_query must be a pandas Timestamp")
	for i, route in enumerate(routes):
		if not route:
			raise ValueError(f"routes[{i}] is empty")
		for j, pt in enumerate(route):
			if len(pt) != 3:
				raise ValueError(f"routes[{i}][{j}]: expected (lat, lon, len_m)")


def _clamp_positive(x):
    	return np.maximum(0.0, x)


# -- 1 & 2. Recency weight & spatial kernel -----------------------------------

def recency_weight(ages_years: np.ndarray, half_life_years: float = 5.0) -> np.ndarray:
	"""w = exp(-ln(2) * age / half_life)."""
	return np.exp(-np.log(2.0) * ages_years / half_life_years)


def spatial_kernel(dist_m: np.ndarray, bw_space: float = 300.0) -> np.ndarray:
	"""Gaussian spatial kernel: exp(-d^2 / 2bw^2)."""
	return np.exp(-(dist_m ** 2) / (2.0 * bw_space ** 2))


# -- Spatial index -------------------------------------------------------------

def build_tree(df: pd.DataFrame) -> BallTree:
	"""Build a BallTree index (haversine metric) on crime locations."""
	coords = np.radians(df[["Latitude", "Longitude"]].values.astype(np.float64))
	return BallTree(coords, metric="haversine")


def _query_radius(tree: BallTree, lat: float, lon: float, radius_m: float, n_total: int):
	"""Return (indices, distances_m) for crime points within radius_m.

	Uses kNN query with post-filtering because BallTree.query_radius with
	haversine metric returns incorrect distances when return_distance=True.

	FIX: k must never exceed the number of points actually in the tree,
	or sklearn will raise (or, in older versions, silently misbehave).
	This was the most likely cause of identical / degenerate results when
	testing with small dummy datasets.
	"""
	k = min(200, n_total)
	d_rad, idx = tree.query(np.radians([[lat, lon]]), k=k)
	dist_m = d_rad[0] * _EARTH_RADIUS_M
	idx = idx[0]
	mask = dist_m <= radius_m
	return idx[mask], dist_m[mask]


def _query_knn(tree: BallTree, lat: float, lon: float, k: int, n_total: int):
	"""Return (indices, distances_m) for k nearest crime neighbors."""
	k = min(k, n_total)
	d_rad, idx = tree.query(np.radians([[lat, lon]]), k=k)
	return idx[0], d_rad[0] * _EARTH_RADIUS_M


def _query_knn_idx(tree: BallTree, lat: float, lon: float, k: int, n_total: int):
	"""Return indices only for k nearest crime neighbors (for base severity lookup)."""
	k = min(k, n_total)
	idx = tree.query(np.radians([[lat, lon]]), k=k, return_distance=False)
	return idx[0]


# -- 4. Temporal modifier -------------------------------------------------------

def temporal_frequencies(df: pd.DataFrame, t_query: pd.Timestamp,
						half_life_years: float = 5.0) -> dict:
	"""Recency-weighted crime frequency by hour, month, and weekend/weekday."""
	delta_sec = (t_query - df["Date"]).dt.total_seconds().values
	ages = _clamp_positive(delta_sec / (365.25 * 24 * 3600.0))
	w = recency_weight(ages, half_life_years)
	total = w.sum()
	if total <= 0:
		total = 1.0

	tmp = df[["Date"]].copy()
	tmp["w"] = w
	tmp["h"] = tmp["Date"].dt.hour
	tmp["m"] = tmp["Date"].dt.month
	tmp["we"] = tmp["Date"].dt.dayofweek.isin([5, 6]).astype(int)

	f_h = tmp.groupby("h")["w"].sum() / total
	f_m = tmp.groupby("m")["w"].sum() / total
	f_we = tmp.groupby("we")["w"].sum() / total

	return {
		"hour": f_h.to_dict(),
		"month": f_m.to_dict(),
		"weekend": f_we.get(1, 0.0),
		"weekday": f_we.get(0, 0.0),
	}


def _temp_modifier_raw(t_query: pd.Timestamp, freq: dict) -> float:
	fh = freq["hour"].get(t_query.hour, 0.0)
	fm = freq["month"].get(t_query.month, 0.0)
	fw = freq["weekend"] if t_query.dayofweek in [5, 6] else freq["weekday"]
	return fh * fm * fw


def _temp_rescale_bounds(freq: dict):
	vals = []
	for h in range(24):
		fh = freq["hour"].get(h, 0.0)
		for m in range(1, 13):
			fm = freq["month"].get(m, 0.0)
			for we in (False, True):
				fw = freq["weekend"] if we else freq["weekday"]
				vals.append(fh * fm * fw)
	return float(np.min(vals)), float(np.max(vals))


# -- 5. Spatial modifier ---------------------------------------------------------

def spatial_modifier_raw(tree: BallTree, df: pd.DataFrame,
                          lat: float, lon: float,
                          t_query: pd.Timestamp,
                          half_life_years: float = 5.0,
                          bw_space: float = 300.0) -> float:
	"""Local weighted severity at (lat, lon) -- the raw spatial modifier."""
	n_total = len(df)
	idx, dist = _query_radius(tree, lat, lon, 3.0 * bw_space, n_total)
	if len(idx) == 0:
		k = min(10, n_total)
		idx, dist = _query_knn(tree, lat, lon, k=k, n_total=n_total)

	sub = df.iloc[idx]
	delta_sec = (t_query - sub["Date"]).dt.total_seconds().values
	ages = _clamp_positive(delta_sec / (365.25 * 24 * 3600.0))
	w = recency_weight(ages, half_life_years) * spatial_kernel(dist, bw_space)
	tw = w.sum()
	if tw < 1e-12:
		return float(sub["Crime_Score"].mean())
	return float(np.average(sub["Crime_Score"].values, weights=w))


def _spatial_rescale_bounds(tree: BallTree, df: pd.DataFrame,
                             t_query: pd.Timestamp,
                             half_life_years: float = 5.0,
                             bw_space: float = 300.0,
                             n_samples: int = 500):
	if len(df) == 0:
		return 0.0, 100.0
	n = min(n_samples, len(df))
	pts = df.sample(n=n, random_state=42)[["Latitude", "Longitude"]].values
	vals = [spatial_modifier_raw(tree, df, p[0], p[1], t_query,
							half_life_years, bw_space) for p in pts]
	return float(np.min(vals)), float(np.max(vals))


# -- Rescaling ---------------------------------------------------------------

def rescale_05_15(x: float, lo: float, hi: float) -> float:
	"""Min-max rescale x from range [lo, hi] -> [0.5, 1.5]."""
	if hi <= lo or np.isclose(hi, lo):
		return 1.0
	return 0.5 + (x - lo) / (hi - lo)


# -- 6. Point risk -------------------------------------------------------------

def point_risk(severity: float, temporal_mod: float,
				spatial_raw: float, spat_lo: float, spat_hi: float) -> float:
	"""R_i = Crime_Score_i x Temporal_Modifier x Spatial_Modifier."""
	spat_mod = rescale_05_15(spatial_raw, spat_lo, spat_hi)
	return severity * temporal_mod * spat_mod


# -- 7. Route aggregation -------------------------------------------------------

def route_scores(risks, lengths):
	"""Length-weighted mean and max of point risks along a route."""
	r = np.asarray(risks, dtype=float)
	l = np.asarray(lengths, dtype=float)
	if l.sum() <= 0:
		l = np.ones_like(l)  # guard against all-zero lengths
	return float(np.average(r, weights=l)), float(np.max(r))


# -- 8. Orchestrator -------------------------------------------------------------

def safest_route(
	routes: list,
	crime_df: pd.DataFrame,
	t_query: pd.Timestamp,
	half_life_years: float = 15.0,
	bw_space: float = 300.0,
	alpha: float = 0.7,
	beta: float = 0.3,
	debug: bool = False,
) -> dict:
	"""Select the safest route from candidates at query time."""
	_validate(routes, crime_df, t_query)

	tree = build_tree(crime_df)
	n_total = len(crime_df)

	# -- Temporal modifier (identical across all routes, by design) --------
	freq = temporal_frequencies(crime_df, t_query, half_life_years)
	t_raw = _temp_modifier_raw(t_query, freq)
	t_lo, t_hi = _temp_rescale_bounds(freq)
	t_mod = rescale_05_15(t_raw, t_lo, t_hi)

	if debug:
		print(f"[debug] Temporal_Modifier = {t_mod:.6f} (same for every route -- expected)")

	# -- Spatial rescaling bounds (fit once on historical sample) ----------
	s_lo, s_hi = _spatial_rescale_bounds(tree, crime_df, t_query, half_life_years, bw_space)
	if debug:
		print(f"[debug] Spatial rescale bounds: lo={s_lo:.4f} hi={s_hi:.4f}")
		if np.isclose(s_lo, s_hi):
			print("[debug] WARNING: s_lo == s_hi -> every Spatial_Modifier will collapse to 1.0")

	# -- Per-route computation ----------------------------------------------
	route_results = []
	for ri, route in enumerate(routes):
		risks = []
		lengths = []
		for pi, (lat, lon, seg_len) in enumerate(route):
			sevy = spatial_modifier_raw(tree, crime_df, lat, lon, t_query, half_life_years, bw_space)
			nearest_idx = _query_knn_idx(tree, lat, lon, k=1, n_total=n_total)
			base_severity = float(crime_df.iloc[nearest_idx[0]]["Crime_Score"])
			r = point_risk(base_severity, t_mod, sevy, s_lo, s_hi)
			risks.append(r)
			lengths.append(seg_len)

			if debug:
				print(f"[debug] route={ri} point={pi} lat={lat:.5f} lon={lon:.5f} "
					f"base_severity={base_severity:.2f} spatial_raw={sevy:.4f} "
					f"R_i={r:.4f}")

		r_mean, r_max = route_scores(risks, lengths)
		combined = alpha * r_mean + beta * r_max
		route_results.append({
			"route": route,
			"R_route_mean": r_mean,
			"R_route_max": r_max,
			"combined_score": combined,
			"point_risks": risks,
		})

	best = min(route_results, key=lambda x: x["combined_score"])
	best_idx = route_results.index(best)

	return {
		"safest_route_index": best_idx,
		"safest_route": best["route"],
		"R_route_mean": best["R_route_mean"],
		"R_route_max": best["R_route_max"],
		"combined_score": best["combined_score"],
		"all_scores": [
			{
				"route_index": i,
				"R_route_mean": r["R_route_mean"],
				"R_route_max": r["R_route_max"],
				"combined_score": r["combined_score"],
				"point_risks": r["point_risks"],
			}
			for i, r in enumerate(route_results)
		],
	}
