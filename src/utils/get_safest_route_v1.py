"""
Route risk-scoring and selection system -- STATIC time, DECAYING space.

This is the "v1 fixed" variant: temporal contribution is now a plain
historical aggregation (every record counts equally, regardless of age).
Spatial contribution still uses a Gaussian distance-decay kernel, exactly
as before. The underlying ML model is still never retrained.

Formulas
--------
1.  K_space(dist_ic)       = exp(-dist_ic^2 / (2 * bw_space^2))
    purpose: how much a crime record counts based purely on physical
    distance from the route point being evaluated. Still decaying.

2.  w_c                    = K_space   (no more recency term)
    purpose: pure spatial relevance weight per record.

3.  freq_hour/month        = plain (unweighted) crime frequency by hour/month
    Temporal_Modifier      = freq_hour * freq_month * freq_weekend  (rescaled -> [0.5, 1.5])
    purpose: static, all-time historical pattern for this hour/month/
    weekend bucket -- no age weighting, just aggregation. t_query is still
    used to pick *which* hour/month/weekend bucket applies, but every
    historical record contributes equally to that bucket's frequency.

4.  Spatial_Modifier_raw   = sum(severity_c * w_c) / sum(w_c)  (rescaled -> [0.5, 1.5])
    purpose: localized severity estimate at a route point, weighted only
    by distance (nearby records dominate; far ones fade out; age no
    longer matters).

5.  R_i(t_query)           = Crime_Score_i x Temporal_Modifier x Spatial_Modifier
6.  R_route_mean           = sum(R_i * len_i) / sum(len_i)
    R_route_max            = max(R_i)
7.  Safest_Route           = argmin[alpha * R_mean + beta * R_max]

(unchanged from before -- only the *inputs* to step 5 changed)

NOTE ON WHY ROUTES CAN LOOK IDENTICAL
--------------------------------------
Same caveat as before, but now purely about space: if candidate routes
are geographically close together AND the crime dataset is sparse
relative to `bw_space`, every point on every route can pull in the same
set of nearby records with nearly identical weights, producing
near-identical Spatial_Modifier values. Temporal_Modifier is ALWAYS
identical across routes by design (it depends only on t_query's
hour/month/weekend bucket, not on location).

If you're seeing identical results with real, well-separated routes and
a large crime dataset, check:
  1. That `k` in _query_radius doesn't exceed len(crime_df).
  2. That your routes list actually contains different lat/lon points.
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


# -- Spatial kernel (still decaying) -------------------------------------------

def spatial_kernel(dist_m: np.ndarray, bw_space: float = 300.0) -> np.ndarray:
	"""Gaussian spatial kernel: exp(-d^2 / 2bw^2). Still distance-decayed."""
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

	k must never exceed the number of points actually in the tree, or
	sklearn will raise (or, in older versions, silently misbehave).
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


# -- 3. Temporal modifier: now STATIC (plain aggregation, no recency decay) ---

def temporal_frequencies(df: pd.DataFrame) -> dict:
	"""Plain (unweighted) crime frequency by hour, month, and weekend/weekday.

	Every historical record counts equally regardless of how old it is --
	this is now a static aggregation, not a recency-weighted one.
	"""
	total = len(df)
	if total <= 0:
		total = 1

	tmp = df[["Date"]].copy()
	tmp["h"] = tmp["Date"].dt.hour
	tmp["m"] = tmp["Date"].dt.month
	tmp["we"] = tmp["Date"].dt.dayofweek.isin([5, 6]).astype(int)

	f_h = tmp.groupby("h").size() / total
	f_m = tmp.groupby("m").size() / total
	f_we = tmp.groupby("we").size() / total

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


# -- 4. Spatial modifier: still decaying by distance, no time term ------------

def spatial_modifier_raw(tree: BallTree, df: pd.DataFrame,
                          lat: float, lon: float,
                          bw_space: float = 300.0) -> float:
	"""Local weighted severity at (lat, lon) -- distance-decay only.

	Time no longer factors into this weight at all: every candidate
	record within range contributes based purely on how close it is,
	regardless of when it happened.
	"""
	n_total = len(df)
	idx, dist = _query_radius(tree, lat, lon, 3.0 * bw_space, n_total)
	if len(idx) == 0:
		k = min(10, n_total)
		idx, dist = _query_knn(tree, lat, lon, k=k, n_total=n_total)

	sub = df.iloc[idx]
	w = spatial_kernel(dist, bw_space)
	tw = w.sum()
	if tw < 1e-12:
		return float(sub["Crime_Score"].mean())
	return float(np.average(sub["Crime_Score"].values, weights=w))


def precompute(crime_df: pd.DataFrame, bw_space: float = 300.0):
    """Precompute everything that is route-independent, once.

    Returns (tree, freq, spatial_bounds) so a serving layer can build them a
    single time and reuse them across requests instead of rebuilding the
    BallTree / temporal aggregation / spatial rescale bounds per call.
    """
    tree = build_tree(crime_df)
    freq = temporal_frequencies(crime_df)
    s_lo, s_hi = _spatial_rescale_bounds(tree, crime_df, bw_space)
    return tree, freq, (s_lo, s_hi)


def _spatial_rescale_bounds(tree: BallTree, df: pd.DataFrame,
                             bw_space: float = 300.0,
                             n_samples: int = 500):
	if len(df) == 0:
		return 0.0, 100.0
	n = min(n_samples, len(df))
	pts = df.sample(n=n, random_state=42)[["Latitude", "Longitude"]].values
	vals = [spatial_modifier_raw(tree, df, p[0], p[1], bw_space) for p in pts]
	return float(np.min(vals)), float(np.max(vals))


# -- Rescaling ---------------------------------------------------------------

def rescale_05_15(x: float, lo: float, hi: float) -> float:
	"""Min-max rescale x from range [lo, hi] -> [0.5, 1.5]."""
	if hi <= lo or np.isclose(hi, lo):
		return 1.0
	return 0.5 + (x - lo) / (hi - lo)


# -- 5. Point risk -------------------------------------------------------------

def point_risk(severity: float, temporal_mod: float,
				spatial_raw: float, spat_lo: float, spat_hi: float) -> float:
	"""R_i = Crime_Score_i x Temporal_Modifier x Spatial_Modifier."""
	spat_mod = rescale_05_15(spatial_raw, spat_lo, spat_hi)
	return severity * temporal_mod * spat_mod


# -- 6. Route aggregation -------------------------------------------------------

def route_scores(risks, lengths):
	"""Length-weighted mean and max of point risks along a route."""
	r = np.asarray(risks, dtype=float)
	l = np.asarray(lengths, dtype=float)
	if l.sum() <= 0:
		l = np.ones_like(l)  # guard against all-zero lengths
	return float(np.average(r, weights=l)), float(np.max(r))


# -- 7. Orchestrator -------------------------------------------------------------

def safest_route(
	routes: list,
	crime_df: pd.DataFrame,
	t_query: pd.Timestamp,
	bw_space: float = 300.0,
	alpha: float = 0.7,
	beta: float = 0.3,
	debug: bool = False,
	tree: BallTree = None,
	freq: dict = None,
	spatial_bounds: tuple = None,
) -> dict:
	"""Select the safest route from candidates at query time.

	Temporal_Modifier is a STATIC aggregation: it reflects the all-time
	historical frequency of crimes for t_query's hour/month/weekend
	bucket, with no recency weighting -- identical for every route.
	Spatial_Modifier keeps its distance-decay weighting per route point.

	Optional precomputed `tree` / `freq` / `spatial_bounds` (see precompute)
	avoid rebuilding route-independent state on every call; when omitted
	they are computed from crime_df as before.
	"""
	_validate(routes, crime_df, t_query)

	if tree is None or freq is None or spatial_bounds is None:
		tree, freq, spatial_bounds = precompute(crime_df, bw_space)
	s_lo, s_hi = spatial_bounds
	n_total = len(crime_df)

	# -- Temporal modifier: static, identical across all routes ------------
	t_raw = _temp_modifier_raw(t_query, freq)
	t_lo, t_hi = _temp_rescale_bounds(freq)
	t_mod = rescale_05_15(t_raw, t_lo, t_hi)

	if debug:
		print(f"[debug] Temporal_Modifier = {t_mod:.6f} (static aggregation, same for every route)")

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
			sevy = spatial_modifier_raw(tree, crime_df, lat, lon, bw_space)
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