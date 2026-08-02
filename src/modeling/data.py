from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# Loading
def load_target() -> pd.DataFrame:
    df = pd.read_parquet(C.TARGET_FILE)
    df[C.DATE_COL] = pd.to_datetime(df[C.DATE_COL])
    return df.sort_values([C.CLUSTER_COL, C.DATE_COL]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    ev = pd.read_parquet(C.EVENTS_FILE)
    ev["Parsed_Date"] = pd.to_datetime(ev["Parsed_Date"], errors="coerce")
    keep = ["Parsed_Date", "Latitude", "Longitude", "Severity_Score"]
    return ev[keep].dropna().reset_index(drop=True)


# Distance
def _haversine_km(lat, lon, lat2, lon2):
    R = 6371.0
    v = np.radians(lat2 - lat), np.radians(lon2 - lon)
    a = np.sin(v[0] / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(lat2)) \
        * np.sin(v[1] / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# Temporal encodings
def _temporal(d: pd.Series) -> dict:
    date = pd.to_datetime(d)
    start = date.min()
    return {
        "year": date.dt.year.astype(float),
        "month": date.dt.month.astype(float),
        "day": date.dt.day.astype(float),
        "dow": date.dt.dayofweek.astype(float),
        "days_since_start": (date - start).dt.days.astype(float),
        "month_sin": np.sin(2 * np.pi * date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * date.dt.month / 12),
        "dow_sin": np.sin(2 * np.pi * date.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * date.dt.dayofweek / 7),
        "day_sin": np.sin(2 * np.pi * date.dt.day / 31),
        "day_cos": np.cos(2 * np.pi * date.dt.day / 31),
        "weekend": (date.dt.dayofweek >= 5).astype(float),
    }


# Static per-cluster features
def _cluster_centres(df):
    return (df.groupby(C.CLUSTER_COL)[["Latitude", "Longitude"]]
            .mean().reset_index())


def neighbourhood_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster static aggregates of nearby incidents (computed once)."""
    centres = _cluster_centres(df)
    ev = events[["Latitude", "Longitude", "Severity_Score"]]
    lat, lon = ev["Latitude"].to_numpy(), ev["Longitude"].to_numpy()
    sev = ev["Severity_Score"].to_numpy(dtype=float)

    rows = []
    for cl, clat, clon in centres.itertuples(index=False, name=None):
        d = _haversine_km(clat, clon, lat, lon)
        inside = d <= C.RADIUS_KM
        if inside.any():
            rows.append((cl, int(inside.sum()), float(sev[inside].mean()),
                         float(sev[inside].max())))
        else:
            rows.append((cl, 0, 0.0, 0.0))
    agg = pd.DataFrame(rows, columns=[C.CLUSTER_COL, "near_count", "near_mean_sev",
                                      "near_max_sev"])
    return df.merge(agg, on=C.CLUSTER_COL, how="left").fillna(0.0)


# Trailing (past-only) temporal features — the real temporal signal
def _per_cluster_day_weights(frame, events):
    """Yield (cluster, distances, day-bucket, severity) for nearby incidents."""
    unique = pd.DatetimeIndex(frame[C.DATE_COL]).floor("D").unique().sort_values()
    D = len(unique)
    row_bucket = np.searchsorted(unique,
                                 pd.DatetimeIndex(frame[C.DATE_COL])).astype(int)

    ev_bucket = np.searchsorted(unique, pd.DatetimeIndex(events["Parsed_Date"])
                                .floor("D")).astype(int)
    ev_bucket = np.clip(ev_bucket, 0, D - 1)
    ev_lat, ev_lon = events["Latitude"].to_numpy(), events["Longitude"].to_numpy()
    ev_sev = events["Severity_Score"].to_numpy(dtype=float)

    for cl in pd.unique(frame[C.CLUSTER_COL]):
        mask = frame[C.CLUSTER_COL].values == cl
        clat, clon = float(frame.loc[mask, "Latitude"].iloc[0]), float(
            frame.loc[mask, "Longitude"].iloc[0])
        d = _haversine_km(clat, clon, ev_lat, ev_lon)
        inside = d <= C.RADIUS_KM
        yield cl, row_bucket[mask], d[inside], ev_bucket[inside], ev_sev[inside]


def _trailing(frame, events, windows=(7, 30, 90)):
    unique = pd.DatetimeIndex(frame[C.DATE_COL]).floor("D").unique().sort_values()
    D = len(unique)
    row_bucket = np.searchsorted(unique, pd.DatetimeIndex(frame[C.DATE_COL])).astype(int)
    ev_bucket = np.searchsorted(unique, pd.DatetimeIndex(events["Parsed_Date"])
                                .floor("D")).astype(int)
    ev_bucket = np.clip(ev_bucket, 0, D - 1)
    ev_lat, ev_lon = events["Latitude"].to_numpy(), events["Longitude"].to_numpy()
    ev_sev = events["Severity_Score"].to_numpy(dtype=float)

    out = {w: np.zeros(len(frame)) for w in windows}
    for cl in pd.unique(frame[C.CLUSTER_COL]):
        mask = frame[C.CLUSTER_COL].values == cl
        clat, clon = float(frame.loc[mask, "Latitude"].iloc[0]), float(
            frame.loc[mask, "Longitude"].iloc[0])
        d = _haversine_km(clat, clon, ev_lat, ev_lon)
        inside = d <= C.RADIUS_KM
        if not inside.any():
            continue
        E = np.zeros(D + 1)
        np.add.at(E, ev_bucket[inside], ev_sev[inside] / (1.0 + C.GAMMA_SPACE * d[inside]))
        prefix = np.concatenate([[0.0], np.cumsum(E)])  # prefix[i] = sum of days < i
        for w in windows:
            lo = np.clip(row_bucket[mask] - w, 0, D)
            out[w][mask] = prefix[row_bucket[mask]] - prefix[lo]
    for w in windows:
        frame[f"risk_{w}d"] = out[w]
    return frame


# Full training feature frame
def build_feature(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with all engineered features (training entry point)."""
    out = df.copy()
    for k, v in _temporal(out[C.DATE_COL]).items():
        out[k] = v
    out = neighbourhood_features(out, events)
    out = _trailing(out, events)
    return out


# Serving: reconstruct the feature row for a query point
def serving_feature(assets: dict, events: pd.DataFrame, lat: float, lon: float,
                    when) -> np.ndarray:
    """Ordered 1×N feature row (in ``assets["feature_cols"]`` order) for the query.

    Uses pre-trained assets (nearest cluster → static features) and live event
    history (past-only trailing windows), mirroring :func:`build_feature`.
    """
    when = pd.to_datetime(when)
    ids = np.array(list(assets["cluster_centers"]))
    pts = np.asarray(list(assets["cluster_centers"].values()), dtype=float)
    d = _haversine_km(lat, lon, pts[:, 0], pts[:, 1])
    k = int(np.argmin(d))
    static = assets["static_features"].get(str(ids[k]), {})

    # past-only neighbourhood events
    ev_dt = pd.to_datetime(events["Parsed_Date"])
    dl = _haversine_km(lat, lon, events["Latitude"].to_numpy(),
                       events["Longitude"].to_numpy())
    past = (dl <= C.RADIUS_KM) & (ev_dt < when)
    past_ev, past_d = events["Severity_Score"].to_numpy()[past.to_numpy()], dl[past.to_numpy()]
    past_days = (when - ev_dt[past.to_numpy()]).dt.days.to_numpy()

    feat = {
        "year": when.year, "month": when.month, "day": when.day,
        "dow": when.dayofweek,
        "days_since_start": (when - ev_dt.min()).days,
        "month_sin": np.sin(2 * np.pi * when.month / 12),
        "month_cos": np.cos(2 * np.pi * when.month / 12),
        "dow_sin": np.sin(2 * np.pi * when.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * when.dayofweek / 7),
        "day_sin": np.sin(2 * np.pi * when.day / 31),
        "day_cos": np.cos(2 * np.pi * when.day / 31),
        "weekend": float(when.dayofweek >= 5),
        "Latitude": float(pts[k, 0]), "Longitude": float(pts[k, 1]),
        "near_count": float(static.get("near_count", 0)),
        "near_mean_sev": float(static.get("near_mean_sev", 0)),
        "near_max_sev": float(static.get("near_max_sev", 0)),
    }
    for w in (7, 30, 90):
        window = (past_days > 0) & (past_days <= w)
        if window.any():
            sev_w = past_ev[window.to_numpy()]
            dd_w = past_d[window.to_numpy()]
            feat[f"risk_{w}d"] = float(np.sum(sev_w / (1.0 + C.GAMMA_SPACE * dd_w)))
        else:
            feat[f"risk_{w}d"] = 0.0

    return np.array([feat[c] for c in assets["feature_cols"]], dtype=float).reshape(1, -1)