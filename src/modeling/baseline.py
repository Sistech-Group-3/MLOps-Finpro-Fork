from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import _haversine_km


# Core: day-bucketed severity with exponential decay (inverse-distance weighted)
def _recurrence(E: np.ndarray, decay: float) -> np.ndarray:
    """R_d = decay · (R_{d-1} + E_{d-1}) with R_0 = 0.

    This yields Σ_k sev_k·exp(-K·Δt_k) in O(D) per cluster (exact for the discrete
    daily grid the target lives on), avoiding an O(n²) history rescan.
    """
    R = np.zeros(len(E), dtype=float)
    acc = 0.0
    for d in range(1, len(E)):
        acc = decay * (acc + E[d - 1])
        R[d] = acc
    return R


def baseline_score_frame(df: pd.DataFrame, events: pd.DataFrame):
    """Compute a normalised baseline score for every (cluster, day) row.

    Returns ``(array, {"lo", "hi"})`` — the normalisation bounds are fit on the
    chronological central 60% band (a stand-in for the training split) so the
    baseline lives on the same [1, 100] scale as the model target with no future leak.
    """
    frame = df.copy().reset_index(drop=True)
    date_rows = pd.to_datetime(frame[C.DATE_COL]).to_numpy()
    day = pd.DatetimeIndex(pd.DatetimeIndex(np.asarray(date_rows)).date)  # floor-D
    unique = pd.DatetimeIndex(np.sort(pd.unique(day)))
    D = len(unique)
    row_bucket = np.searchsorted(unique, day).astype(np.int64)

    ev = events.dropna(subset=["Parsed_Date", "Latitude", "Longitude",
                               "Severity_Score"]).reset_index(drop=True)
    ev_day = pd.DatetimeIndex(pd.to_datetime(ev["Parsed_Date"].values).floor("D"))
    ev_bucket = np.clip(np.searchsorted(unique, ev_day).astype(np.int64), 0, D - 1)
    ev_sev = ev["Severity_Score"].to_numpy(dtype=float)
    ev_lat = ev["Latitude"].to_numpy()
    ev_lon = ev["Longitude"].to_numpy()

    decay = float(np.exp(-C.LAMBDA_TIME))
    raw = np.zeros(len(frame))

    # + nearest-centre mapping: each row belongs to its cluster centre
    clusters = pd.unique(frame[C.CLUSTER_COL])
    for cl in clusters:
        mask = frame[C.CLUSTER_COL].values == cl
        clat, clon = (float(frame.loc[mask, "Latitude"].iloc[0]),
                      float(frame.loc[mask, "Longitude"].iloc[0]))
        d = _haversine_km(clat, clon, ev_lat, ev_lon)
        inside = d <= C.RADIUS_KM
        if not inside.any():
            continue
        sub_sev = ev_sev[inside]
        sub_b = ev_bucket[inside]
        sub_d = d[inside]
        E = np.zeros(D)
        w = sub_sev / (1.0 + C.GAMMA_SPACE * sub_d)
        np.add.at(E, sub_b, w)
        R = _recurrence(E, decay)
        raw[mask] = R[row_bucket[mask]]

    band = _train_band(row_bucket, D)
    lo = float(np.min(raw[band])) if band.any() else float(np.min(raw))
    hi = float(np.max(raw[band])) if band.any() else float(np.max(raw))
    if hi - lo < 1e-12:
        hi = lo + 1.0
    normalized = 1.0 + 99.0 * (raw - lo) / (hi - lo)
    return np.clip(normalized, 1.0, 100.0), {"lo": lo, "hi": hi}


def _train_band(row_bucket, D):
    """Boolean mask for the central 60% of temporal buckets (train-like band)."""
    return (row_bucket >= int(D * 0.0)) & (row_bucket <= int(D * 0.6))


# Runtime geo baseline used by the API
def risk_score_geo(lat: float, lon: float, when, events: pd.DataFrame):
    """Raw baseline formula value at (lat, lon, datetime).

    The API maps the returned unbounded sum to [0, 100] using the (lo, hi) bounds
    stored in the registry after training.
    """
    ev = events.dropna(subset=["Latitude", "Longitude", "Severity_Score",
                               "Parsed_Date"])
    if ev.empty:
        return 0.0
    when = pd.to_datetime(when)
    d = _haversine_km(lat, lon, ev["Latitude"].to_numpy(), ev["Longitude"].to_numpy())
    inside = d <= C.RADIUS_KM
    if not inside.any():
        return 0.0
    ev_dt = pd.to_datetime(ev["Parsed_Date"].to_numpy())
    dt_days = (when - ev_dt).total_seconds().to_numpy() / 86400.0
    recent = inside & (dt_days >= 0) & (dt_days <= C.BASELINE_WINDOW_DAYS)
    if not recent.any():
        return 0.0
    sev = ev["Severity_Score"].to_numpy(dtype=float)[recent]
    dr = d[recent]
    wt = np.exp(-C.LAMBDA_TIME * dt_days[recent])
    ws = 1.0 / (1.0 + C.GAMMA_SPACE * dr)
    return float(np.sum(sev * wt * ws))