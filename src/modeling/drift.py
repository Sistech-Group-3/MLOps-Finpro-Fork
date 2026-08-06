"""Drift detection for the continual-learning pipeline.

Before absorbing a new window we ask "has the world changed?" on three
complementary signals, combined into a verdict (none/low/medium/high) that
continual.py uses to decide whether to retrain:

* **Feature drift** — Population Stability Index (PSI) per engineered feature.
* **Target drift** — PSI plus a Kolmogorov-Smirnov test on the labelled score.
* **Prediction-error drift** — the champion's residuals on the fresh window
  vs. its training residuals (KS test + MAE ratio).

Everything is deterministic and JSON-serialisable for the drift log.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import config as C


def psi(reference, current, n_bins=10) -> float:
    """Population Stability Index between two 1-D samples (0 = identical,
    ~0.10-0.25 moderate, >0.25 significant shift)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf  # sweep outliers into the tails
    r = np.histogram(ref, bins=edges)[0].astype(float) + 1e-9
    c = np.histogram(cur, bins=edges)[0].astype(float) + 1e-9
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((r - c) * np.log(r / c)))


def ks_pvalue(a, b) -> float:
    """Two-sample KS p-value; p < KS_ALPHA means the samples differ."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 1.0
    try:
        return float(stats.ks_2samp(a, b).pvalue)
    except Exception:
        return 1.0


def feature_psi(reference, current, feature_names) -> dict:
    """PSI per feature column (arrays in feature_names order); calendar-
    deterministic features (DRIFT_IGNORE_FEATURES) are skipped."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    return {name: psi(ref[:, i], cur[:, i])
            for i, name in enumerate(feature_names)
            if name not in C.DRIFT_IGNORE_FEATURES}


def _predict(model, X, model_name) -> np.ndarray:
    """Predict with the model, handling the XGBoost DMatrix API."""
    if model_name == "XGBRegressor":
        import xgboost as xgb
        return model.predict(xgb.DMatrix(X)).astype(float)
    return model.predict(X).astype(float)


def error_drift(champion, ref_X, ref_y, cur_X, cur_y, model_name) -> dict:
    """Champion's prediction-error drift: training residuals vs. fresh-window
    residuals (KS test + MAE ratio) with the same predictor."""
    ref_resid = np.abs(_predict(champion, ref_X, model_name) - ref_y)
    cur_resid = np.abs(_predict(champion, cur_X, model_name) - cur_y)
    ref_mae = float(ref_resid.mean()) if ref_resid.size else float("nan")
    cur_mae = float(cur_resid.mean()) if cur_resid.size else float("nan")
    return {"model": model_name, "ref_mae": ref_mae, "cur_mae": cur_mae,
            "mae_ratio": (cur_mae / ref_mae) if ref_mae else float("nan"),
            "ks_pvalue": ks_pvalue(ref_resid, cur_resid)}


def compose_drift_report(*, checkpoint, window_start, window_end,
                         ref_X, cur_X, ref_y, cur_y, feature_names,
                         champion=None, model_name="XGBRegressor") -> dict:
    """Assemble a JSON drift report + verdict from all three detectors. If
    champion is None the prediction-error leg is skipped (pre-first-model)."""
    fpsi = feature_psi(ref_X, cur_X, feature_names)
    flagged = sorted(f for f, v in fpsi.items() if v >= C.PSI_WARN)
    high = sorted(f for f, v in fpsi.items() if v >= C.PSI_HIGH)
    target_psi = psi(ref_y, cur_y)
    target_ks = ks_pvalue(ref_y, cur_y)
    err = error_drift(champion, ref_X, ref_y, cur_X, cur_y, model_name) if champion else None

    triggers = [f"feature PSI {f}={fpsi[f]:.3f}>=warn" for f in flagged]
    triggers += [f"feature PSI {f}={fpsi[f]:.3f}>=high" for f in high]
    if target_psi >= C.PSI_WARN:
        triggers.append(f"target PSI={target_psi:.3f}")
    if target_ks < C.KS_ALPHA:
        triggers.append(f"target KS p={target_ks:.4f}<{C.KS_ALPHA}")
    if err and err["mae_ratio"] >= C.ERROR_DRIFT_WARN:
        triggers.append(f"error ratio={err['mae_ratio']:.2f}")

    ratio = err["mae_ratio"] if err else 0.0
    if len(high) >= 1 or target_psi >= C.PSI_HIGH or ratio >= C.ERROR_DRIFT_HIGH:
        severity = "high"
    elif len(flagged) >= 2 or target_psi >= C.PSI_WARN or ratio >= C.ERROR_DRIFT_WARN:
        severity = "medium"
    elif len(flagged) >= 1:
        severity = "low"
    else:
        severity = "none"

    return {
        "checkpoint": checkpoint, "window_start": str(window_start),
        "window_end": str(window_end),
        "feature_psi": {k: round(v, 5) for k, v in fpsi.items()},
        "features_flagged": flagged, "features_high": high,
        "target_psi": round(target_psi, 5),
        "target_ks_pvalue": round(target_ks, 5),
        "error_drift": None if err is None else {
            "model": err["model"], "ref_mae": round(err["ref_mae"], 4),
            "cur_mae": round(err["cur_mae"], 4),
            "mae_ratio": round(err["mae_ratio"], 4),
            "ks_pvalue": round(err["ks_pvalue"], 5)},
        "verdict": {"severity": severity, "drift": severity in ("medium", "high")},
        "triggers": triggers,
    }


def should_retrain(report, checkpoint_index) -> bool:
    """True when drift is medium/high, or FORCE_RETRAIN_EVERY elapses so the
    model refreshes even in calm periods."""
    drift = report["verdict"]["drift"]
    force = C.FORCE_RETRAIN_EVERY and (checkpoint_index % C.FORCE_RETRAIN_EVERY == 0)
    return bool(drift or force)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    print("same dist PSI:", psi(ref, rng.normal(0, 1, 2000)))
    print("shifted PSI:", psi(ref, rng.normal(0.5, 1.2, 2000)))
