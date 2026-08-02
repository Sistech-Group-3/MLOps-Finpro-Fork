from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
    }


def improvement(reference: dict, candidate: dict):
    """% improvement of candidate over a reference (reference as the baseline).

    Lower-is-better for MAE/RMSE, higher-is-better for R² — each is reported as the
    relative change in the favourable direction.
    """
    return {
        "MAE_pct": (reference["MAE"] - candidate["MAE"]) / reference["MAE"] * 100.0,
        "RMSE_pct": (reference["RMSE"] - candidate["RMSE"]) / reference["RMSE"] * 100.0,
        "R2_delta": candidate["R2"] - reference["R2"],
    }


def feature_importances(model, feature_names):
    """Best-effort importance extraction for tree-based models."""
    for attr in ("feature_importances_",):
        if hasattr(model, attr):
            vals = np.asarray(getattr(model, attr))
            return dict(sorted(
                zip(feature_names, vals), key=lambda kv: kv[1], reverse=True))
    # XGBoost booster path
    try:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            score = model.get_score(importance_type="gain")
            total = sum(score.values()) or 1.0
            return {k: v / total for k, v in sorted(
                score.items(), key=lambda kv: kv[1], reverse=True)}
    except Exception:
        pass
    return {}