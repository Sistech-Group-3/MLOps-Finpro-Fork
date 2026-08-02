"""Model training pipeline (brief Section 2).

Trains the four regressors (Linear Regression, Decision Tree, Random Forest,
XGBoost) on a shared temporal split against the pseudo-labelled Risk Score, then:

* evaluates each with MAE / RMSE / R² on train/val/test;
* compares test metrics against the statistical baseline (the "% improvement" claim);
* extracts feature importances (tree models) for interpretability;
* persists versioned artifacts + metadata and registers them in ``registry.json``.

Run as::

    python -m modeling.train --subset 0.20
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from . import config as C
from . import data as D
from . import baseline as BL
from . import split as SP
from . import evaluate as EV
from . import registry as REG


# Dataset
def build_dataset(subset_frac: float = 1.0):
    """Build features + baseline, then temporal-split into (train, val, test)."""
    df = D.load_target()
    events = D.load_events()
    if subset_frac < 1.0:
        df = df.sample(frac=subset_frac, random_state=C.RANDOM_STATE).sort_index()

    frame = D.build_feature(df, events)
    base_scores, base_bounds = BL.baseline_score_frame(frame, events)
    frame["baseline"] = base_scores

    train, val, test, _ = SP.prepare_targets(frame)
    return train, val, test, base_bounds


# Model fitting helpers
def _make(params):
    return params


def _param_candidates(name, boomers_seed=None):
    if name == "LinearRegression":
        return [{}]
    if name == "DecisionTreeRegressor":
        return [{"max_depth": d, "min_samples_leaf": l}
                for d in (6, 10, 14) for l in (20, 60)]
    if name == "RandomForestRegressor":
        return [{"n_estimators": 250, "max_depth": d} for d in (15, 25)]
    if name == "XGBRegressor":
        return [{"max_depth": d, "learning_rate": lr, "subsample": 0.8,
                 "n_estimators": 1000, "objective": "reg:squarederror",
                 "verbosity": 0, "tree_method": "hist", "seed": C.RANDOM_STATE}
                for d in (4, 6) for lr in (0.05, 0.1)]
    raise ValueError(name)


def _fit_sklearn(name, params, Xtr, ytr):
    if name == "LinearRegression":
        return LinearRegression().fit(Xtr, ytr)
    if name == "DecisionTreeRegressor":
        return (DecisionTreeRegressor(random_state=C.RANDOM_STATE, **params)
                .fit(Xtr, ytr))
    if name == "RandomForestRegressor":
        return (RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1,
                                      **params).fit(Xtr, ytr))
    raise ValueError(name)


def _tune_and_fit(name, Xtr, ytr, Xval, yval, Xte):
    """Train the best candidate (by val RMSE); return (model, params, yv, yte)."""
    best_model, best_params, best_rmse = None, None, float("inf")
    for params in _param_candidates(name):
        if name == "XGBRegressor":
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dval = xgb.DMatrix(Xval, label=yval)
            m = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dval, "val")], early_stopping_rounds=20,
                          verbose_eval=False)
            yv = m.predict(dval)
        else:
            m = _fit_sklearn(name, params, Xtr, ytr)
            yv = m.predict(Xval)
        rmse = float(np.sqrt(np.mean((yv - yval) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_model, best_params = rmse, m, params

    if name == "XGBRegressor":
        yte = best_model.predict(xgb.DMatrix(Xte))
        yv = best_model.predict(xgb.DMatrix(Xval))
    else:
        yv = best_model.predict(Xval)
        yte = best_model.predict(Xte)
    return best_model, best_params, yv, yte


def _range_str(f):
    d = pd.to_datetime(f[C.DATE_COL])
    return [str(d.min()), str(d.max())]


def register_and_save(name, model, version, meta, registry):
    if name == "XGBRegressor":
        REG.save_xgboost(model, name, version)
    else:
        REG.save_model_artifact(model, name, version)
    REG.write_metadata(name, version, meta)
    REG.register_version(registry, name, version, meta)


# Orchestration
def run(subset_frac: float = 1.0) -> dict:
    train, val, test, base_bounds = build_dataset(subset_frac)
    Xtr, ytr = SP.build_xy(train)
    Xval, yval = SP.build_xy(val)
    Xte, yte = SP.build_xy(test)

    b_tr = EV.evaluate(ytr, train["baseline"])
    b_va = EV.evaluate(yval, val["baseline"])
    b_te = EV.evaluate(yte, test["baseline"])
    print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")
    print(f"Baseline  -> train {b_tr} | val {b_va} | test {b_te}")

    summary = {
        "baseline": {"train": b_tr, "val": b_va, "test": b_te},
        "normalizer": base_bounds, "feature_cols": C.FEATURE_COLS,
        "split": {"train": _range_str(train), "val": _range_str(val),
                  "test": _range_str(test)},
        "models": {},
    }
    registry = REG.load_registry()

    for name in ["LinearRegression", "DecisionTreeRegressor",
                 "RandomForestRegressor", "XGBRegressor"]:
        t0 = time.time()
        model, params, yv, yte_pred = _tune_and_fit(name, Xtr, ytr, Xval, yval, Xte)

        met_tr = EV.evaluate(ytr, model.predict(Xtr) if name != "XGBRegressor"
                             else model.predict(xgb.DMatrix(Xtr)))
        met_va = EV.evaluate(yval, yv)
        met_te = EV.evaluate(yte, yte_pred)

        imp_va = EV.improvement(b_va, met_va)
        imp_te = EV.improvement(b_te, met_te)

        version = REG.next_version(registry, name)
        meta = {
            "hyperparams": _as_serializable(params),
            "train_samples": int(len(train)),
            "train_range": _range_str(train), "val_range": _range_str(val),
            "test_range": _range_str(test),
            "metrics": {"train": met_tr, "val": met_va, "test": met_te},
            "improvement_vs_baseline": {"val": imp_va, "test": imp_te},
            "feature_importance": EV.feature_importances(model, C.FEATURE_COLS),
            "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fit_seconds": round(time.time() - t0, 2),
        }
        register_and_save(name, model, version, meta, registry)
        REG.log_metrics_ledger(name, version, met_te)
        summary["models"][name] = {"version": version, "metrics": meta["metrics"],
                                   "improvement_vs_baseline": imp_te,
                                   "top_importance": dict(
                                       list(meta["feature_importance"].items())[:8])}
        print(f"[{name}] version={version} test={met_te} "
              f"improve(MAE%={imp_te['MAE_pct']:.1f} R2Δ={imp_te['R2_delta']:.3f})")

    (C.REPORT_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2))
    REG.save_registry(registry)
    save_serving_assets(train, base_bounds, summary)
    return summary


def save_serving_assets(train, base_bounds, summary) -> None:
    """Persist the minimal assets the API needs to reproduce features at runtime.

    This is the bridge between training and serving: cluster centres (for nearest-
    location lookup), the per-cluster static features, the normaliser bounds, and
    the default (best) model. Stored as a compact JSON file the server loads once.
    """
    cl = (train.groupby(C.CLUSTER_COL)[["Latitude", "Longitude"]]
          .mean().reset_index().sort_values(C.CLUSTER_COL))
    centres = {int(r[C.CLUSTER_COL]): [float(r["Latitude"]), float(r["Longitude"])]
               for _, r in cl.iterrows()}

    # per-cluster static features (used at runtime for the target cluster)
    cols = [c for c in ("near_count", "near_mean_sev", "near_max_sev",
                        "near_seed_density") if c in train.columns]
    static = train.groupby(C.CLUSTER_COL)[cols].mean().reset_index()
    static_map = {int(r[C.CLUSTER_COL]): {c: float(r[c]) for c in cols}
                  for _, r in static.iterrows()}

    best = min(summary["models"],
               key=lambda k: summary["models"][k]["metrics"]["test"]["MAE"])
    assets = {"cluster_centers": centres, "static_features": static_map,
              "feature_cols": C.FEATURE_COLS,
              "normalizer": {"lo": base_bounds["lo"], "hi": base_bounds["hi"]},
              "default_model": best,
              "model_versions": {k: summary["models"][k]["version"]
                                 for k in summary["models"]}}
    (C.ARTIFACT_DIR / "serving_assets.json").write_text(json.dumps(assets, indent=2))


def _as_serializable(params):
    if isinstance(params, dict):
        return {k: (str(v) if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in params.items()}
    return params


def metrics(y_true, y_pred):
    return EV.evaluate(y_true, y_pred)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=float, default=0.20,
                    help="Subsample the dataset for a quick run (default 0.20).")
    args = ap.parse_args()
    run(subset_frac=args.subset)