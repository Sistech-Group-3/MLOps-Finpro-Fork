"""Continual learning / drift adaptation pipeline.

Splits data into chronological windows; a v1 champion is trained on the first
and each later window triggers drift detection (drift.py), challenger
retraining on all data seen so far, and promotion only if it beats the champion
on the fresh window (quality gate). Versioned under artifacts/continual/models/.

Run: python -m modeling.continual --quick
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import drift as DR
from . import evaluate as EV
from . import train as TR

# --- Persistence: versioning, artifact storage, and event logging ---

def load_registry() -> dict:
    """Read the model registry JSON, or a fresh empty registry if missing."""
    if C.CONTINUAL_REGISTRY.exists():
        return json.loads(C.CONTINUAL_REGISTRY.read_text())
    return {"updated": _dt.datetime.now().isoformat(timespec="seconds"), "models": {}}

def save_registry(registry: dict) -> None:
    """Persist the registry dict to disk with a fresh 'updated' timestamp."""
    registry["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
    C.CONTINUAL_REGISTRY.write_text(json.dumps(registry, indent=2))

def _deploy_model(registry: dict, model, model_name: str, meta: dict) -> str:
    """Version, save, and register a model as the new champion; returns version."""
    entry = registry.setdefault("models", {}).setdefault(model_name, {})
    history = entry.setdefault("versions", [])
    highest = 0
    for item in history:
        try:
            highest = max(highest, int(item["version"].lstrip("v")))
        except (KeyError, ValueError):
            continue
    version = f"v{highest + 1}"
    d = C.CONTINUAL_MODELS_DIR / model_name / version
    d.mkdir(parents=True, exist_ok=True)
    if model_name == "XGBRegressor":
        p = d / "model.ubj"
        model.save_model(str(p))
    else:
        p = d / "model.joblib"
        joblib.dump(model, p)
    (d / "metadata.json").write_text(
        json.dumps({"model_name": model_name, "version": version}, indent=2))
    history.append({"version": version,
                    "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
                    **meta})
    entry["active_version"] = entry["champion"] = version
    save_registry(registry)
    return version

def _append_log(path, entry: dict) -> None:
    """Append one JSON event line to a log file (evolution or drift log)."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

# --- Time partitioning (simulated data arrival) ---

def split_windows(df: pd.DataFrame, initial_days: int, ckpt_days: int,
                  min_tail_frac: float = 0.5) -> list:
    """Chronological windows; [0] initial batch, [1:] arrive in turn (short tail merged)."""
    dates = pd.to_datetime(df[C.DATE_COL])
    start, end = dates.min(), dates.max()
    cuts, cut = [], start + pd.Timedelta(days=initial_days)
    while cut < end:
        if (end - cut).days >= ckpt_days * min_tail_frac:
            cuts.append(cut)
        cut += pd.Timedelta(days=ckpt_days)
    windows, prev = [], start
    for c in cuts:
        windows.append(df[(dates >= prev) & (dates < c)])
        prev = c
    windows.append(df[dates >= prev])
    return [w for w in windows if len(w) > 0]

# --- Window features + fixed-scale normalisation ---

def extract_static_map(frame: pd.DataFrame) -> dict:
    """Per-cluster static neighbourhood features, frozen at deploy time."""
    static = frame.groupby(C.CLUSTER_COL)[C.STATIC_FEATURES].mean()
    return {int(cl): {c: float(row[c]) for c in C.STATIC_FEATURES} for cl, row in static.iterrows()}

def build_window(frame: pd.DataFrame, events: pd.DataFrame, anchor,
                 static_map: dict | None = None) -> pd.DataFrame:
    """Feature-build a window with only past events (no leakage) and global anchoring."""
    w_end = pd.to_datetime(frame[C.DATE_COL]).max()
    past = events[pd.to_datetime(events["Parsed_Date"]) <= w_end]
    out = D.build_feature(frame, past)
    if static_map:
        for cl, vals in static_map.items():
            m = out[C.CLUSTER_COL] == cl
            for c, v in vals.items():
                out.loc[m, c] = v
    out["days_since_start"] = (pd.to_datetime(out[C.DATE_COL]) - anchor).dt.days.astype(float)
    return out

def normalize_fixed(frame: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Map raw Risk_Score to [1, 100] with bounds frozen on the initial window."""
    out = frame.copy()
    out[C.TARGET_COL] = np.clip(
        1.0 + 99.0 * (frame["Risk_Score"].to_numpy(dtype=float) - lo) / (hi - lo + 1e-12),
        1.0, 100.0)
    return out

def build_xy(frame: pd.DataFrame):
    """Feature matrix and target vector from a window."""
    return (frame[C.FEATURE_COLS].to_numpy(dtype=float),
            frame[C.TARGET_COL].to_numpy(dtype=float))

def _chrono_split(frame: pd.DataFrame, val_frac: float):
    """Chronological train/validation split by date quantile."""
    dates = pd.to_datetime(frame[C.DATE_COL])
    cut = dates.quantile(1.0 - val_frac)
    return frame[dates < cut], frame[dates >= cut]

# --- Training + quality gate ---

def _fit_challenger(model_name: str, train_frame: pd.DataFrame, val_frame: pd.DataFrame):
    """Tune and fit the best candidate on (train, val); return (model, params, metrics)."""
    Xtr, ytr = build_xy(train_frame)
    Xval, yval = build_xy(val_frame)
    model, params, yv, _ = TR._tune_and_fit(model_name, Xtr, ytr, Xval, yval, Xval)
    return model, params, EV.evaluate(yval, yv)

def _evaluate_on(model, model_name: str, frame: pd.DataFrame) -> dict:
    """Evaluation metrics for a model on a fully-built window."""
    X, y = build_xy(frame)
    return EV.evaluate(y, DR._predict(model, X, model_name))

def quality_gate(candidate: dict, champion: dict,
                 min_improvement: float | None = None) -> dict:
    """Promote candidate only if it beats champion on the fresh window (MAE gain, RMSE tolerance)."""
    min_imp = C.MIN_PROMOTION_IMPROVEMENT if min_improvement is None else min_improvement
    rel_mae = (champion["MAE"] - candidate["MAE"]) / champion["MAE"]
    rel_rmse = (champion["RMSE"] - candidate["RMSE"]) / champion["RMSE"]
    mae_ok, rmse_ok = rel_mae >= min_imp, rel_rmse >= -C.MAX_REGRESSION_RMSE
    if mae_ok and rmse_ok:
        reason = (f"MAE improved {rel_mae*100:+.2f}% (>= {min_imp*100:.1f}%) "
                  f"and RMSE {rel_rmse*100:+.2f}% within tolerance")
    else:
        f_ = "MAE" if not mae_ok else "RMSE"
        reason = (f"{f_} gate not met: MAE {rel_mae*100:+.2f}%, "
                  f"RMSE {rel_rmse*100:+.2f}% (required MAE >= {min_imp*100:.1f}%)")
    return {"promoted": bool(mae_ok and rmse_ok), "rel_mae_pct": round(rel_mae * 100, 3),
            "rel_rmse_pct": round(rel_rmse * 100, 3),
            "rel_r2": round(candidate["R2"] - champion["R2"], 4), "reason": reason}

# --- Orchestrator ---

def _range_str(frame: pd.DataFrame) -> list[str]:
    """[start, end] date strings of a window (for logs/metadata)."""
    d = pd.to_datetime(frame[C.DATE_COL])
    return [str(d.min().date()), str(d.max().date())]

def run_continual(model_name: str = C.CONTINUAL_MODEL, *,
                  initial_days: int = C.INITIAL_WINDOW_DAYS,
                  ckpt_days: int = C.CHECKPOINT_WINDOW_DAYS,
                  max_windows: int | None = None,
                  subset: float = 1.0,
                  min_improvement: float | None = None) -> dict:
    """Seed a v1 champion, then per window detect drift and gate challengers."""
    df = D.load_target()
    events = D.load_events()
    if subset < 1.0:
        df = df.sample(frac=subset, random_state=C.RANDOM_STATE).sort_index()
    anchor = pd.to_datetime(df[C.DATE_COL]).min()
    windows = split_windows(df, initial_days, ckpt_days)
    if max_windows is not None:
        windows = windows[: max_windows + 1]  # +1 keeps the initial window
    print(f"Simulated arrival: {len(windows)} windows "
          f"({len(windows) - 1} checkpoints)")

    registry = load_registry()
    champion_versions = []

    # Seed the v1 champion from the initial window
    w0 = build_window(windows[0], events, anchor)
    static_map = extract_static_map(w0)  # frozen for the system lifetime
    lo, hi = float(w0["Risk_Score"].min()), float(w0["Risk_Score"].max())
    w0 = normalize_fixed(w0, lo, hi)
    tr0, va0 = _chrono_split(w0, C.CONTINUAL_VAL_FRACTION)
    t0 = time.time()
    champion, params, val_met = _fit_challenger(model_name, tr0, va0)
    version = _deploy_model(registry, champion, model_name, {
        "checkpoint": 0, "role": "initial",
        "train_range": _range_str(tr0), "val_range": _range_str(va0),
        "fresh_window_range": _range_str(w0), "train_samples": int(len(tr0)),
        "hyperparams": TR._as_serializable(params),
        "metrics": {"val": val_met, "fresh": {}},
        "decision": "promoted", "reason": "initial champion (seed model)",
        "drift": {"severity": "none", "triggers": []},
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fit_seconds": round(time.time() - t0, 2)})
    champion_versions.append(version)
    _append_log(C.EVOLUTION_LOG, {"event": "init", "model": model_name,
                                  "version": version, "train_samples": int(len(tr0)),
                                  "train_range": _range_str(w0), "metrics_val": val_met})
    print(f"[init] {model_name} {version} val={val_met} (seed)")

    # Checkpoint loop: detect drift, retrain, gate promotion
    training_pool = [w0]
    summary = {"initial_bounds": {"lo": lo, "hi": hi}, "versions": [{
        "version": version, "checkpoint": 0, "role": "initial", "metrics": val_met,
        "improvement_vs_champion": None, "decision": "promoted"}]}

    for i, window in enumerate(windows[1:], start=1):
        t_c = time.time()
        fresh = normalize_fixed(build_window(window, events, anchor, static_map), lo, hi)
        cur_X, cur_y = build_xy(fresh)
        ref = pd.concat(training_pool).reset_index(drop=True)
        ref_X, ref_y = build_xy(ref)
        w_rng = _range_str(fresh)
        report = DR.compose_drift_report(
            checkpoint=i, window_start=str(pd.to_datetime(fresh[C.DATE_COL]).min()),
            window_end=str(pd.to_datetime(fresh[C.DATE_COL]).max()),
            ref_X=ref_X, cur_X=cur_X, ref_y=ref_y, cur_y=cur_y,
            feature_names=C.FEATURE_COLS, champion=champion, model_name=model_name)
        drift_info = {"severity": report["verdict"]["severity"], "triggers": report["triggers"]}
        _append_log(C.DRIFT_LOG, report)
        print(f"[ckpt {i}] {report['window_start']} -> {report['window_end']} "
              f"severity={drift_info['severity']} triggers={len(drift_info['triggers'])}")

        if not DR.should_retrain(report, i):
            _append_log(C.EVOLUTION_LOG, {"event": "skip", "checkpoint": i,
                                          "model": model_name, "window_range": w_rng,
                                          "severity": drift_info["severity"],
                                          "reason": "no drift, cadence not reached"})
            print(f"  skip: no update (severity={drift_info['severity']})")
            continue

        training_pool.append(fresh)
        train_all = pd.concat(training_pool).reset_index(drop=True)
        tr_rng = _range_str(train_all)
        t_tr, va = _chrono_split(train_all, C.CONTINUAL_VAL_FRACTION)
        candidate, params, cand_val = _fit_challenger(model_name, t_tr, va)
        champ_fresh = _evaluate_on(champion, model_name, fresh)
        cand_fresh = _evaluate_on(candidate, model_name, fresh)
        gate = quality_gate(cand_fresh, champ_fresh, min_improvement=min_improvement)
        log = {"checkpoint": i, "model": model_name, "train_range": tr_rng,
               "fresh_window_range": w_rng, "train_samples": int(len(train_all)),
               "candidate_metrics_fresh": cand_fresh, "champion_metrics_fresh": champ_fresh,
               "gate": gate, "drift": drift_info, "champion_before": champion_versions[-1]}

        if gate["promoted"]:
            version = _deploy_model(registry, candidate, model_name, {
                "checkpoint": i, "role": "promoted",
                "train_range": tr_rng, "fresh_window_range": w_rng,
                "train_samples": int(len(train_all)),
                "hyperparams": TR._as_serializable(params),
                "metrics": {"val": cand_val, "fresh": cand_fresh},
                "improvement_vs_champion": gate,
                "decision": "promoted", "reason": gate["reason"],
                "drift": drift_info,
                "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fit_seconds": round(time.time() - t_c, 2)})
            champion = candidate
            champion_versions.append(version)
            summary["versions"].append({"version": version, "checkpoint": i,
                                        "role": "promoted", "metrics": cand_fresh,
                                        "improvement_vs_champion": gate, "decision": "promoted"})
            _append_log(C.EVOLUTION_LOG, {"event": "promote", "version": version, **log})
            print(f"  PROMOTE {version} fresh={cand_fresh} gate={gate['reason']} "
                  f"(+{time.time() - t_c:.1f}s)")
        else:
            _append_log(C.EVOLUTION_LOG, {"event": "reject", "candidate_version": None, **log})
            summary["versions"].append({"version": None, "checkpoint": i, "role": "rejected",
                                        "metrics": cand_fresh, "improvement_vs_champion": gate,
                                        "decision": "rejected"})
            print(f"  keep {champion_versions[-1]}: {gate['reason']}")

    registry["models"][model_name]["champion_versions"] = champion_versions
    save_registry(registry)
    summary["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    C.CONTINUAL_SUMMARY.write_text(json.dumps(summary, indent=2))
    return {"model_name": model_name, "champion_versions": champion_versions,
            "registry": C.CONTINUAL_REGISTRY, "evolution_log": C.EVOLUTION_LOG,
            "drift_log": C.DRIFT_LOG, "versions": summary["versions"]}

def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse flags and kick off a continual run."""
    ap = argparse.ArgumentParser(description="Continual learning / drift adaptation")
    ap.add_argument("--model", default=C.CONTINUAL_MODEL)
    ap.add_argument("--initial-days", type=int, default=C.INITIAL_WINDOW_DAYS)
    ap.add_argument("--ckpt-days", type=int, default=C.CHECKPOINT_WINDOW_DAYS)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--min-improvement", type=float, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.initial_days, args.ckpt_days, args.max_windows = 3000, 2000, 2
        subset = 0.2
    else:
        subset = 1.0
    result = run_continual(args.model, initial_days=args.initial_days,
                           ckpt_days=args.ckpt_days, max_windows=args.max_windows,
                           subset=subset, min_improvement=args.min_improvement)
    print(f"\nChampion versions: {result['champion_versions']}")
    print(f"Registry:     {C.CONTINUAL_REGISTRY}")
    print(f"Evolution:    {C.EVOLUTION_LOG}")
    print(f"Drift log:    {C.DRIFT_LOG}")

if __name__ == "__main__":
    main()
