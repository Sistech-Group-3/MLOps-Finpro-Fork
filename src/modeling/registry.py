from __future__ import annotations

import csv
import datetime as _dt
import json

import joblib

from . import config as C


def _now_iso():
    return _dt.datetime.now().isoformat(timespec="seconds")


def load_registry() -> dict:
    if C.REGISTRY_FILE.exists():
        return json.loads(C.REGISTRY_FILE.read_text())
    return {"updated": _now_iso(), "models": {}}


def save_registry(registry: dict) -> None:
    C.ensure_dirs()
    registry["updated"] = _now_iso()
    C.REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def next_version(registry: dict, model_name: str) -> str:
    """Return the next monotonically increasing version label for a model."""
    history = registry.get("models", {}).get(model_name, {}).get("versions", [])
    highest = 1
    for item in history:
        try:
            highest = max(highest, int(item.get("version", "v0").lstrip("v")))
        except ValueError:
            continue
    return f"v{highest + 1}"


def promote_active(registry: dict, model_name: str, version: str) -> None:
    model = registry["models"].setdefault(model_name, {})
    model["active_version"] = version
    save_registry(registry)


def version_dir(model_name: str, version: str):
    d = C.MODELS_DIR / model_name / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model_artifact(model, model_name: str, version: str):
    """Persist a sklearn model to a versioned folder; returns its absolute path."""
    d = version_dir(model_name, version)
    p = d / "model.joblib"
    joblib.dump(model, p)
    return str(p)


def save_xgboost(booster, model_name: str, version: str):
    """Persist an XGBoost booster with its native serialiser (non-pickle)."""
    d = version_dir(model_name, version)
    p = d / "model.ubj"
    booster.save_model(str(p))
    return str(p)


def write_metadata(model_name: str, version: str, meta: dict) -> None:
    d = version_dir(model_name, version)
    meta = {"model_name": model_name, "version": version, **meta}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))


def register_version(registry: dict, model_name: str, version: str,
                     meta: dict, promote: bool = True) -> dict:
    model = registry.setdefault("models", {}).setdefault(model_name, {})
    history = model.setdefault("versions", [])
    entry = {"version": version, "saved_at": _now_iso(), **meta}
    history.append(entry)
    if promote:
        model["active_version"] = version
    save_registry(registry)
    return entry


def load_model_artifact(model_name: str, version: str):
    """Load and return the model object for a given name/version."""
    d = C.MODELS_DIR / model_name / version
    if (d / "model.ubj").exists():
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(d / "model.ubj"))
        return booster
    return joblib.load(d / "model.joblib")


def load_active(registry: dict, model_name: str):
    """Return (model_object_or_None, active_meta_or_None) for a model."""
    model = registry.get("models", {}).get(model_name, {})
    version = model.get("active_version")
    if not version:
        return None, None
    entry = next((m for m in model.get("versions", [])
                  if m["version"] == version), None)
    try:
        obj = load_model_artifact(model_name, version)
    except FileNotFoundError:
        obj = None
    return obj, entry or {}


_metrics_fields = ["model", "version", "saved_at"]


def log_metrics_ledger(model_name: str, version: str, metrics: dict) -> None:
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.METRICS_CSV
    new = not path.exists()
    fields = _metrics_fields + [k for k in metrics.keys() if k not in _metrics_fields]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"model": model_name, "version": version,
                    "saved_at": _now_iso(), **metrics})


def read_metrics_ledger():
    if not C.METRICS_CSV.exists():
        return []
    with open(C.METRICS_CSV) as f:
        return list(csv.DictReader(f))