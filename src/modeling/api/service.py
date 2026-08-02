from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config as C
from .. import data as D
from .. import baseline as BL
from .. import registry as REG


class RiskService:
    def __init__(self, assets_path=C.ARTIFACT_DIR / "serving_assets.json"):
        self.assets = json.loads(Path(assets_path).read_text())
        self.events = D.load_events()
        self.registry = REG.load_registry()
        self.models = self._load_active_models()

    def _load_active_models(self) -> dict:
        loaded = {}
        for name in C.MODEL_NAMES:
            if name == "baseline":
                continue
            try:
                obj, meta = REG.load_active(self.registry, name)
                if obj is not None:
                    loaded[name] = {"model": obj, "meta": meta}
            except FileNotFoundError:
                continue
        return loaded

    def predict_all(self, lat: float, lon: float, when) -> dict[str, tuple]:
        x = D.serving_feature(self.assets, self.events, lat, lon, when)

        results = {}
        raw = BL.risk_score_geo(lat, lon, when, self.events)
        nz = self.assets["normalizer"]
        base = float(np.clip(1.0 + 99.0 * (raw - nz["lo"]) / (nz["hi"] - nz["lo"] + 1e-12),
                             1.0, 100.0))
        results["baseline"] = (base, None)

        for name, entry in self.models.items():
            model = entry["model"]
            if name == "XGBRegressor":
                import xgboost as xgb
                y = model.predict(xgb.DMatrix(x))[0]
            else:
                y = model.predict(x)[0]
            results[name] = (float(np.clip(y, 0.0, 100.0)),
                             entry["meta"].get("version", "?"))
        return results

    def default_model(self) -> str:
        return self.assets.get("default_model", "XGBRegressor")

    def list_models(self) -> list[dict]:
        info = []
        for name in C.MODEL_NAMES:
            model_meta = self.registry.get("models", {}).get(name, {})
            versions = [v["version"] for v in model_meta.get("versions", [])]
            active = model_meta.get("active_version")
            last = {}
            if active:
                entry = next((v for v in model_meta.get("versions", [])
                              if v["version"] == active), {})
                last = entry.get("metrics", {}).get("test", {})
            info.append({"name": name, "active_version": active or "-",
                         "versions": versions, "last_metrics": last})
        return info