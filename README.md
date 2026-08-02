# Chicago Safe Route Recommendation & Crime Risk Score

> **Note:** `chicago_walk_graph.joblib` is not committed due to GitHub's file-size
> limit. Download it by running `src/notebooks/chicago_map.ipynb`. Until it exists,
> the `/route/*` endpoints return 503 but the rest of the service stays healthy.

## What it does

An MLOps platform for Chicago that predicts a **crime risk score (0–100) at any
location & time** and recommends the **safest walking route** between two points,
using historical crime data (2001–2024). Risk is computed from recency-weighted
(exponential / half-life decay) and spatially-weighted (kernel) statistics.

## Prediction & serving stack

- **Baseline** (`baseline.py`): deterministic, non-learned formula
  `Σ Severity × exp(-λ·Δt) × 1/(1+γ·d)` for a transparent reference point.
- **ML regressors** (`train.py`): Linear Regression, Decision Tree, Random Forest,
  XGBoost fit on a shared chronological split against pseudo-labelled risk scores,
  tuned by validation RMSE and reported with MAE / RMSE / R² + % improvement vs
  baseline.
- **Versioning & registry** (`registry.py`): every trained model is stored as a
  versioned artifact (`artifacts/models/<name>/<v>/`) with metadata; `registry.json`
  tracks the active version. `serving_assets.json` bridges training → serving
  (cluster centres, static features, normalizer bounds, default/best model).
- **Serving** (`api/app.py`): FastAPI + uvicorn. Endpoints: `/health`,
  `/risk-score`, `/risk-score/compare`, `/risk-score/level-buckets`, `/models`,
  `/models/{name}/history`, `/logs/recent`, and `/route/v1` & `/route/v2`.
  Strict Pydantic validation; every prediction logged to JSONL + SQLite
  (thread-safe) for monitoring.

### Safe route recommendation

- **V1** — K-shortest paths (Yen's) → densify → score → pick min combined risk
  (`α·R_mean + β·R_max`, α=0.7, β=0.3). Simple; but candidates are geometrically
  similar.
- **V2** — `generate_diverse_routes()` uses penalized iterative re-routing to
  force divergent candidates, then safety-first selection
  (`select_safest_then_fastest` / `select_lexicographic`). More diverse, but
  heavier and may pick longer routes.
- Evaluation: the formula is still in experimental and will be update in later checkpoints

## Project structure

```
src/
├── modeling/                  # Prediction, ML, serving
│   ├── api/                   # FastAPI app, schemas, service, persistence
│   ├── baseline.py            # Statistical baseline risk model
│   ├── train.py               # Trains & registers the 4 regressors
│   ├── registry.py            # Versioning, serialization, metrics ledger
│   ├── drift.py               # Continual-learning / drift detection
│   ├── data.py / split.py / evaluate.py / config.py
├── utils/                     # Route recommendation (v1, v2, k-shortest, converter)
├── notebooks/                 # chicago_map, training, api_serving, risk_score
├── graph/chicago_walk_graph.joblib   # OSMnx walking graph (download via notebook)
├── dataset/                   # Processed crime / pseudo-label parquet + csv
└── artifacts/                 # models/<name>/<v>/, registry.json, logs, metrics, report
```

## Run it

```bash
uv install            # or: pip install -e .
python -m modeling.train --subset 0.20   # train + register versioned models
uvicorn modeling.api.app:app --reload    # serve the API
```