"""Central configuration for the risk-score modeling / serving stack.

Every tunable hyperparameter, domain constant, and path used across the package
lives here so the decision logic is transparent and auditable (per brief 3 & 4).
"""
from __future__ import annotations

import os
import pathlib

# Paths
PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE = PACKAGE_DIR.parent

DATA_DIR = WORKSPACE / "dataset"
# Pre-computed (upstream, per brief) datasets produced by the pseudo-labeling
# notebook. `TARGET_FILE` holds one row per (spatial cluster, day) with the
# pseudo-labelled risk score. `EVENTS_FILE` holds sampled raw incidents used both
# for feature engineering and for the statistical (non-learned) baseline.
TARGET_FILE = DATA_DIR / "chicago_risk_pseudo_labels-2.parquet"
EVENTS_FILE = DATA_DIR / "sampled_dataset_engineered.parquet"

# Everything the pipeline writes (models, registry, metrics, logs).
ARTIFACT_DIR = ROOT / "artifacts" if (ROOT := WORKSPACE) else None
ARTIFACT_DIR = WORKSPACE / "artifacts"
MODELS_DIR = ARTIFACT_DIR / "models"
LOG_DIR = ARTIFACT_DIR / "logs"
REPORT_DIR = ARTIFACT_DIR / "report"
METRICS_DIR = ARTIFACT_DIR / "metrics"

REGISTRY_FILE = ARTIFACT_DIR / "registry.json"
PREDICTION_LOG_FILE = LOG_DIR / "prediction_log.jsonl"
PREDICTION_DB_FILE = LOG_DIR / "prediction_logs.sqlite"
METRICS_CSV = METRICS_DIR / "metrics_history.csv"

# Domain constants — Baseline (Section 1)
# These constants are hand-chosen by domain reasoning and are *not* learned.
# They are the hyper-parameters of a fixed-formula statistical baseline.

SEVERITY_WEIGHT = {
    "HOMICIDE": 100.0,
    "ASSAULT": 70.0,
    "BATTERY": 65.0,
    "ROBBERY": 80.0,
    "BURGLARY": 60.0,
    "THEFT": 40.0,
    "MOTOR VEHICLE THEFT": 50.0,
    "NARCOTICS": 45.0,
    "DECEPTIVE PRACTICE": 30.0,
}
SEVERITY_DEFAULT = 25.0  # extremely broad fallback for any unlisted offence/label.

# Exponential time-decay rate (units: 1 / days). exp(-LAMBDA_TIME * dt) halves the
# influence of a crime every ln(2)/LAMBDA_TIME ~ 13.9 days. Chosen so that a crime
# older than ~60 days barely contributes (matching the upstream 30–60 day window).
LAMBDA_TIME = 0.05

# Radiometric spatial decay, applied as 1 / (1 + GAMMA_SPACE * dist_km) so that a
# crime 1 km away contributes 1/2 and 2 km away 1/3 (inverse-distance tail).
GAMMA_SPACE = 1.0

# Neighborhood radius used for spatial aggregation / windowing (km).
RADIUS_KM = 2.0

# Hard backwards time-window for the *baseline* look-back (days).  Beyond this the
# exponential decay is negligible, so we truncate for cheap, stable computation.
BASELINE_WINDOW_DAYS = 60

# Feature engineering
FEATURE_COLS = [
    # temporal (cyclic + level)
    "year", "month", "day", "dow", "days_since_start",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "day_sin", "day_cos",
    "weekend",
    # spatial (cluster centre coordinates, used as a proxy for the Area)
    "Latitude", "Longitude",
    # static neighbourhood aggregates (precomputed once per cluster)
    "near_count", "near_mean_sev", "near_max_sev",
    # trailing temporal aggregates (past crime only — no future / no same-day leak)
    "risk_7d", "risk_30d", "risk_90d",
]

TARGET_COL = "Normalized_Risk_Score"
CLUSTER_COL = "Spatial_Cluster_ID"
DATE_COL = "Date"

# Temporal split (Section 2) — chronological, mimics data arriving over time.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.20

# Model registry / versioning
MODEL_NAMES = ["baseline", "LinearRegression", "DecisionTreeRegressor",
               "RandomForestRegressor", "XGBRegressor"]

# Random seed kept consistent across the 4 learners so the split *and* any
# stochastic model behaviour are reproducible.
RANDOM_STATE = 42

# Level bucketing (documented, fixed thresholds per brief 3).
LEVEL_BUCKETS = [
    (0.0, 25.0, "Low"),
    (25.0, 50.0, "Medium"),
    (50.0, 75.0, "High"),
    (75.0, 100.0, "Very High"),
]

# --------------------------------------------------------------------------- #
# Continual learning / drift adaptation
# --------------------------------------------------------------------------- #
# The continual pipeline writes to its own workspace so the historical
# production registry / artifacts stay untouched and the experiment is
# self-contained (models/<name>/<v>/, registry.json, logs).
CONTINUAL_DIR = ARTIFACT_DIR / "continual"
CONTINUAL_MODELS_DIR = CONTINUAL_DIR / "models"
CONTINUAL_REGISTRY = CONTINUAL_DIR / "registry.json"
EVOLUTION_LOG = CONTINUAL_DIR / "model_evolution.jsonl"
DRIFT_LOG = CONTINUAL_DIR / "drift_report.jsonl"
CONTINUAL_SUMMARY = CONTINUAL_DIR / "continual_summary.json"

# Simulator: new data arrives as chronological windows.
#   * initial window  -> trains the v1 champion
#   * checkpoint windows -> each simulates a freshly arrived batch
INITIAL_WINDOW_DAYS = 3600      # ~ first 10 years -> seed the champion
CHECKPOINT_WINDOW_DAYS = 1400   # ~ 3.8 years per arriving batch

# Chronological split inside every retraining round (tuning vs. training).
CONTINUAL_VAL_FRACTION = 0.20

# Drift detection thresholds (PSI on features/target, KS on target, and the
# champion's prediction-error ratio on the fresh window).
PSI_WARN = 0.10     # feature PSI above this -> flagged
PSI_HIGH = 0.25     # feature PSI above this -> strong / high-severity signal
KS_ALPHA = 0.01     # KS p-value below this -> distributions differ
ERROR_DRIFT_WARN = 1.10   # fresh-window MAE / training MAE above this -> warn
ERROR_DRIFT_HIGH = 1.20   # above this -> high severity

# Features that are deterministic functions of the calendar (linear time) are
# *excluded* from feature-PSI: they drift simply because time advances, not
# because the underlying pattern changed. PSI therefore runs on the stationary
# signal (cyclic temporal + spatial + trailing risk aggregates).
DRIFT_IGNORE_FEATURES = ["year", "days_since_start"]

# Static per-cluster neighbourhood features are computed once on the initial
# window (the deployment-time environment) and frozen for later windows, so the
# trailing risk_* aggregates remain the genuine change signal.
STATIC_FEATURES = ["near_count", "near_mean_sev", "near_max_sev"]

# Update policy / quality gate. A candidate is *promoted* only when it beats the
# reigning champion on the fresh holdout window by at least
# MIN_PROMOTION_IMPROVEMENT relative MAE *and* does not regress RMSE beyond
# MAX_REGRESSION_RMSE. Otherwise the champion stays and the candidate is logged.
MIN_PROMOTION_IMPROVEMENT = 0.005   # 0.5% relative MAE gain
MAX_REGRESSION_RMSE = 0.01          # allow up to 1% RMSE regression on promotion
FORCE_RETRAIN_EVERY = 3             # checkpoint cadence for scheduled refreshes

# Default learner used by the continual pipeline (keeps the demo tractable).
CONTINUAL_MODEL = "XGBRegressor"

def level_for(score: float) -> str:
    """Map a numeric 0-100 risk score to a categorical safety level."""
    for lo, hi, label in LEVEL_BUCKETS:
        if lo <= score < hi:
            return label
    return "Very High" if score >= 75.0 else "Low"


# ------------------------------------------------------------------------- #
# Optional convenience: allow overriding data/artifact roots via env vars so the
# same code runs in a CI or containerised environment.
# ------------------------------------------------------------------------- #
if os.getenv("RISK_DATA_DIR"):
    DATA_DIR = pathlib.Path(os.environ["RISK_DATA_DIR"]) or DATA_DIR
if os.getenv("RISK_ARTIFACT_DIR"):
    ARTIFACT_DIR = pathlib.Path(os.environ["RISK_ARTIFACT_DIR"]) or ARTIFACT_DIR


def ensure_dirs() -> None:
    for d in (MODELS_DIR, LOG_DIR, REPORT_DIR, METRICS_DIR, CONTINUAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()