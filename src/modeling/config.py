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
    for d in (MODELS_DIR, LOG_DIR, REPORT_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()