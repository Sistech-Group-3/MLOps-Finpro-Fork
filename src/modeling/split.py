from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def temporal_split(df: pd.DataFrame):
    """Split a frame chronologically into (train, val, test) by its Date column."""
    date = df[C.DATE_COL]
    span = date.max() - date.min()
    train_end = date.min() + span * (1.0 - (C.VAL_FRACTION + C.TEST_FRACTION))
    val_end = date.min() + span * (1.0 - C.TEST_FRACTION)

    train_idx = date <= train_end
    val_idx = (date > train_end) & (date <= val_end)
    test_idx = date > val_end
    return df[train_idx].copy(), df[val_idx].copy(), df[test_idx].copy()


def _normalize(df: pd.DataFrame, lo: float, hi: float):
    scale = hi - lo
    arr = df["Risk_Score"].to_numpy(dtype=float)
    df = df.copy()
    df["Normalized_Risk_Score"] = np.clip(1.0 + (arr - lo) / scale * 99.0, 1.0, 100.0)
    return df


def prepare_targets(df: pd.DataFrame):
    """Temporally split and re-normalise the raw target using train-only statistics.

    Returns ``(train, val, test, {"lo": float, "hi": float})``.
    """
    train, val, test = temporal_split(df)
    lo = float(train["Risk_Score"].min())
    hi = float(train["Risk_Score"].max())
    if hi - lo < 1e-12:
        raise ValueError("Constant raw target in train split; cannot normalise.")
    return _normalize(train, lo, hi), _normalize(val, lo, hi), _normalize(test, lo, hi), {
        "lo": lo, "hi": hi}


def build_xy(df: pd.DataFrame):
    """Split a frame into feature array and target array."""
    X = df[C.FEATURE_COLS].to_numpy(dtype=float)
    y = df[C.TARGET_COL].to_numpy(dtype=float)
    return X, y