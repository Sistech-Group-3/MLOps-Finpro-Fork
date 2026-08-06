import json

import numpy as np
import pandas as pd
import pytest

from modeling import config as C
from modeling import continual as CL


def _tiny_frame(n_days=30, n_clusters=3):
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rows = []
    for c in range(n_clusters):
        for d in dates:
            rows.append({"Date": d, "Spatial_Cluster_ID": c, "Risk_Score": 50.0,
                         "Latitude": 41.8 + c * 0.01, "Longitude": -87.7})
    return pd.DataFrame(rows)


def test_split_windows_chronological_and_contiguous():
    df = _tiny_frame(n_days=100)
    windows = CL.split_windows(df, initial_days=30, ckpt_days=30)
    # 30 + 30 + (30+10 merged tail) -> the short remainder is absorbed so no
    # degenerate one-off checkpoint exists
    assert len(windows) == 3
    # contiguous, non-overlapping, ordered
    prev_end = None
    for w in windows:
        d = pd.to_datetime(w[C.DATE_COL])
        assert (d.min() == prev_end) if prev_end is not None else True
        prev_end = d.max() + pd.Timedelta(days=1)
    assert sum(len(w) for w in windows) == 3 * 100


def test_split_windows_keeps_large_tail_as_own_window():
    df = _tiny_frame(n_days=100)
    windows = CL.split_windows(df, initial_days=30, ckpt_days=30, min_tail_frac=0.1)
    assert len(windows) == 4  # 30 + 30 + 30 + 10 (tail >= 10% of ckpt)
    assert sum(len(w) for w in windows) == 3 * 100


def test_normalize_fixed_uses_reference_bounds():
    df = _tiny_frame()
    df.loc[0, "Risk_Score"] = 0.0
    df.loc[1, "Risk_Score"] = 200.0
    out = CL.normalize_fixed(df, lo=0.0, hi=200.0)
    assert out[C.TARGET_COL].min() == pytest.approx(1.0)
    assert out[C.TARGET_COL].max() == pytest.approx(100.0)
    # clipping: values beyond bounds clamp into [1, 100]
    df.loc[2, "Risk_Score"] = 1000.0
    out2 = CL.normalize_fixed(df, lo=0.0, hi=200.0)
    assert out2[C.TARGET_COL].max() <= 100.0


def test_build_window_anchors_days_since_start_and_freezes_static():
    df = _tiny_frame(n_days=60)
    events = df.rename(columns={"Date": "Parsed_Date"})
    events["Severity_Score"] = 5.0
    events["Latitude"] = 41.8
    events["Longitude"] = -87.7
    anchor = pd.to_datetime(df[C.DATE_COL]).min()

    w0 = CL.build_window(df[df[C.DATE_COL] < "2015-01-31"], events, anchor)
    static_map = CL.extract_static_map(w0)
    w1 = CL.build_window(df[df[C.DATE_COL] >= "2015-01-31"], events, anchor,
                         static_map=static_map)
    # days_since_start is globally anchored (not per-window)
    assert w1["days_since_start"].min() > w0["days_since_start"].max()
    # static features are frozen across windows
    for cl, vals in static_map.items():
        for col, v in vals.items():
            assert np.isclose(w1.loc[w1[C.CLUSTER_COL] == cl, col].iloc[0], v)


def test_quality_gate_promotes_only_real_improvement():
    champion = {"MAE": 2.0, "RMSE": 3.0, "R2": 0.90}
    clearly_better = {"MAE": 1.8, "RMSE": 2.7, "R2": 0.92}
    gate = CL.quality_gate(clearly_better, champion)
    assert gate["promoted"] is True

    marginal = {"MAE": 1.995, "RMSE": 2.99, "R2": 0.9001}
    assert CL.quality_gate(marginal, champion)["promoted"] is False

    worse = {"MAE": 2.2, "RMSE": 3.3, "R2": 0.88}
    assert CL.quality_gate(worse, champion)["promoted"] is False


def test_run_continual_end_to_end():
    # conftest.py redirects the artifact workspace to a throwaway temp dir
    result = CL.run_continual("LinearRegression", initial_days=200,
                              ckpt_days=150, max_windows=1, subset=0.02)
    assert result["champion_versions"]  # at least the initial v1
    assert C.CONTINUAL_REGISTRY.exists()
    assert C.EVOLUTION_LOG.exists()
    assert C.DRIFT_LOG.exists()

    registry = CL.load_registry()
    versions = registry["models"][result["model_name"]]["versions"]
    assert versions[0]["version"] == "v1"
    assert versions[0]["role"] == "initial"
    # every registered version carries context
    for v in versions:
        assert "train_range" in v and "drift" in v and "decision" in v

    events = [json.loads(l) for l in open(C.EVOLUTION_LOG)]
    assert events[0]["event"] == "init"
    kinds = {e["event"] for e in events}
    assert kinds <= {"init", "skip", "promote", "reject"}
