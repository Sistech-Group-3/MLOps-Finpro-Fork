import numpy as np
import pandas as pd
import pytest

from modeling import config as C
from modeling import drift as DR


def test_psi_identical_distribution_is_small():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 10_000)
    b = rng.normal(0, 1, 10_000)
    assert DR.psi(a, b) < C.PSI_WARN


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 10_000)
    b = rng.normal(1.5, 2.0, 10_000)
    assert DR.psi(a, b) > C.PSI_HIGH


def test_psi_constant_input_is_zero():
    a = np.zeros(100)
    b = np.zeros(100)
    assert DR.psi(a, b) == 0.0


def test_ks_pvalue_rejects_shift():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 5_000)
    b = rng.normal(2, 1, 5_000)
    assert DR.ks_pvalue(a, b) < C.KS_ALPHA


def test_feature_psi_skips_calendar_features():
    rng = np.random.default_rng(1)
    n = 1_000
    names = ["year", "days_since_start", "risk_90d", "month_sin"]
    ref = np.column_stack([rng.uniform(2001, 2010, n), rng.uniform(0, 4000, n),
                           rng.normal(10, 1, n), rng.normal(0, 1, n)])
    cur = np.column_stack([rng.uniform(2015, 2025, n), rng.uniform(5000, 9000, n),
                           rng.normal(30, 5, n), rng.normal(0, 1, n)])
    fpsi = DR.feature_psi(ref, cur, names)
    assert "year" not in fpsi
    assert "days_since_start" not in fpsi
    assert fpsi["risk_90d"] > C.PSI_HIGH  # the genuine signal is still caught
    assert fpsi["month_sin"] < C.PSI_WARN


def test_compose_drift_report_verdict_none_when_stable():
    rng = np.random.default_rng(3)
    ref_X = rng.normal(0, 1, (2_000, 3))
    cur_X = rng.normal(0, 1, (2_000, 3))
    ref_y = rng.normal(50, 5, 2_000)
    cur_y = rng.normal(50, 5, 2_000)
    report = DR.compose_drift_report(checkpoint=1, window_start="2020-01-01",
                                     window_end="2020-06-01",
                                     ref_X=ref_X, cur_X=cur_X,
                                     ref_y=ref_y, cur_y=cur_y,
                                     feature_names=["f0", "f1", "f2"])
    assert report["verdict"]["severity"] == "none"
    assert report["verdict"]["drift"] is False


def test_compose_drift_report_verdict_high_on_shift():
    rng = np.random.default_rng(3)
    ref_X = rng.normal(0, 1, (2_000, 3))
    cur_X = rng.normal(2.0, 1.5, (2_000, 3))
    ref_y = rng.normal(50, 5, 2_000)
    cur_y = rng.normal(60, 8, 2_000)
    report = DR.compose_drift_report(checkpoint=1, window_start="2020-01-01",
                                     window_end="2020-06-01",
                                     ref_X=ref_X, cur_X=cur_X,
                                     ref_y=ref_y, cur_y=cur_y,
                                     feature_names=["f0", "f1", "f2"])
    assert report["verdict"]["severity"] == "high"
    assert report["verdict"]["drift"] is True
    assert report["features_high"]  # at least one feature exceeds PSI_HIGH


def test_should_retrain_on_drift_and_cadence():
    report = {"verdict": {"drift": True}}
    assert DR.should_retrain(report, checkpoint_index=1) is True
    assert DR.should_retrain({"verdict": {"drift": False}}, checkpoint_index=1) is False
    # scheduled cadence forces a refresh even without drift
    assert DR.should_retrain({"verdict": {"drift": False}},
                             checkpoint_index=C.FORCE_RETRAIN_EVERY) is True
