import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validation5", ROOT / "analysis" / "05_validation_member5.py")
v5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v5)


def _cohort():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "default": [0, 1, 0, 1],
                       "issue_date": ["2017-06-01", "2018-01-01", "2018-02-01", "2018-03-01"]})
    return df.assign(issue_dt=pd.to_datetime(df["issue_date"]), id_key=v5._norm_id(df["id"]))


def _preds(**changes):
    df = pd.DataFrame({"id": [2, 3, 4], "issue_date": ["2018-01-01", "2018-02-01", "2018-03-01"],
                       "default": [1, 0, 1], "predicted_probability": [0.7, 0.2, 0.6],
                       "model": "m", "selected_final_model": True})
    for key, value in changes.items():
        df[key] = value
    return df


def test_valid_predictions_pass():
    assert len(v5.verify_predictions(_cohort(), _preds(), "2018-01-01")) == 3


@pytest.mark.parametrize("changes", [
    {"id": [2, 2, 4]},                                   # duplicate ID
    {"id": [1, 3, 4]},                                   # training loan included
    {"default": [0, 0, 1]},                              # outcome mismatch
    {"default": ["x", 0, 1]},                            # non-binary outcome
    {"predicted_probability": [1.2, 0.2, 0.6]},          # invalid probability
    {"selected_final_model": False},                     # no selected model
    {"selected_final_model": "maybe"},                   # unparseable flag
    {"issue_date": ["2018-01-01", "2018-05-01", "2018-03-01"]},  # date mismatch
])
def test_invalid_predictions_fail(changes):
    with pytest.raises(ValueError):
        v5.verify_predictions(_cohort(), _preds(**changes), "2018-01-01")


def test_threshold_stats():
    s = v5._threshold_stats(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.1, 0.2]), 0.5, 5.0)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1)
    assert s["expected_cost"] == pytest.approx(6 / 4)
    assert s["approved_default_rate"] == pytest.approx(0.5)


def test_calibration_recovers_ideal_values():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 20_000)
    c = v5.calibration_stats(rng.binomial(1, p), p)
    assert abs(c["calibration_intercept"]) < 0.05
    assert abs(c["calibration_slope"] - 1) < 0.1