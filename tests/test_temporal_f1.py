import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.temporal_f1 import f1_rule, loan_terms, recency_weights


def test_month_weights_preserve_total_and_use_12_month_half_life():
    dates = ["2015-01-01", "2014-01-01", "2013-01-01", "2015-01-29"]
    w = recency_weights(dates)
    assert w.sum() == pytest.approx(4)
    np.testing.assert_allclose(w / w[0], [1., .5, .25, 1.])
    np.testing.assert_allclose(recency_weights(dates[::-1]), w[::-1])
    with pytest.raises(ValueError):
        recency_weights([None])


def test_loan_terms_use_percentage_point_dti_and_keep_undefined_ratios_missing():
    raw = pd.DataFrame({"annual_inc": [60000., 0., 24000.], "dti": [20., 30., 0.],
                        "installment": [250., 300., 120.], "loan_amnt": [3000., 2750., np.nan],
                        "revol_bal": [2400., 0., np.nan], "open_acc": [3., 0., 2.]})
    expected = [[1000., .25, 800., 1.], [np.nan, np.nan, np.nan, 0.], [0., np.nan, np.nan, np.nan]]
    np.testing.assert_allclose(loan_terms(raw), expected, equal_nan=True)
    pd.testing.assert_frame_equal(loan_terms(raw), loan_terms(raw.assign(default=1, issue_date="2015-12-01")))


def test_maximum_default_f1_above_accuracy_floor_handles_score_ties():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    p = np.array([.7, .2, .2, .1, .8, .7, .4, .7])
    thresholds = [*np.unique(p), np.nextafter(p.max(), np.inf)]
    for floor in (0., .5, .7):
        eligible = [(f1_score(y, p >= t), accuracy_score(y, p >= t), t) for t in thresholds
                    if accuracy_score(y, p >= t) >= floor]
        best = f1_rule(y, p, floor)
        assert (best.f1, best.accuracy, best.threshold) == pytest.approx(max(eligible))
    with pytest.raises(ValueError):
        f1_rule(y, p, 1.)
