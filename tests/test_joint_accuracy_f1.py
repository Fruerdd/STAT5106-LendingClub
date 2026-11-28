import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.joint_accuracy_f1 import OrdinalCreditEncoder, joint_rule, operating_curve, paired_intervals
from tests.test_predictive_improvement import extension_frame


def test_all_joint_thresholds_match_independent_metrics_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0., .2, .2, .8, 1., 1.])
    baseline = {"accuracy": .5, "f1": .4}
    curve = operating_curve(y, scores)
    assert len(curve) == len(np.unique(scores)) + 1
    gains = []
    for row in curve.itertuples():
        flags = scores >= row.threshold
        accuracy = accuracy_score(y, flags)
        f1 = f1_score(y, flags)
        assert (row.accuracy, row.f1) == pytest.approx((accuracy, f1))
        gains.append(min(accuracy - baseline["accuracy"], f1 - baseline["f1"]))
    assert joint_rule(curve, baseline).min_gain == pytest.approx(max(gains))
    assert joint_rule(curve, {"accuracy": 1., "f1": 1.}).min_gain <= 0


def test_ordered_subgrade_and_ratios_depend_only_on_registered_predictors():
    # The retained preprocessing groups categories with fewer than 1,000 training rows.
    train = pd.concat([extension_frame()] * 8, ignore_index=True)
    enc = OrdinalCreditEncoder().fit(train)
    cases = train.iloc[:4].copy().assign(sub_grade=["A1", "B1", "G5", "UNKNOWN"],
                                         revol_bal=[1800., 0., 0., 75.], annual_inc=[18000., 0., 100., 75.],
                                         revol_util=[25., 0., np.nan, 100.])
    result = enc.transform(cases)
    columns = enc.columns
    assert len(columns) == len(enc.mask) == result.shape[1] == 30
    assert not {"grade", "sub_grade", "int_rate_per_5", "issue_date", "id", "default"}.intersection(columns)
    assert not enc.mask[columns.index("subgrade_rank")]
    assert np.allclose(result[:, columns.index("subgrade_rank")], [1, 6, 35, np.nan], equal_nan=True)
    assert result[0, columns.index("implied_revolving_limit")] == pytest.approx(7200.)
    assert result[0, columns.index("revolving_balance_to_income")] == pytest.approx(.1)
    assert np.isnan(result[1, -3:]).all()
    assert not np.isinf(result).any()
    changed = cases.assign(default=1 - cases.default, grade="A", int_rate=55., id="unused")
    np.testing.assert_allclose(result, enc.transform(changed), equal_nan=True)


def test_paired_bootstrap_order_and_identical_predictions():
    y = np.tile([0, 0, 1, 1], 100)
    wrong = 1 - y
    np.testing.assert_array_equal(paired_intervals(y, wrong, wrong), np.zeros((2, 2)))
    np.testing.assert_array_equal(paired_intervals(y, wrong, y), np.ones((2, 2)))
