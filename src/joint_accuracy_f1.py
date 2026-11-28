"""One origination-only challenger and a joint default-F1/accuracy operating rule."""
import numpy as np
import pandas as pd

from src.predictive_extension import ExtendedEncoding, features
from src.predictive_modelling import TrainOnlyPreprocessor

SUBGRADE_RANK = {f"{letter}{level}": 5 * i + level
                 for i, letter in enumerate("ABCDEFG") for level in range(1, 6)}


def credit_ratios(frame):
    balance = frame.revol_bal.where(frame.revol_bal.ge(0))
    # Utilization is in percentage points. Rounding in the source makes the limit approximate.
    return pd.DataFrame({
        "revolving_balance_to_income": balance / frame.annual_inc.where(frame.annual_inc.gt(0)),
        "loan_to_revolving_balance": frame.loan_amnt / balance.where(balance.gt(0)),
        "implied_revolving_limit": 100 * balance / frame.revol_util.where(frame.revol_util.gt(0)),
    }, index=frame.index)


class OrdinalCreditEncoder:
    def fit(self, frame):
        engineered = features(frame)
        prep = TrainOnlyPreprocessor().fit(engineered)
        self.base = ExtendedEncoding().fit(prep, engineered, "sub_grade")
        self.rank_index = self.base.columns.index("sub_grade")
        self.columns = [*self.base.columns, *credit_ratios(frame.iloc[:0]).columns]
        self.columns[self.rank_index] = "subgrade_rank"
        self.mask = [*self.base.mask, False, False, False]
        self.mask[self.rank_index] = False
        return self

    def transform(self, frame):
        values = self.base.transform(features(frame))
        values[:, self.rank_index] = frame.sub_grade.map(SUBGRADE_RANK).to_numpy(dtype="float32")
        return np.column_stack((values, credit_ratios(frame).to_numpy(dtype="float32")))


def operating_curve(y, p):
    """All distinct inclusive score cutoffs, including predicting no defaults."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Expected aligned scores and both binary outcome classes")
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Expected finite probability scores in [0, 1]")
    order = np.argsort(-p, kind="stable")
    scores = p[order]
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    flagged = np.r_[0, ends + 1]
    tp = np.r_[0, np.cumsum(y[order], dtype=int)[ends]]
    tn = len(y) - y.sum() - (flagged - tp)
    return pd.DataFrame({
        "threshold": np.r_[np.nextafter(scores[0], np.inf), scores[ends]],
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * tp / (flagged + y.sum()),
    })


def joint_rule(curve, reference):
    """Maximize the smaller absolute metric gain. Break ties by F1, then accuracy."""
    gains = np.minimum(curve.accuracy - reference["accuracy"], curve.f1 - reference["f1"])
    return (curve.assign(min_gain=gains)
            .sort_values(["min_gain", "f1", "accuracy", "threshold"], ascending=False)
            .iloc[0])


def paired_intervals(y, old, new, draws=5000):
    """Paired individual-loan bootstrap via the eight possible (label, old, new) cells."""
    cells = np.bincount(4 * np.asarray(y, int) + 2 * np.asarray(old, int) + np.asarray(new, int),
                        minlength=8)
    samples = np.random.default_rng(5106).multinomial(cells.sum(), cells / cells.sum(), size=draws)
    joint = samples.reshape(-1, 2, 2, 2)

    def metrics(cm):
        tp = cm[:, 1, 1]
        return np.array([(cm[:, 0, 0] + tp) / cells.sum(),
                         2 * tp / np.maximum(1, cm[:, 1, :].sum(axis=1) + cm[:, :, 1].sum(axis=1))])

    delta = metrics(joint.sum(axis=2)) - metrics(joint.sum(axis=3))
    return np.quantile(delta, [.025, .975], axis=1).T
