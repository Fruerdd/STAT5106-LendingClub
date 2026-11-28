"""Precision-focused operating rule for the LendingClub research comparison."""
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.xgboost_benchmark import f1_threshold


def precision_at_recall_floor(y, scores, recall_floor):
    """Maximize precision subject to recall, breaking ties toward more defaults caught."""
    if not 0 < recall_floor <= 1:
        raise ValueError("Recall floor must be in (0, 1]")
    f1_threshold(y, scores)  # Apply the existing score and outcome checks.
    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(recall[:-1] >= recall_floor)
    order = np.lexsort((-thresholds[eligible], -recall[eligible], -precision[eligible]))
    return float(thresholds[eligible[order[0]]])
