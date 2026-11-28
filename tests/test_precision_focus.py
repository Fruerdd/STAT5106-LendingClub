import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from src.precision_focus import precision_at_recall_floor


@pytest.mark.parametrize("floor", [.25, .6, 1.])
def test_precision_cutoff_matches_exhaustive_search_with_ties(floor):
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([.4, .8, .4, .5, .7, .7, .5, .2])
    candidates = []
    for threshold in np.unique(scores):
        flags = scores >= threshold
        recall = recall_score(y, flags)
        if recall >= floor:
            candidates.append((precision_score(y, flags), recall, threshold))
    assert precision_at_recall_floor(y, scores, floor) == max(candidates)[2]
