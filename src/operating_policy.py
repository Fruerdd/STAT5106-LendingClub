"""Choose operating rules using supplied validation outcomes only."""
import numpy as np
from sklearn.metrics import precision_recall_curve
from src.xgboost_benchmark import f1_threshold
from src.predictive_improvement import threshold_metrics


def validation_options(y, p, minimum_flags=1000):
    y, p = np.asarray(y), np.asarray(p)
    best = f1_threshold(y, p)  # Also validates inputs.
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    sorted_p = np.sort(p)
    flags = len(p)-np.searchsorted(sorted_p, thresholds, side='left')
    rows = [{'rule': 'maximum_f1', 'constraint_met': True, **threshold_metrics(y, p, best)}]
    for metric, target in [('precision', .4), ('precision', .5), ('precision', .6), ('recall', .8)]:
        values = precision if metric == 'precision' else recall
        objective = recall if metric == 'precision' else precision
        candidates = np.flatnonzero((values >= target) & (flags >= minimum_flags))
        row = {'rule': f'{metric}_at_least_{int(target*100)}', 'constraint_met': bool(len(candidates)),
               'minimum_flags': minimum_flags}
        if len(candidates):
            i = candidates[np.argmax(objective[candidates])]
            row.update(threshold_metrics(y, p, float(thresholds[i])))
        rows.append(row)
    return rows
