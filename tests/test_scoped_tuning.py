import numpy as np
import pandas as pd
import pytest
from src.scoped_tuning import CANDIDATES, training_folds, rank_candidates, cohort_signature
from src.lending_club_data_cleaning import LEAKAGE_FIELDS
from src.predictive_extension import READ_COLUMNS


def test_training_folds_use_disjoint_forward_months_and_inner_stopping():
    dates = np.repeat(pd.date_range('2010-01-01', periods=40, freq='MS'), 20)
    frame = pd.DataFrame({'id': range(len(dates)), 'issue_date': dates, 'default': np.arange(len(dates))%2})
    for _, fit, threshold, score, early, stop in training_folds(frame):
        assert fit.issue_date.max() < threshold.issue_date.min()
        assert threshold.issue_date.max() < score.issue_date.min()
        assert early.issue_date.max() < stop.issue_date.min()
        assert set(early.id)|set(stop.id) == set(fit.id)
        assert len(set(fit.id)|set(threshold.id)|set(score.id)) == len(fit)+len(threshold)+len(score)


def test_bounded_specs_have_no_new_raw_sources_or_combined_proxies():
    assert len(CANDIDATES)==6 and len({s['name'] for s in CANDIDATES})==6
    assert all(s['proxy'] in {'grade','sub_grade'} for s in CANDIDATES)
    assert not LEAKAGE_FIELDS.intersection(READ_COLUMNS)


def test_ranking_requires_complete_folds_and_uses_f1_before_ap():
    rows = [{'candidate': name, 'fold': fold, 'f1': f1, 'pr_auc': ap, 'log_loss': .5, 'rounds': 100}
            for name, f1, ap in [('a', .4, .8), ('b', .5, .3)] for fold in ['first','second']]
    assert rank_candidates(pd.DataFrame(rows)).iloc[0].candidate=='b'
    with pytest.raises(ValueError):rank_candidates(pd.DataFrame(rows[:-1]))
    f=pd.DataFrame({'id':[1,2], 'issue_date':['2015-01-01']*2, 'default':[0,1]})
    assert cohort_signature(f)!=cohort_signature(f.assign(default=[1,0]))
