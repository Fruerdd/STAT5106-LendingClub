"""Bounded training-only tuning within the registered LendingClub feature extension."""
import numpy as np
import pandas as pd
from src.xgboost_benchmark import make_model, feature_types
from src.predictive_modelling import TrainOnlyPreprocessor
from src.predictive_extension import ExtendedEncoding, features


CANDIDATES = [
    dict(name='grade_depth5_double_reference', proxy='grade', max_depth=5, weight='double', min_child_weight=20, reg_lambda=10., subsample=.85, colsample_bytree=.9),
    dict(name='grade_depth5_unweighted', proxy='grade', max_depth=5, weight='unweighted', min_child_weight=20, reg_lambda=10., subsample=.85, colsample_bytree=.9),
    dict(name='grade_depth3_regularized', proxy='grade', max_depth=3, weight='unweighted', min_child_weight=40, reg_lambda=20., subsample=.9, colsample_bytree=.9),
    dict(name='grade_depth5_regularized', proxy='grade', max_depth=5, weight='unweighted', min_child_weight=40, reg_lambda=20., subsample=.9, colsample_bytree=1.),
    dict(name='grade_depth7_regularized', proxy='grade', max_depth=7, weight='unweighted', min_child_weight=40, reg_lambda=20., subsample=.85, colsample_bytree=.9),
    dict(name='subgrade_depth5_regularized', proxy='sub_grade', max_depth=5, weight='unweighted', min_child_weight=40, reg_lambda=20., subsample=.9, colsample_bytree=1.),
]


def training_folds(train):
    """Whole-month forward folds; callers supply only the frozen outer training 60%."""
    dates = pd.to_datetime(train.issue_date)
    ordered = dates.sort_values().reset_index(drop=True)
    for name, fit_end, threshold_end, score_end in [('forward1', .5, .6, .75), ('forward2', .7, .8, 1.)]:
        a, b = (ordered.iloc[int(q*len(train))] for q in (fit_end, threshold_end))
        c = ordered.iloc[int(score_end*len(train))] if score_end < 1 else dates.max()+pd.Timedelta(days=1)
        fit = train.loc[dates.lt(a)]
        threshold = train.loc[dates.ge(a) & dates.lt(b)]
        score = train.loc[dates.ge(b) & dates.lt(c)]
        fit_dates = pd.to_datetime(fit.issue_date).sort_values().reset_index(drop=True)
        stop_start = fit_dates.iloc[int(.9*len(fit))] if len(fit_dates) else a
        early = fit.loc[pd.to_datetime(fit.issue_date).lt(stop_start)]
        stop = fit.loc[pd.to_datetime(fit.issue_date).ge(stop_start)]
        if any(p.empty or p.default.nunique()!=2 for p in [fit, threshold, score, early, stop]):
            raise ValueError('Forward folds need enough whole months and both classes in every block')
        yield name, fit, threshold, score, early, stop


def encode(train, *others, proxy='grade'):
    f = features(train)
    prep = TrainOnlyPreprocessor().fit(f)
    enc = ExtendedEncoding().fit(prep, f, proxy)
    return enc, [enc.transform(f), *[enc.transform(features(p)) for p in others]]


def model(spec, rounds, mask, y, early_stopping=False):
    result = make_model(spec, rounds, feature_types(mask), y, early_stopping=early_stopping)
    return result.set_params(**{k: spec[k] for k in ['min_child_weight','reg_lambda','subsample','colsample_bytree']})


def cohort_signature(frame):
    import hashlib
    records = frame[['id','issue_date','default']].astype('string').to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(records.encode()).hexdigest()


def rank_candidates(results):
    summary = results.groupby('candidate', as_index=False).agg(
        mean_f1=('f1','mean'), minimum_f1=('f1','min'), mean_pr_auc=('pr_auc','mean'),
        mean_log_loss=('log_loss','mean'), median_rounds=('rounds','median'), folds=('fold','nunique'))
    if not summary.folds.eq(2).all():
        raise ValueError('Both forward folds are required for every candidate')
    return summary.sort_values(['mean_f1','mean_pr_auc','mean_log_loss','candidate'], ascending=[False,False,True,True])
