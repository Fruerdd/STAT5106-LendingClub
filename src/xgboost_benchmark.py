"""XGBoost benchmark helpers; original predictors and validation code remain unchanged."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_curve

CANDIDATES=[{'name':f'xgb_depth{depth}_{weight}','max_depth':depth,'weight':weight}
            for depth in (3,5) for weight in ('unweighted','double','balanced')]


def training_weight(kind,y):
    values=np.asarray(y)
    if not np.isin(values,[0,1]).all() or np.unique(values).size!=2:
        raise ValueError('Training labels must contain both binary outcome classes')
    return {'unweighted':1.,'double':2.,'balanced':float((values==0).sum()/(values==1).sum())}[kind]


def f1_threshold(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if y.size==0 or y.size!=p.size or not np.isin(y,[0,1]).all() or np.unique(y).size!=2:
        raise ValueError('Threshold labels must contain both binary classes and match probabilities')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Threshold probabilities must be finite and in [0,1]')
    precision,recall,thresholds=precision_recall_curve(y,p)
    precision,recall=precision[:-1],recall[:-1]
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=precision+recall>0)
    return float(thresholds[np.argmax(f1)])


def feature_types(mask):
    return ['c' if flag else 'q' for flag in mask]


def make_model(spec,rounds,types,y,seed=5106,early_stopping=False):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError('Install requirements-xgboost.txt; macOS also requires brew install libomp') from exc
    return XGBClassifier(objective='binary:logistic',tree_method='hist',enable_categorical=True,
        feature_types=types,n_estimators=int(rounds),max_depth=spec['max_depth'],learning_rate=.05,
        min_child_weight=20,reg_lambda=10.,reg_alpha=.1,subsample=.85,colsample_bytree=.9,
        max_cat_to_onehot=4,scale_pos_weight=training_weight(spec['weight'],y),
        eval_metric='logloss',early_stopping_rounds=40 if early_stopping else None,
        random_state=seed,n_jobs=2,verbosity=0)


def check_handoff(preds,cohort):
    if preds.id.isna().any() or preds.id.duplicated().any() or set(preds.id)!=set(cohort.id):
        raise ValueError('Cached prediction identities do not match the expected cohort')
    aligned=preds.set_index('id').reindex(cohort.id)
    if not np.array_equal(aligned.default.to_numpy(),cohort.default.to_numpy()):
        raise ValueError('Cached outcomes differ')
    if not np.array_equal(aligned.issue_date.to_numpy(),cohort.issue_date.to_numpy()):
        raise ValueError('Cached dates differ')
    p=aligned.predicted_probability.to_numpy(float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Cached probabilities are invalid')
    return p
