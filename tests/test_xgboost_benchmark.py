"""Chronological benchmark controls and optional XGBoost categorical smoke check."""
import numpy as np
import pandas as pd
import pytest
from src.xgboost_benchmark import CANDIDATES,training_weight,f1_threshold,feature_types,make_model,check_handoff
from src.predictive_improvement import threshold_metrics


def test_training_class_weights_use_only_supplied_training_labels():
    y=np.array([0,0,0,1])
    assert training_weight('balanced',y)==3
    assert training_weight('unweighted',y)==1
    assert training_weight('double',y)==2
    with pytest.raises(ValueError):training_weight('balanced',[0,0])
    assert len(CANDIDATES)==6 and len({s['name'] for s in CANDIDATES})==6


def test_f1_threshold_matches_best_available_threshold_without_test_inputs():
    y=np.array([0,1,0,1,1,0]);p=np.array([.1,.2,.3,.5,.5,.9])
    t=f1_threshold(y,p)
    achieved=threshold_metrics(y,p,t)['f1']
    assert achieved==max(threshold_metrics(y,p,v)['f1'] for v in np.unique(p))
    with pytest.raises(ValueError):f1_threshold(y,[np.nan]*len(y))


def test_reference_cache_requires_exact_identity_dates_and_outcomes():
    cohort=pd.DataFrame({'id':[1,2],'issue_date':['2018-01-01','2018-02-01'],'default':[0,1]})
    cache=cohort.assign(predicted_probability=[.2,.7]).iloc[::-1]
    assert np.allclose(check_handoff(cache,cohort),[.2,.7])
    for col,val in [('id',3),('issue_date','2017-01-01'),('default',1),('predicted_probability',np.inf)]:
        bad=cohort.assign(predicted_probability=[.2,.7]);bad.loc[0,col]=val
        with pytest.raises(ValueError):check_handoff(bad,cohort)


def test_optional_xgboost_handles_native_categories_missing_values_and_reload(tmp_path):
    xgb=pytest.importorskip('xgboost')
    rng=np.random.default_rng(5106)
    x=np.column_stack([rng.normal(size=600),rng.integers(0,3,size=600)]).astype('float32')
    x[0,1]=np.nan;y=(x[:,0]+.4*(np.nan_to_num(x[:,1])==1)>0).astype(int)
    spec=CANDIDATES[0];types=feature_types([False,True])
    model=make_model(spec,8,types,y);model.fit(x,y,verbose=False)
    p=model.predict_proba(x)[:,1];assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    path=tmp_path/'model.ubj';model.save_model(path)
    restored=xgb.XGBClassifier();restored.load_model(path)
    assert np.allclose(p,restored.predict_proba(x)[:,1])
