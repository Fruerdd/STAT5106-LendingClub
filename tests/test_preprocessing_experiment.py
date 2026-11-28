import numpy as np
import pytest
from sklearn.metrics import f1_score,precision_score,recall_score
from tests.test_predictive_improvement import extension_frame
from src.predictive_modelling import TrainOnlyPreprocessor,engineer_features,NUMERIC_FEATURES
from src.preprocessing_experiment import AblationEncoder,SigmoidCalibration,complementary,split_calibration_threshold
from src.xgboost_benchmark import f1_threshold
from src.predictive_improvement import threshold_metrics


def test_all_missing_numeric_column_keeps_original_layout():
    frame=engineer_features(extension_frame())
    frame['dti_per_10']=np.nan
    prep=TrainOnlyPreprocessor(rare_min_count=1).fit(frame)
    numeric=prep.transformer.named_transformers_['numeric'].transform(frame[list(NUMERIC_FEATURES)])
    assert numeric.shape[1]==len(NUMERIC_FEATURES)
    assert np.isfinite(prep.transform(frame)).all()
    assert np.all(numeric[:,NUMERIC_FEATURES.index('dti_per_10')]==0)


def test_native_preserves_missing_values_extremes_and_unseen_categories():
    train=extension_frame(); future=train.iloc[:4].copy()
    future.loc[future.index[0],'dti']=np.nan
    future.loc[future.index[1],'dti']=10000.
    future['grade']='UNSEEN';future['addr_state']='UNSEEN'
    enc=AblationEncoder('native_indicators').fit(train)
    values=enc.transform(future)
    assert values.shape[1]==45
    assert np.isnan(values[0,enc.columns.index('dti_per_10')])
    assert values[1,enc.columns.index('dti_per_10')]==1000.
    assert values[0,enc.columns.index('dti_per_10_missing')]==1
    assert np.isnan(values[:,enc.columns.index('grade')]).all()
    assert np.isnan(values[:,enc.columns.index('addr_state')]).all()
    assert np.allclose(values,enc.transform(future.assign(default=1-future.default)),equal_nan=True)


def test_calibration_and_threshold_metrics_use_correct_positive_class_and_ties():
    y=np.array([0,0,1,0,1,1,0,1]);p=np.array([.1,.2,.2,.3,.4,.5,.5,.9])
    c=SigmoidCalibration().fit(y,p);q=c.transform(p)
    t=f1_threshold(y,p);stats=threshold_metrics(y,p,t);flags=p>=t
    assert stats['f1']==pytest.approx(f1_score(y,flags))
    assert stats['precision']==pytest.approx(precision_score(y,flags))
    assert stats['recall']==pytest.approx(recall_score(y,flags))
    assert threshold_metrics(y,q,f1_threshold(y,q))['f1']==pytest.approx(stats['f1'])
    assert np.all(np.diff(q[np.argsort(p)])>=0)


def test_identical_models_are_not_ensemble_eligible():
    result=complementary([0,0,1,1],[.1,.4,.3,.9],[.1,.4,.3,.9],.5,.5)
    assert not result['ensemble_eligible'] and result['decision_disagreement']==0


def test_calibration_threshold_months_do_not_overlap():
    import pandas as pd
    frame=extension_frame().iloc[:80].copy()
    frame['issue_date']=np.repeat(pd.date_range('2014-01-01',periods=4,freq='MS'),20)
    a,b=split_calibration_threshold(frame)
    assert a.issue_date.max()<b.issue_date.min()
    assert len(a)+len(b)==len(frame)


def test_prediction_export_keeps_float32_threshold_boundary_exact(tmp_path):
    import pandas as pd
    from src.preprocessing_experiment import write_predictions
    frame=extension_frame().iloc[:3]
    scores=np.array([.29415860772132874,.4,.2],dtype='float32')
    path=tmp_path/'scores.csv.gz'
    write_predictions(frame,scores,path,'test')
    restored=pd.read_csv(path,float_precision='round_trip').predicted_probability.to_numpy()
    assert np.array_equal(scores.astype(float),restored)
    assert np.array_equal(scores>=float(scores[0]),restored>=float(scores[0]))


def test_calibration_split_remains_nonempty_with_uneven_issue_volumes():
    import pandas as pd
    frame=extension_frame().iloc[:80].copy()
    frame['issue_date']=pd.to_datetime(['2014-01-01']*60+['2014-02-01']*20)
    a,b=split_calibration_threshold(frame)
    assert len(a)==60 and len(b)==20
    assert a.issue_date.max()<b.issue_date.min()
