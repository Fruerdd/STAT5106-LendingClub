"""Check train-only native encoding and matched-recall comparison rules."""
import numpy as np
from src.predictive_modelling import TrainOnlyPreprocessor,engineer_features,temporal_partitions
from src.predictive_improvement import NativeEncoding,operating_thresholds,threshold_metrics,curve_summary,CANDIDATES
from tests.test_modelling import synthetic_modelling_frame


def test_native_encoding_uses_training_categories_and_numeric_preprocessing():
    frame=synthetic_modelling_frame()
    train,validation,_=temporal_partitions(frame)
    train_features=engineer_features(train)
    validation=validation.copy();validation.loc[validation.index[0],'purpose']='unseen_category'
    prep=TrainOnlyPreprocessor(rare_min_count=1).fit(train_features)
    encoder=NativeEncoding().fit(prep,train_features)
    result=encoder.transform(engineer_features(validation))
    assert result.shape[1]==19
    assert sum(encoder.mask)==5
    assert np.isnan(result[0,15])
    assert np.isfinite(result[:,:14]).all()
    assert 'unseen_category' not in encoder.maps['purpose_grouped']


def test_threshold_targets_recall_instead_of_reporting_precision_alone():
    y=np.array([0,0,1,0,1,1,1,0])
    p=np.array([.1,.2,.3,.4,.5,.6,.7,.8])
    rules=operating_thresholds(y,p)
    assert threshold_metrics(y,p,rules['recall_at_least_70'])['recall']>=.7
    score=curve_summary(y,p)['precision_at_recall70']
    assert 0<=score<=1
    assert len(CANDIDATES)==7
    assert len({r['name'] for r in CANDIDATES})==7


def test_grade_specification_removes_interest_rate_from_predictor_matrix():
    frame=synthetic_modelling_frame().assign(grade='A')
    train,validation,_=temporal_partitions(frame)
    train_features=engineer_features(train).assign(grade=train.grade)
    val_features=engineer_features(validation).assign(grade=validation.grade)
    prep=TrainOnlyPreprocessor(rare_min_count=1).fit(train_features)
    encoder=NativeEncoding().fit(prep,train_features,with_grade=True)
    original=encoder.transform(val_features)
    changed=val_features.assign(int_rate_per_5=1000000)
    assert np.allclose(original,encoder.transform(changed),equal_nan=True)
    assert 'int_rate_per_5' not in encoder.numeric_columns
    assert sum(encoder.mask)==6


def extension_frame():
    return synthetic_modelling_frame().assign(grade='A',sub_grade='A1',addr_state='CA',
        installment=300.,revol_bal=10000.,total_acc=20.,mths_since_last_delinq=np.nan,
        mths_since_last_record=np.nan)


def test_extension_uses_one_lender_proxy_and_no_outcome_fields():
    from src.predictive_extension import features,ExtendedEncoding
    frame=extension_frame();train,validation,_=temporal_partitions(frame)
    ft,fv=features(train),features(validation)
    prep=TrainOnlyPreprocessor(rare_min_count=1).fit(ft)
    for proxy in ['int_rate','grade','sub_grade']:
        enc=ExtendedEncoding().fit(prep,ft,proxy)
        a=enc.transform(fv)
        b=enc.transform(features(validation.assign(default=1-validation.default,loan_status='Charged Off')))
        assert np.allclose(a,b,equal_nan=True)
        assert a.shape[1]==27
        assert sum(c in enc.columns for c in ['int_rate_per_5','grade','sub_grade'])==1
        assert not {'default','loan_status','id','issue_date'}.intersection(enc.columns)


def test_extension_preserves_structural_missingness_and_unknown_state():
    from src.predictive_extension import features,ExtendedEncoding
    frame=extension_frame();train,validation,_=temporal_partitions(frame)
    ft=features(train);prep=TrainOnlyPreprocessor(rare_min_count=1).fit(ft)
    enc=ExtendedEncoding().fit(prep,ft,'int_rate')
    changed=validation.copy();changed.loc[changed.index[0],'annual_inc']=0
    changed.loc[changed.index[0],'addr_state']='UNSEEN'
    values=enc.transform(features(changed))
    assert not np.isinf(values).any()
    assert np.isnan(values[0,-1])
    assert np.isnan(values[:,enc.columns.index('months_since_delinquency')]).all()
    assert np.isnan(values[0,enc.columns.index('payment_to_monthly_income')])
