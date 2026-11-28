"""Three bounded numerical preparations and training-only probability calibration."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from src.predictive_modelling import NUMERIC_FEATURES, CATEGORICAL_FEATURES, CategoricalPreparer, TrainOnlyPreprocessor
from src.predictive_extension import EXTRA_NUMERIC, ExtendedEncoding, features
from src.xgboost_benchmark import make_model, feature_types

MODES = ('current', 'native', 'native_indicators')
FAMILIES = ('xgb', 'hgb')


class AblationEncoder:
    def __init__(self, mode):
        if mode not in MODES:raise ValueError('Unknown preprocessing mode')
        self.mode=mode

    def fit(self, frame):
        f=features(frame)
        self.base_numeric=[c for c in NUMERIC_FEATURES if c!='int_rate_per_5']
        self.cats=[*CATEGORICAL_FEATURES,'grade']
        self.numeric=[*self.base_numeric,*EXTRA_NUMERIC]
        self.columns=[*self.base_numeric,*self.cats,*EXTRA_NUMERIC,'addr_state']
        self.mask=[False]*len(self.base_numeric)+[True]*len(self.cats)+[False]*len(EXTRA_NUMERIC)+[True]
        if self.mode=='current':
            prep=TrainOnlyPreprocessor().fit(f)
            self.legacy=ExtendedEncoding().fit(prep,f,'grade')
            assert self.columns==self.legacy.columns
        else:
            self.preparer=CategoricalPreparer().fit(f[list(CATEGORICAL_FEATURES)])
            grouped=self.preparer.transform(f[list(CATEGORICAL_FEATURES)])
            grouped['grade']=f.grade.fillna('MISSING')
            self.maps={c:{v:i for i,v in enumerate(sorted(grouped[c].unique()))} for c in self.cats}
            self.states={v:i for i,v in enumerate(sorted(f.addr_state.dropna().unique()))}
        self.flag_columns=[c for c in self.numeric if c not in {'income_or_dti_missing','emp_length_years_was_missing'}]
        if self.mode=='native_indicators':
            self.columns.extend(c+'_missing' for c in self.flag_columns)
            self.mask.extend([False]*len(self.flag_columns))
        assert not {'default','loan_status','id','issue_date','int_rate_per_5','sub_grade'}.intersection(self.columns)
        return self

    def transform(self, frame):
        f=features(frame)
        if self.mode=='current':return self.legacy.transform(f)
        f[self.numeric]=f[self.numeric].replace([np.inf,-np.inf],np.nan)
        grouped=self.preparer.transform(f[list(CATEGORICAL_FEATURES)])
        grouped['grade']=f.grade.fillna('MISSING')
        cats=np.column_stack([grouped[c].map(self.maps[c]).to_numpy(float) for c in self.cats])
        values=np.column_stack([f[self.base_numeric].to_numpy(float),cats,f[EXTRA_NUMERIC].to_numpy(float),f.addr_state.map(self.states).to_numpy(float)])
        if self.mode=='native_indicators':values=np.column_stack([values,f[self.flag_columns].isna().to_numpy(dtype='float32')])
        result=values.astype('float32')
        if np.isinf(result).any():raise ValueError('Predictor overflow')
        return result


class SigmoidCalibration:
    def fit(self, y, p):
        self.model=LogisticRegression(C=1000.,solver='lbfgs',max_iter=500,random_state=5106)
        self.model.fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),y)
        if self.model.coef_[0,0]<=0:raise ValueError('Calibration did not preserve positive risk ordering')
        return self

    def transform(self, p):
        return expit(self.model.intercept_[0]+self.model.coef_[0,0]*logit(np.clip(p,1e-6,1-1e-6)))


def boost(family, rounds, mask, y, early=False):
    if family=='xgb':
        return make_model({'max_depth':5,'weight':'double'},rounds,feature_types(mask),y,early_stopping=early)
    return HistGradientBoostingClassifier(max_iter=rounds,max_leaf_nodes=31,learning_rate=.05,
        min_samples_leaf=100,l2_regularization=5.,categorical_features=mask,
        early_stopping=early,n_iter_no_change=30,random_state=5106)


def fit_stopped(family, mode, fit, early, stop, *others):
    e0=AblationEncoder(mode).fit(early)
    m=boost(family,800,e0.mask,early.default,early=True)
    if family=='xgb':
        m.fit(e0.transform(early),early.default,eval_set=[(e0.transform(stop),stop.default)],verbose=False)
        rounds=int(m.best_iteration)+1
    else:
        m.fit(e0.transform(early),early.default,X_val=e0.transform(stop),y_val=stop.default)
        rounds=int(m.n_iter_)
    enc=AblationEncoder(mode).fit(fit)
    m=boost(family,rounds,enc.mask,fit.default)
    m.fit(enc.transform(fit),fit.default)
    return enc,m,rounds,[m.predict_proba(enc.transform(f))[:,1] for f in others]


def split_calibration_threshold(frame):
    dates=pd.to_datetime(frame.issue_date)
    counts=dates.value_counts().sort_index()
    if len(counts)<2:raise ValueError('Calibration/threshold split requires at least two issue months')
    # A row median can fall in the first month when issuance is uneven, leaving no earlier rows.
    cumulative=counts.cumsum().iloc[:-1]
    midpoint=counts.index[int(np.argmin(np.abs(cumulative.to_numpy()-len(frame)/2)))+1]
    calibration=frame.loc[dates.lt(midpoint)];threshold=frame.loc[dates.ge(midpoint)]
    if any(p.empty or p.default.nunique()!=2 for p in [calibration,threshold]):
        raise ValueError('Calibration and threshold blocks require distinct months and both classes')
    return calibration,threshold


def complementary(y, px, ph, tx, th):
    y=np.asarray(y,dtype=bool);x=np.asarray(px)>=tx;h=np.asarray(ph)>=th
    only_x=np.mean((x==y)&(h!=y));only_h=np.mean((h==y)&(x!=y));disagreement=np.mean(x!=h)
    return {'decision_disagreement':float(disagreement),'only_xgb_correct':float(only_x),
        'only_hgb_correct':float(only_h),'probability_correlation':float(np.corrcoef(px,ph)[0,1]),
        'ensemble_eligible':bool(disagreement>=.01 and only_x>=.005 and only_h>=.005)}


def write_predictions(frame, p, path, model_name):
    # CSV's default float32 formatting can move scores across a stored float64 threshold.
    frame[['id','issue_date','default']].assign(model=model_name,predicted_probability=np.asarray(p,dtype='float64')).to_csv(
        path,index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
