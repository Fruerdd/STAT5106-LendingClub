"""Bounded follow-up experiments using the existing approved predictors."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix
from src.predictive_modelling import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features

# Declared before fitting; candidate choice uses 2016 average precision only.
CANDIDATES = [
    {'name':'original_baseline','native':False,'max_iter':300,'max_leaf_nodes':31,'learning_rate':.08,'min_samples_leaf':20,'l2_regularization':1.,'patience':20},
    {'name':'longer_31','native':False,'max_iter':800,'max_leaf_nodes':31,'learning_rate':.08,'min_samples_leaf':20,'l2_regularization':1.,'patience':30},
    {'name':'slower_31','native':False,'max_iter':1000,'max_leaf_nodes':31,'learning_rate':.04,'min_samples_leaf':50,'l2_regularization':5.,'patience':30},
    {'name':'regularized_15','native':False,'max_iter':800,'max_leaf_nodes':15,'learning_rate':.05,'min_samples_leaf':100,'l2_regularization':10.,'patience':30},
    {'name':'richer_63','native':False,'max_iter':800,'max_leaf_nodes':63,'learning_rate':.05,'min_samples_leaf':100,'l2_regularization':5.,'patience':30},
    {'name':'native_31','native':True,'max_iter':800,'max_leaf_nodes':31,'learning_rate':.05,'min_samples_leaf':100,'l2_regularization':5.,'patience':30},
    {'name':'native_63','native':True,'max_iter':800,'max_leaf_nodes':63,'learning_rate':.05,'min_samples_leaf':100,'l2_regularization':10.,'patience':30},
]


class NativeEncoding:
    """Reuse training-fitted numeric transforms and rare maps; learn category codes on train."""
    def fit(self, prep, features, with_grade=False, proxy=None):
        self.prep = prep
        self.proxy = proxy or ('grade' if with_grade else 'int_rate')
        if self.proxy not in {'int_rate','grade','sub_grade'}:
            raise ValueError('Unknown lender proxy')
        self.extra_category = self.proxy if self.proxy != 'int_rate' else None
        self.numeric_columns = [c for c in NUMERIC_FEATURES if not (self.extra_category and c == 'int_rate_per_5')]
        self.numeric_indices = [NUMERIC_FEATURES.index(c) for c in self.numeric_columns]
        self.categorical_columns = [*CATEGORICAL_FEATURES, *([self.extra_category] if self.extra_category else [])]
        self.numeric = prep.transformer.named_transformers_['numeric']
        self.categorical = prep.transformer.named_transformers_['categorical'].named_steps['prepare']
        grouped = self.categorical.transform(features[list(CATEGORICAL_FEATURES)])
        if self.extra_category:
            grouped[self.extra_category] = features[self.extra_category].astype('string').fillna('MISSING')
        self.maps = {col: {value:i for i,value in enumerate(sorted(grouped[col].unique()))}
                     for col in self.categorical_columns}
        self.mask = [False]*len(self.numeric_columns)+[True]*len(self.categorical_columns)
        return self

    def transform(self, features):
        numeric = self.numeric.transform(features[list(NUMERIC_FEATURES)])[:,self.numeric_indices]
        grouped = self.categorical.transform(features[list(CATEGORICAL_FEATURES)])
        if self.extra_category:
            grouped[self.extra_category] = features[self.extra_category].astype('string').fillna('MISSING')
        categorical = np.column_stack([grouped[col].map(self.maps[col]).to_numpy(float)
                                       for col in self.categorical_columns])
        return np.column_stack([numeric, categorical]).astype('float32')


def estimator(spec, seed, mask=None):
    keys = ['max_iter','max_leaf_nodes','learning_rate','min_samples_leaf','l2_regularization']
    return HistGradientBoostingClassifier(**{k:spec[k] for k in keys},
        early_stopping=True, validation_fraction=.1, n_iter_no_change=spec['patience'],
        random_state=seed, categorical_features=mask if spec['native'] else None)


def curve_summary(y,p):
    precision, recall, _ = precision_recall_curve(y,p)
    eligible = recall >= .70
    return {'precision_at_recall70':float(precision[eligible].max())}


def operating_thresholds(y,p):
    precision, recall, thresholds = precision_recall_curve(y,p)
    precision,recall = precision[:-1],recall[:-1]
    eligible = np.flatnonzero(recall >= .7)
    chosen = eligible[np.argmax(precision[eligible])]
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=(precision+recall)>0)
    return {'recall_at_least_70':float(thresholds[chosen]),'maximum_f1':float(thresholds[np.argmax(f1)])}


def threshold_metrics(y,p,threshold):
    tn,fp,fn,tp = confusion_matrix(y,np.asarray(p)>=threshold,labels=[0,1]).ravel()
    def divide(a,b):return float(a/b) if b else None
    return {'threshold':threshold,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn),
            'f1':divide(2*tp,2*tp+fp+fn),'flagged_fraction':divide(tp+fp,len(y))}
