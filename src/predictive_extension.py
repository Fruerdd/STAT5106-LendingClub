"""Explicitly registered extension using existing application-time cleaned fields."""
import numpy as np
import pandas as pd
from src.predictive_modelling import INPUT_COLUMNS,engineer_features
from src.predictive_improvement import NativeEncoding

EXTRA_SOURCES=['grade','sub_grade','addr_state','installment','revol_bal','total_acc',
               'mths_since_last_delinq','mths_since_last_record']
READ_COLUMNS=list(dict.fromkeys([*INPUT_COLUMNS,*EXTRA_SOURCES]))
EXTRA_NUMERIC=['log1p_loan_amount','payment_to_monthly_income','log1p_revolving_balance',
               'total_credit_accounts','fraction_open_accounts','months_since_delinquency','months_since_record']
SPECS=[{'name':f'expanded_{proxy}_{leaves}','proxy':proxy,'native':True,'max_iter':800,
        'max_leaf_nodes':leaves,'learning_rate':.05,'min_samples_leaf':100,
        'l2_regularization':10. if leaves==15 else 5.,'patience':30}
       for proxy,leaves in [('int_rate',15),('int_rate',31),('grade',31),('sub_grade',31)]]


def features(frame):
    result=engineer_features(frame)
    for column in ['grade','sub_grade','addr_state']:
        result[column]=frame[column].astype('string')
    numeric={c:pd.to_numeric(frame[c],errors='coerce') for c in
             ['loan_amnt','annual_inc','installment','revol_bal','total_acc','open_acc','mths_since_last_delinq','mths_since_last_record']}
    result['log1p_loan_amount']=np.log1p(numeric['loan_amnt'].where(numeric['loan_amnt'].ge(0)))
    result['payment_to_monthly_income']=12*numeric['installment']/numeric['annual_inc'].where(numeric['annual_inc'].gt(0))
    result['log1p_revolving_balance']=np.log1p(numeric['revol_bal'].where(numeric['revol_bal'].ge(0)))
    result['total_credit_accounts']=numeric['total_acc']
    result['fraction_open_accounts']=numeric['open_acc']/numeric['total_acc'].where(numeric['total_acc'].gt(0))
    result['months_since_delinquency']=numeric['mths_since_last_delinq']
    result['months_since_record']=numeric['mths_since_last_record']
    result[EXTRA_NUMERIC]=result[EXTRA_NUMERIC].replace([np.inf,-np.inf],np.nan)
    return result


class ExtendedEncoding:
    def fit(self,prep,features,proxy):
        self.base=NativeEncoding().fit(prep,features,proxy=proxy)
        self.state_map={v:i for i,v in enumerate(sorted(features.addr_state.dropna().unique()))}
        self.mask=self.base.mask+[False]*len(EXTRA_NUMERIC)+[True]
        self.columns=self.base.numeric_columns+self.base.categorical_columns+EXTRA_NUMERIC+['addr_state']
        assert not {'id','default','loan_status','issue_date'}.intersection(self.columns)
        assert sum(c in self.columns for c in ['int_rate_per_5','grade','sub_grade'])==1
        return self

    def transform(self,features):
        extra=features[EXTRA_NUMERIC].to_numpy(dtype='float32')
        states=features.addr_state.map(self.state_map).to_numpy(dtype='float32')[:,None]
        return np.column_stack([self.base.transform(features),extra,states]).astype('float32')
