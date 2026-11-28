"""Exploratory approved Model B benchmark: grade replaces interest rate; 2018 excluded."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR,TABLES_DIR
from src.predictive_modelling import INPUT_COLUMNS,TrainOnlyPreprocessor,engineer_features,binary_metrics
from src.predictive_improvement import NativeEncoding,estimator,curve_summary,operating_thresholds,threshold_metrics

SPECS=[{'name':f'grade_{leaves}','native':True,'max_iter':800,'max_leaf_nodes':leaves,
        'learning_rate':.05,'min_samples_leaf':100,'l2_regularization':5. if leaves==31 else 10.,'patience':30}
       for leaves in [31,63]]


def features(frame):
    return engineer_features(frame).assign(grade=frame.grade.astype('string'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/grade_benchmark')
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':'planned','candidates':SPECS,'selection':'2016 average precision; lower log loss breaks ties',
       'features':'Approved Model B without issue_year: grade replaces interest rate; no sub_grade',
       'preprocessing':'Numeric imputers/caps and rare maps fitted on training only; native categorical splits',
       'tuning':'Train before 2016; evaluate 2016','comparison':'Refit before 2017; evaluate 2017 only',
       '2018_evaluated':False,'scope':'Additional exploratory phase; neither 2017 nor 2018 is a new untouched holdout',
       'seed':5106}
    path=args.output_dir/'experiment_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    pieces=[]
    for chunk in pd.read_csv(args.input,usecols=[*INPUT_COLUMNS,'grade'],chunksize=100000):
        pieces.append(chunk.loc[pd.to_datetime(chunk.issue_date).lt('2018-01-01')])
    frame=pd.concat(pieces,ignore_index=True);dates=pd.to_datetime(frame.issue_date)
    train=frame.loc[dates.lt('2016-01-01')];tune=frame.loc[dates.ge('2016-01-01')&dates.lt('2017-01-01')]
    ftrain,ftune=features(train),features(tune)
    prep=TrainOnlyPreprocessor().fit(ftrain);encoder=NativeEncoding().fit(prep,ftrain,with_grade=True)
    contract=pd.read_csv(TABLES_DIR/'feature_specification.csv')
    expected=set(contract.loc[contract.model_b.eq('yes'),'engineered_feature'])-{'default','issue_year'}
    actual=set(encoder.numeric_columns+encoder.categorical_columns)
    if actual!=expected or 'int_rate_per_5' in actual or 'sub_grade' in actual:
        raise ValueError('Predictive Model B feature contract mismatch')
    manifest['predictor_columns']=encoder.numeric_columns+encoder.categorical_columns
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    xtrain,xtune=encoder.transform(ftrain),encoder.transform(ftune)
    rows=[];rules={}
    for spec in SPECS:
        started=time.perf_counter();model=estimator(spec,5106,encoder.mask);model.fit(xtrain,train.default)
        p=model.predict_proba(xtune)[:,1]
        row={'candidate':spec['name'],'iterations':int(model.n_iter_),**binary_metrics(tune.default,p),**curve_summary(tune.default,p),
             'fit_seconds':round(time.perf_counter()-started,2)}
        rows.append(row);rules[spec['name']]=operating_thresholds(tune.default,p)
        pd.DataFrame(rows).to_csv(args.output_dir/'tuning_comparison.csv',index=False)
        print(f"{spec['name']}: 2016 AP={row['pr_auc']:.6f}, precision at >=70% recall={row['precision_at_recall70']:.4f}, AUC={row['roc_auc']:.6f}",flush=True)
    winner=pd.DataFrame(rows).sort_values(['pr_auc','log_loss'],ascending=[False,True]).iloc[0].candidate
    manifest.update(selected_on_2016=winner,status='selected_before_2017_comparison')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    train=frame.loc[dates.lt('2017-01-01')];compare=frame.loc[dates.ge('2017-01-01')]
    ftrain,fcompare=features(train),features(compare)
    prep=TrainOnlyPreprocessor().fit(ftrain);encoder=NativeEncoding().fit(prep,ftrain,with_grade=True)
    model=estimator(next(s for s in SPECS if s['name']==winner),5106,encoder.mask)
    model.fit(encoder.transform(ftrain),train.default)
    p=model.predict_proba(encoder.transform(fcompare))[:,1]
    row={'candidate':winner,'rows':len(compare),**binary_metrics(compare.default,p),**curve_summary(compare.default,p)}
    pd.DataFrame([row]).to_csv(args.output_dir/'comparison_2017.csv',index=False)
    policy=[{'candidate':winner,'rule':name,**threshold_metrics(compare.default,p,t)} for name,t in rules[winner].items()]
    pd.DataFrame(policy).to_csv(args.output_dir/'threshold_transfer_2017.csv',index=False)
    manifest.update(status='completed',comparison_2017=row,thresholds_selected_on_2016=rules)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"2017 {winner}: AP={row['pr_auc']:.6f}, precision at >=70% recall={row['precision_at_recall70']:.4f}, AUC={row['roc_auc']:.6f}",flush=True)
    print('Completed. Compare with original_baseline in the previous experiment; 2018 was not evaluated.',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
