"""Exploratory model improvement: tune on 2016, compare on 2017, do not evaluate 2018.

Outputs go to outputs/experiments/predictive_improvement. Primary results and Ivy's code
remain unchanged. The experiment manifest is written before any fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR, CLEANED_DATA_DIR
from src.predictive_modelling import INPUT_COLUMNS, TrainOnlyPreprocessor, engineer_features, binary_metrics
from src.predictive_improvement import CANDIDATES, NativeEncoding, estimator, operating_thresholds, threshold_metrics, curve_summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/predictive_improvement')
    parser.add_argument('--seed',type=int,default=5106)
    args=parser.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':'planned','candidates':CANDIDATES,'selection_metric':'2016 average precision; lower log loss breaks ties',
        'tuning_train':'issue_date < 2016-01-01','tuning_evaluation':'2016-01-01 <= issue_date < 2017-01-01',
        'comparison_train':'issue_date < 2017-01-01','comparison_evaluation':'2017-01-01 <= issue_date < 2018-01-01',
        'comparison_models':'Original baseline and the single 2016-selected candidate only',
        'test_2018_evaluated':False,'scope':'Exploratory follow-up: 2017 and 2018 have previously been inspected in the main analysis',
        'features':'Existing Model C feature contract; no additions or removals; training-only imputation, caps and rare maps',
        'encoding_change':'Native candidates replace one-hot categories with train-learned category codes; unseen codes become missing',
        'threshold_selection':'2016 predictions only; frozen thresholds transferred to refitted models for 2017',
        'curve_metric':'Maximum precision on the evaluation PR curve subject to recall >= 70%; descriptive, not a deployment threshold',
        'bootstrap':'500 paired loan resamples; 95% percentile CI for difference in 2017 average precision; fixed fitted models',
        'seed':args.seed}
    manifest_path=args.output_dir/'experiment_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    pieces=[]
    for chunk in pd.read_csv(args.input,usecols=list(INPUT_COLUMNS),chunksize=100000):
        pieces.append(chunk.loc[pd.to_datetime(chunk.issue_date).lt('2018-01-01')])
    frame=pd.concat(pieces,ignore_index=True)
    dates=pd.to_datetime(frame.issue_date)
    tune_train=frame.loc[dates.lt('2016-01-01')]
    tune=frame.loc[dates.ge('2016-01-01')&dates.lt('2017-01-01')]
    compare=frame.loc[dates.ge('2017-01-01')]
    if any(f.empty or f.default.nunique()!=2 for f in [tune_train,tune,compare]):
        raise ValueError('Training, tuning and comparison cohorts must each contain both outcomes')
    manifest['cohort_rows']={'tune_train':len(tune_train),'tune':len(tune),'comparison':len(compare)}
    print('Cohorts:',manifest['cohort_rows'],flush=True)
    train_features,tune_features=engineer_features(tune_train),engineer_features(tune)
    prep=TrainOnlyPreprocessor().fit(train_features)
    native=NativeEncoding().fit(prep,train_features)
    matrices={False:(prep.transform(train_features),prep.transform(tune_features)),
              True:(native.transform(train_features),native.transform(tune_features))}
    rows=[];thresholds={}
    for spec in CANDIDATES:
        started=time.perf_counter()
        train_x,tune_x=matrices[spec['native']]
        model=estimator(spec,args.seed,native.mask)
        model.fit(train_x,tune_train.default)
        p=model.predict_proba(tune_x)[:,1]
        row={'candidate':spec['name'],'iterations':int(model.n_iter_),
             **binary_metrics(tune.default,p),**curve_summary(tune.default,p),
             'fit_seconds':round(time.perf_counter()-started,2)}
        thresholds[spec['name']]=operating_thresholds(tune.default,p)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir/'tuning_comparison.csv',index=False)
        print(f"{spec['name']}: AP={row['pr_auc']:.6f}, precision at >=70% recall={row['precision_at_recall70']:.4f}, AUC={row['roc_auc']:.6f}; iterations={model.n_iter_}; {row['fit_seconds']}s",flush=True)
        del model
    ranked=pd.DataFrame(rows).sort_values(['pr_auc','log_loss'],ascending=[False,True])
    selected=ranked.iloc[0].candidate
    manifest['selected_on_2016']=selected
    manifest['status']='selected_before_2017_comparison'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen selection before 2017 comparison:',selected,flush=True)
    final_train=frame.loc[dates.lt('2017-01-01')]
    final_features,compare_features=engineer_features(final_train),engineer_features(compare)
    prep=TrainOnlyPreprocessor().fit(final_features)
    native=NativeEncoding().fit(prep,final_features)
    matrices={False:(prep.transform(final_features),prep.transform(compare_features)),
              True:(native.transform(final_features),native.transform(compare_features))}
    comparison_rows=[];policy_rows=[];probabilities={}
    for name in dict.fromkeys(['original_baseline',selected]):
        spec=next(s for s in CANDIDATES if s['name']==name)
        train_x,compare_x=matrices[spec['native']]
        model=estimator(spec,args.seed,native.mask)
        model.fit(train_x,final_train.default)
        p=model.predict_proba(compare_x)[:,1];probabilities[name]=p
        row={'candidate':name,'selected_on_2016':name==selected,'rows':len(compare),
             **binary_metrics(compare.default,p),**curve_summary(compare.default,p)}
        comparison_rows.append(row)
        for goal,threshold in thresholds[name].items():
            policy_rows.append({'candidate':name,'rule':goal,'threshold_selected_on':'2016',
                                **threshold_metrics(compare.default,p,threshold)})
        print(f"2017 {name}: AP={row['pr_auc']:.6f}, precision at >=70% recall={row['precision_at_recall70']:.4f}, AUC={row['roc_auc']:.6f}",flush=True)
        del model
    y=compare.default.to_numpy()
    baseline=probabilities['original_baseline'];candidate=probabilities[selected]
    rng=np.random.default_rng(args.seed);draws=[]
    for _ in range(500):
        indices=rng.integers(0,len(y),size=len(y))
        draws.append(average_precision_score(y[indices],candidate[indices])-average_precision_score(y[indices],baseline[indices]))
    low,high=np.quantile(draws,[.025,.975])
    delta=average_precision_score(y,candidate)-average_precision_score(y,baseline)
    manifest.update(status='completed',comparison_ap_delta=float(delta),comparison_ap_delta_ci=[float(low),float(high)],
                    positive_conditional_ap_interval_on_2017=bool(delta>0 and low>0),thresholds_selected_on_2016=thresholds)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir/'comparison_2017.csv',index=False)
    pd.DataFrame(policy_rows).to_csv(args.output_dir/'threshold_transfer_2017.csv',index=False)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'2017 AP difference: {delta:+.6f}; paired 95% CI [{low:+.6f}, {high:+.6f}]',flush=True)
    print('Completed. 2018 was not evaluated. Original outputs were not replaced.',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
