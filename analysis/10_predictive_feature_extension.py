"""Additional reviewed application-time fields; 2016 selection and exploratory 2018 assessment."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,sys,time,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR,TABLES_DIR
from src.predictive_modelling import TrainOnlyPreprocessor,binary_metrics
from src.predictive_improvement import estimator,curve_summary,threshold_metrics
from src.predictive_extension import READ_COLUMNS,EXTRA_NUMERIC,SPECS,features,ExtendedEncoding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/feature_extension')
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':'planned','candidates':SPECS,'selection':'2016 average precision, lower log loss breaks ties',
       'specification':'report/predictive_feature_extension_specification.md',
       'tuning_train':'Before 2016','tuning_evaluation':'2016','final_train':'Before 2017',
       'threshold_selection':'2017 only; same full-PR-curve F1/cost selection as the preceding baseline comparison',
       'assessment':'2018 reused test; exploratory follow-up, not an untouched final holdout',
       'bootstrap':'1000 paired loan draws; percentile 95% CI; fitted models and thresholds fixed',
       'seed':5106,'primary_outputs_replaced':False}
    path=args.output_dir/'experiment_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    frame=pd.read_csv(args.input,usecols=READ_COLUMNS,low_memory=False);dates=pd.to_datetime(frame.issue_date)
    train=frame.loc[dates.lt('2016-01-01')];tune=frame.loc[dates.ge('2016-01-01')&dates.lt('2017-01-01')]
    ft,fv=features(train),features(tune);prep=TrainOnlyPreprocessor().fit(ft)
    rows=[]
    for s in SPECS:
        start=time.perf_counter();encoder=ExtendedEncoding().fit(prep,ft,s['proxy'])
        model=estimator(s,5106,encoder.mask);model.fit(encoder.transform(ft),train.default)
        p=model.predict_proba(encoder.transform(fv))[:,1]
        row={'candidate':s['name'],'iterations':int(model.n_iter_),**binary_metrics(tune.default,p),**curve_summary(tune.default,p),
             'fit_seconds':round(time.perf_counter()-start,2)};rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir/'tuning_comparison.csv',index=False)
        print(f"{s['name']}: 2016 AP={row['pr_auc']:.6f}, precision at >=70% recall={row['precision_at_recall70']:.4f}, AUC={row['roc_auc']:.6f}",flush=True)
    winner=pd.DataFrame(rows).sort_values(['pr_auc','log_loss'],ascending=[False,True]).iloc[0].candidate
    s=next(s for s in SPECS if s['name']==winner)
    manifest.update(status='selected_before_2017_and_2018_evaluation',selected_on_2016=winner)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    train=frame.loc[dates.lt('2017-01-01')];validation=frame.loc[dates.ge('2017-01-01')&dates.lt('2018-01-01')];test=frame.loc[dates.ge('2018-01-01')]
    ft,fv,fe=features(train),features(validation),features(test);prep=TrainOnlyPreprocessor().fit(ft)
    encoder=ExtendedEncoding().fit(prep,ft,s['proxy']);manifest['predictor_columns']=encoder.columns
    model=estimator(s,5106,encoder.mask);model.fit(encoder.transform(ft),train.default)
    pval=model.predict_proba(encoder.transform(fv))[:,1]
    m=importlib.util.spec_from_file_location('frozen_assessment',PROJECT_DIR/'analysis/09_frozen_improvement_assessment.py')
    mod=importlib.util.module_from_spec(m);m.loader.exec_module(mod)
    rules=mod.rules(validation.default.to_numpy(),pval)
    pd.DataFrame([{'candidate':winner,**binary_metrics(validation.default,pval),**curve_summary(validation.default,pval)}]).to_csv(args.output_dir/'comparison_2017.csv',index=False)
    p=model.predict_proba(encoder.transform(fe))[:,1];y=test.default.to_numpy()
    original=pd.read_csv(TABLES_DIR/'out_of_time_predictions.csv');original=original.loc[original.model.eq('hist_gradient_boosting_leaves31_none')].set_index('id')
    aligned=original.reindex(test.id)
    if not np.array_equal(aligned.default.to_numpy(),y) or set(original.index)!=set(test.id):raise ValueError('Baseline identity mismatch')
    p0=aligned.predicted_probability.to_numpy()
    comparison=[{'candidate':'original_baseline','rows':len(test),**binary_metrics(y,p0),**curve_summary(y,p0)},
                {'candidate':winner,'rows':len(test),**binary_metrics(y,p),**curve_summary(y,p)}]
    pd.DataFrame(comparison).to_csv(args.output_dir/'comparison_2018.csv',index=False)
    old_policy=pd.read_csv(PROJECT_DIR/'outputs/experiments/frozen_2018_assessment/thresholds_2018.csv')
    baseline_rules=old_policy.loc[old_policy.candidate.eq('original_baseline')].set_index('rule').threshold.to_dict()
    policy=[]
    for name,probs,rs in [('original_baseline',p0,baseline_rules),(winner,p,rules)]:
        for rule,t in rs.items():policy.append({'candidate':name,'rule':rule,**threshold_metrics(y,probs,t)})
    pd.DataFrame(policy).to_csv(args.output_dir/'thresholds_2018.csv',index=False)
    rng=np.random.default_rng(5106);draws=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),size=len(y))
        draws.append(average_precision_score(y[i],p[i])-average_precision_score(y[i],p0[i]))
    low,high=np.quantile(draws,[.025,.975])
    manifest.update(status='completed',ap_delta=float(average_precision_score(y,p)-average_precision_score(y,p0)),ap_delta_ci=[float(low),float(high)],
                    thresholds_selected_on_2017=rules,validation_average_precision=float(average_precision_score(validation.default,pval)))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"2018 {winner}: AP={comparison[1]['pr_auc']:.6f}, precision at >=70% recall={comparison[1]['precision_at_recall70']:.4f}, AUC={comparison[1]['roc_auc']:.6f}; AP delta={manifest['ap_delta']:+.6f}; CI [{low:+.6f},{high:+.6f}]",flush=True)
    print('Complete. Existing primary results and Ivy validation were not overwritten.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
