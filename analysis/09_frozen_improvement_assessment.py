"""One exploratory 2018 assessment of the candidate chosen on 2016 results.

The 2018 main results were previously inspected. This is explicitly reused-test evaluation,
not a new untouched holdout. It never changes primary outputs or Ivy's original validation.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,average_precision_score
if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR
from src.predictive_modelling import INPUT_COLUMNS,TrainOnlyPreprocessor,engineer_features,binary_metrics
from src.predictive_improvement import CANDIDATES,NativeEncoding,estimator,curve_summary,threshold_metrics
from importlib.util import spec_from_file_location,module_from_spec


def rules(y,p):
    precision,recall,t=precision_recall_curve(y,p)
    precision,recall=precision[:-1],recall[:-1]
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=precision+recall>0)
    # Same cost assumption as Ivy; choose on validation only.
    tp=recall*np.sum(y); fp=np.divide(tp,precision,out=np.full_like(tp,np.inf),where=precision>0)-tp
    costs=5*(np.sum(y)-tp)+fp
    return {'maximum_validation_f1':float(t[np.argmax(f1)]),'validation_cost_5_to_1':float(t[np.argmin(costs)])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/frozen_2018_assessment')
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=True)
    base=PROJECT_DIR/'outputs/experiments'
    c=pd.read_csv(base/'predictive_improvement/tuning_comparison.csv')
    b=pd.read_csv(base/'grade_benchmark/tuning_comparison.csv')
    winner=pd.concat([c,b],ignore_index=True).sort_values(['pr_auc','log_loss'],ascending=[False,True]).iloc[0].candidate
    grade_manifest=json.loads((base/'grade_benchmark/experiment_manifest.json').read_text())
    specs={s['name']:s for s in [*CANDIDATES,*grade_manifest['candidates']]}
    manifest={'status':'frozen_before_assessment','selected_on_2016':winner,'selection':'Maximum 2016 average precision across nine registered candidates; log loss breaks ties',
       'threshold_selection':'2017 only; maximum F1 and illustrative FN:FP cost 5:1',
       'training':'Before 2017; same approved predictors and original outcome',
       'assessment':'2018; previously examined in main study, explicitly exploratory test reuse',
       'candidate_specification':specs[winner],'seed':5106,'bootstrap_resamples':1000,
       'primary_outputs_replaced':False,'subgroup_or_test_result_model_selection':False}
    path=args.output_dir/'experiment_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen winner from 2016:',winner,flush=True)
    frame=pd.read_csv(args.input,usecols=[*INPUT_COLUMNS,'grade'])
    dates=pd.to_datetime(frame.issue_date);train=frame.loc[dates.lt('2017-01-01')]
    validation=frame.loc[dates.ge('2017-01-01')&dates.lt('2018-01-01')];test=frame.loc[dates.ge('2018-01-01')]
    def features(f):return engineer_features(f).assign(grade=f.grade.astype('string'))
    tr,va,te=features(train),features(validation),features(test)
    prep=TrainOnlyPreprocessor().fit(tr)
    rows=[];policy=[];probabilities={};thresholds={}
    for name in dict.fromkeys(['original_baseline',winner]):
        s=specs[name]
        if s['native']:
            enc=NativeEncoding().fit(prep,tr,with_grade=name.startswith('grade_'))
            xtr,xva,xte=enc.transform(tr),enc.transform(va),enc.transform(te)
            mask=enc.mask
        else:
            xtr,xva,xte=prep.transform(tr),prep.transform(va),prep.transform(te);mask=None
        model=estimator(s,5106,mask);model.fit(xtr,train.default)
        pval=model.predict_proba(xva)[:,1]
        frozen=rules(validation.default.to_numpy(),pval);thresholds[name]=frozen
        # Only now evaluate this frozen candidate and its frozen operating rules on 2018.
        p=model.predict_proba(xte)[:,1];probabilities[name]=p
        rows.append({'candidate':name,'cohort':'2018 reused test','rows':len(test),**binary_metrics(test.default,p),**curve_summary(test.default,p)})
        for goal,t in frozen.items():policy.append({'candidate':name,'rule':goal,**threshold_metrics(test.default,p,t)})
        print(f"2018 {name}: AP={rows[-1]['pr_auc']:.6f}, precision at >=70% recall={rows[-1]['precision_at_recall70']:.4f}, AUC={rows[-1]['roc_auc']:.6f}",flush=True)
    y=test.default.to_numpy();p0=probabilities['original_baseline'];p1=probabilities[winner]
    rng=np.random.default_rng(5106);draws=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),size=len(y))
        draws.append(average_precision_score(y[i],p1[i])-average_precision_score(y[i],p0[i]))
    low,high=np.quantile(draws,[.025,.975])
    manifest.update(status='completed',ap_delta=float(average_precision_score(y,p1)-average_precision_score(y,p0)),ap_delta_ci=[float(low),float(high)],
                    validation_selected_thresholds=thresholds)
    pd.DataFrame(rows).to_csv(args.output_dir/'comparison_2018.csv',index=False)
    pd.DataFrame(policy).to_csv(args.output_dir/'thresholds_2018.csv',index=False)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Completed exploratory assessment; primary outputs unchanged.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
