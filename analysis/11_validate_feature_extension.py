"""Validate the frozen feature-extension candidate using Ivy's unchanged utility functions.

This is follow-up exploratory validation, not a replacement for Ivy's original contribution.
Prediction handoffs remain in ignored data/cleaned; only aggregate results/plots are exported.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR,TABLES_DIR
from src.predictive_modelling import TrainOnlyPreprocessor,binary_metrics
from src.predictive_improvement import estimator,threshold_metrics
from src.predictive_extension import READ_COLUMNS,SPECS,features,ExtendedEncoding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/feature_extension')
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'experiment_manifest.json').read_text());winner=manifest['selected_on_2016']
    spec=next(s for s in SPECS if s['name']==winner)
    m=importlib.util.spec_from_file_location('ivy_original_utilities',PROJECT_DIR/'analysis/05_validation_member5.py')
    ivy=importlib.util.module_from_spec(m);m.loader.exec_module(ivy)
    frame=pd.read_csv(args.input,usecols=READ_COLUMNS,low_memory=False);dates=pd.to_datetime(frame.issue_date)
    train=frame.loc[dates.lt('2017-01-01')];validation=frame.loc[dates.ge('2017-01-01')&dates.lt('2018-01-01')];test=frame.loc[dates.ge('2018-01-01')]
    ft,fv,fe=features(train),features(validation),features(test)
    prep=TrainOnlyPreprocessor().fit(ft);encoder=ExtendedEncoding().fit(prep,ft,spec['proxy'])
    model=estimator(spec,5106,encoder.mask);model.fit(encoder.transform(ft),train.default)
    pv=model.predict_proba(encoder.transform(fv))[:,1];p=model.predict_proba(encoder.transform(fe))[:,1]
    current=binary_metrics(test.default,p)
    expected=pd.read_csv(out/'comparison_2018.csv').set_index('candidate').loc[winner]
    if not all(np.isclose(current[k],expected[k],rtol=1e-8,atol=1e-10) for k in current):
        raise ValueError('Repeated feature-extension fit does not match saved metrics')
    print('Repeated fit matches the saved feature-extension metrics.',flush=True)
    handoff=CLEANED_DATA_DIR/'feature_extension_handoff';handoff.mkdir(parents=True,exist_ok=True)
    def prediction_table(f,prob):
        return pd.DataFrame({'id':f.id.to_numpy(),'issue_date':f.issue_date.to_numpy(),
            'default':f.default.to_numpy(),'model':winner,'predicted_probability':prob,'selected_final_model':True})
    preds=prediction_table(test,p)
    for name,f,prob in [('validation',validation,pv),('test',test,p)]:
        prediction_table(f,prob).to_csv(handoff/f'{name}_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    subgroup_cols=['grade','term_months','home_ownership','purpose','verification_status','application_type','emp_length_years_was_missing']
    cohort=frame.assign(issue_dt=dates,id_key=ivy._norm_id(frame.id))
    enriched,identity=ivy.verify_predictions(cohort,preds,'2018-01-01',subgroup_cols)
    overall=ivy.bootstrap_metrics(test.default.to_numpy(),p,1000,5106,calibration=True)
    pd.DataFrame([{'model':winner,**overall}]).to_csv(out/'validation_overall_metrics.csv',index=False)
    print('Overall bootstrap and calibration completed.',flush=True)
    subgroups=ivy.evaluate_subgroups(enriched,subgroup_cols,200,30,500,5106)
    subgroups.to_csv(out/'validation_subgroups.csv',index=False)
    print('Subgroup validation completed.',flush=True)
    original=pd.read_csv(TABLES_DIR/'out_of_time_predictions.csv');baseline_id='hist_gradient_boosting_leaves31_none'
    baseline=original.loc[original.model.eq(baseline_id)].set_index('id').reindex(test.id)
    if not np.array_equal(baseline.default.to_numpy(),test.default.to_numpy()):raise ValueError('Baseline outcome alignment mismatch')
    p0=baseline.predicted_probability.to_numpy();y=test.default.to_numpy()
    # These operating thresholds were frozen in earlier stages, not selected again here.
    old_policy=pd.read_csv(PROJECT_DIR/'outputs/experiments/frozen_2018_assessment/thresholds_2018.csv')
    old_rules=old_policy.loc[old_policy.candidate.eq('original_baseline')].set_index('rule').threshold.to_dict()
    new_rules=manifest['thresholds_selected_on_2017']
    rng=np.random.default_rng(5106);intervals=[];threshold_rows=[]
    for rule,t1 in new_rules.items():
        t0=float(old_rules[rule]);a=threshold_metrics(y,p0,t0);b=threshold_metrics(y,p,t1)
        threshold_rows.extend([{'candidate':'original_baseline','rule':rule,**a},{'candidate':winner,'rule':rule,**b}])
        draws={k:[] for k in ['precision','recall','f1']}
        for _ in range(1000):
            i=rng.integers(0,len(y),size=len(y));v0=threshold_metrics(y[i],p0[i],t0);v1=threshold_metrics(y[i],p[i],t1)
            for k in draws:draws[k].append(v1[k]-v0[k])
        for k in draws:
            low,high=np.quantile(draws[k],[.025,.975]);intervals.append({'rule':rule,'metric':k,'delta':b[k]-a[k],'ci_low':low,'ci_high':high})
    pd.DataFrame(intervals).to_csv(out/'paired_threshold_deltas.csv',index=False)
    pd.DataFrame(threshold_rows).to_csv(out/'validated_thresholds.csv',index=False)
    grid=np.round(np.linspace(.01,.8,159),6)
    curve=pd.DataFrame([ivy._threshold_stats(validation.default.to_numpy(),pv,float(t),5.) for t in grid])
    threshold_plot=pd.DataFrame([{'operating_goal':rule,**ivy._threshold_stats(y,p,float(t),5.)} for rule,t in new_rules.items()])
    baseline_plot=original.loc[original.model.eq(baseline_id)]
    bins=ivy.generate_figures(pd.concat([baseline_plot,preds],ignore_index=True),winner,curve,threshold_plot,out/'figures')
    bins.to_csv(out/'validation_calibration_bins.csv',index=False)
    audit={'status':'completed','candidate':winner,'identity':{k:str(v) if isinstance(v,pd.Timestamp) else v for k,v in identity.items()},
       'repeated_fit_matches':True,'overall_bootstrap':1000,'subgroup_bootstrap':500,'paired_threshold_bootstrap':1000,'seed':5106,
       'thresholds_frozen_before_this_run':True,'calibration_fitted_for_diagnostics_only':True,
       'ivy_code_modified':False,'attribution':'Original validation utilities by Ivy; follow-up extension implementation/execution assisted by Codex',
       'scope':'Exploratory reused-test evaluation; original primary analysis and Ivy report preserved',
       'additional_temporal_robustness':'Not run for this extension; original primary-model robustness does not transfer automatically'}
    (out/'validation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Follow-up validation completed; original Ivy code and outputs preserved.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
