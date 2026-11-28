"""XGBoost on the audited expanded grade predictors; select by late-2016 F1.

Training-only late-2015 early stopping sets round counts. Early-2016 thresholds are frozen
before late-2016 candidate comparison. Refit the selected XGBoost setting before 2017,
choose its final threshold on 2017, and assess reused 2018 once against the current best.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,hashlib,importlib.util,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR
from src.predictive_modelling import TrainOnlyPreprocessor,binary_metrics
from src.predictive_improvement import estimator,threshold_metrics,curve_summary
from src.predictive_extension import READ_COLUMNS,SPECS,features,ExtendedEncoding
from src.xgboost_benchmark import CANDIDATES,feature_types,make_model,f1_threshold,check_handoff,training_weight


def encode(train,*others):
    ft=features(train);prep=TrainOnlyPreprocessor().fit(ft)
    encoder=ExtendedEncoding().fit(prep,ft,'grade')
    arrays=[encoder.transform(ft),*[encoder.transform(features(f)) for f in others]]
    return encoder,arrays


def main(argv=None):
    import xgboost
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/xgboost_benchmark')
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    manifest={'status':'planned','candidates':CANDIDATES,'seed':5106,
       'features':'Same 27 expanded grade predictors and training-only transforms as expanded_grade_31',
       'round_selection':'Train before 2015-10-01; early stop on 2015 Q4 log loss, max 800 rounds, patience 40',
       'candidate_fit':'Refit each candidate before 2016 using its fixed training-only round count',
       'threshold_selection_for_candidate_comparison':'2016 H1 only; maximum F1; ties lowest threshold',
       'candidate_selection':'Maximum 2016 H2 F1 at frozen H1 threshold; AP then lower log loss break ties',
       'final_fit':'Before 2017; same selected rounds and training-derived class weight',
       'final_threshold_selection':'2017 only; maximum F1',
       'assessment':'2018 reused test; exploratory follow-up, not a fresh independent holdout',
       'bootstrap':'1000 paired individual-loan percentile draws conditional on frozen fitted models and thresholds',
       'fixed_settings':{'learning_rate':.05,'min_child_weight':20,'reg_lambda':10.,'reg_alpha':.1,'subsample':.85,'colsample_bytree':.9,'n_jobs':2},
       'versions':{'python':platform.python_version(),'xgboost':xgboost.__version__,'numpy':np.__version__,'pandas':pd.__version__},
       'primary_outputs_replaced':False,'current_reference':'expanded_grade_31'}
    path=out/'experiment_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    frame=pd.read_csv(args.input,usecols=READ_COLUMNS,low_memory=False);dates=pd.to_datetime(frame.issue_date)
    early=frame.loc[dates.lt('2015-10-01')];stop=frame.loc[dates.ge('2015-10-01')&dates.lt('2016-01-01')]
    train=frame.loc[dates.lt('2016-01-01')];threshold_set=frame.loc[dates.ge('2016-01-01')&dates.lt('2016-07-01')]
    select=frame.loc[dates.ge('2016-07-01')&dates.lt('2017-01-01')]
    if any(f.empty or f.default.nunique()!=2 for f in [early,stop,train,threshold_set,select]):
        raise ValueError('Every development partition must contain both outcome classes')
    manifest['development_counts']={k:len(f) for k,f in [('early_train',early),('early_stop',stop),('refit_train',train),('threshold_h1',threshold_set),('selection_h2',select)]}
    print('Development partitions:',manifest['development_counts'],flush=True)
    e0,(xearly,xstop)=encode(early,stop)
    enc,(xtrain,xthreshold,xselect)=encode(train,threshold_set,select)
    if e0.columns!=enc.columns:raise ValueError('Predictor schema changed between training stages')
    types=feature_types(enc.mask);rows=[]
    # Current best model gets the same independent threshold/selection split.
    baseline_spec=next(s for s in SPECS if s['name']=='expanded_grade_31')
    baseline=estimator(baseline_spec,5106,enc.mask);baseline.fit(xtrain,train.default)
    bp=baseline.predict_proba(xthreshold)[:,1];bthreshold=f1_threshold(threshold_set.default,bp)
    bp=baseline.predict_proba(xselect)[:,1]
    control={'candidate':'expanded_grade_31',**binary_metrics(select.default,bp),**threshold_metrics(select.default,bp,bthreshold)}
    pd.DataFrame([control]).to_csv(out/'development_reference.csv',index=False)
    print(f"Development reference: F1={control['f1']:.6f}, AP={control['pr_auc']:.6f}",flush=True)
    del baseline
    for spec in CANDIDATES:
        started=time.perf_counter();model=make_model(spec,800,feature_types(e0.mask),early.default,early_stopping=True)
        model.fit(xearly,early.default,eval_set=[(xstop,stop.default)],verbose=False)
        rounds=int(model.best_iteration)+1
        model=make_model(spec,rounds,types,train.default);model.fit(xtrain,train.default,verbose=False)
        p=model.predict_proba(xthreshold)[:,1];threshold=f1_threshold(threshold_set.default,p)
        p=model.predict_proba(xselect)[:,1]
        row={'candidate':spec['name'],'rounds':rounds,'training_positive_weight':training_weight(spec['weight'],train.default),
             **binary_metrics(select.default,p),**threshold_metrics(select.default,p,threshold),'seconds':round(time.perf_counter()-started,2)}
        rows.append(row);pd.DataFrame(rows).to_csv(out/'development_comparison.csv',index=False)
        print(f"{spec['name']}: H2 F1={row['f1']:.6f}, precision={row['precision']:.4f}, recall={row['recall']:.4f}, AP={row['pr_auc']:.6f}; rounds={rounds}; {row['seconds']}s",flush=True)
        del model
    ranked=pd.DataFrame(rows).sort_values(['f1','pr_auc','log_loss'],ascending=[False,False,True])
    winner=ranked.iloc[0];spec=next(s for s in CANDIDATES if s['name']==winner.candidate)
    manifest.update(status='selected_before_final_assessment',selected_candidate=winner.candidate,selected_rounds=int(winner.rounds),predictor_columns=enc.columns,
       source_hashes={str(f.relative_to(PROJECT_DIR)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),PROJECT_DIR/'src/xgboost_benchmark.py',PROJECT_DIR/'src/predictive_extension.py']})
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen XGBoost choice:',winner.candidate,flush=True)
    final_train=frame.loc[dates.lt('2017-01-01')];validation=frame.loc[dates.ge('2017-01-01')&dates.lt('2018-01-01')];test=frame.loc[dates.ge('2018-01-01')]
    enc,(xt,xv,xe)=encode(final_train,validation,test)
    model=make_model(spec,int(winner.rounds),feature_types(enc.mask),final_train.default);model.fit(xt,final_train.default,verbose=False)
    pv=model.predict_proba(xv)[:,1];threshold=f1_threshold(validation.default,pv)
    manifest.update(status='final_model_and_threshold_frozen',final_threshold=threshold,
       final_training_positive_weight=training_weight(spec['weight'],final_train.default))
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    p=model.predict_proba(xe)[:,1]
    artifact=PROJECT_DIR/'artifacts/xgboost_benchmark';artifact.mkdir(parents=True,exist_ok=True)
    model.save_model(artifact/'model.ubj')
    restored=xgboost.XGBClassifier();restored.load_model(artifact/'model.ubj')
    replay=restored.predict_proba(xe)[:,1]
    if not np.allclose(p,replay,rtol=1e-7,atol=1e-8):raise ValueError('Model reload prediction mismatch')
    handoff=CLEANED_DATA_DIR/'xgboost_handoff';handoff.mkdir(parents=True,exist_ok=True)
    for name,f,probs in [('validation',validation,pv),('test',test,p)]:
        pd.DataFrame({'id':f.id.to_numpy(),'issue_date':f.issue_date.to_numpy(),'default':f.default.to_numpy(),
            'model':winner.candidate,'predicted_probability':probs,'selected_final_model':True}).to_csv(handoff/f'{name}_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    base_handoff=CLEANED_DATA_DIR/'feature_extension_handoff'
    p0=check_handoff(pd.read_csv(base_handoff/'test_predictions.csv.gz'),test)
    pv0=check_handoff(pd.read_csv(base_handoff/'validation_predictions.csv.gz'),validation)
    threshold0=f1_threshold(validation.default,pv0);y=test.default.to_numpy()
    comparison=[]
    for name,probs,t in [('expanded_grade_31',p0,threshold0),(winner.candidate,p,threshold)]:
        comparison.append({'candidate':name,'cohort':'2018 reused test','rows':len(test),**binary_metrics(y,probs),**threshold_metrics(y,probs,t),**curve_summary(y,probs)})
    pd.DataFrame(comparison).to_csv(out/'comparison_2018.csv',index=False)
    rng=np.random.default_rng(5106);draws=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),size=len(y));a=threshold_metrics(y[i],p0[i],threshold0);b=threshold_metrics(y[i],p[i],threshold)
        draws.append([b['precision']-a['precision'],b['recall']-a['recall'],b['f1']-a['f1'],
            average_precision_score(y[i],p[i])-average_precision_score(y[i],p0[i]),roc_auc_score(y[i],p[i])-roc_auc_score(y[i],p0[i])])
    intervals=[]
    for j,key in enumerate(['precision','recall','f1','pr_auc','roc_auc']):
        low,high=np.quantile(np.asarray(draws)[:,j],[.025,.975]);intervals.append({'metric':key,'delta':comparison[1][key]-comparison[0][key],'ci_low':low,'ci_high':high})
    pd.DataFrame(intervals).to_csv(out/'paired_differences.csv',index=False)
    m=importlib.util.spec_from_file_location('ivy_xgboost_diagnostics',PROJECT_DIR/'analysis/05_validation_member5.py')
    ivy=importlib.util.module_from_spec(m);m.loader.exec_module(ivy)
    pd.DataFrame([{'candidate':winner.candidate,**ivy.calibration_stats(y,p)}]).to_csv(out/'calibration.csv',index=False)
    manifest.update(status='completed',model_reload_predictions_match=True,reference_prediction_identity_checks_passed=True,
       development_f1_exceeds_reference=bool(winner.f1>control['f1']),test_f1_delta=comparison[1]['f1']-comparison[0]['f1'])
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(comparison)[['candidate','precision','recall','f1','pr_auc','roc_auc']].to_string(index=False),flush=True)
    print('Completed: original study, Ivy outputs and feature-extension results were preserved.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
