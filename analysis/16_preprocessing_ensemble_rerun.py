"""Full reference refit, three preprocessing alternatives, conditional calibrated equal blend."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score
if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR
from src.proportional_split import frozen_602020
from src.predictive_extension import READ_COLUMNS,SPECS
from src.predictive_modelling import binary_metrics,NUMERIC_FEATURES
from src.predictive_improvement import estimator,threshold_metrics
from src.xgboost_benchmark import f1_threshold,check_handoff
from src.scoped_tuning import training_folds,cohort_signature
from src.preprocessing_experiment import (MODES,FAMILIES,AblationEncoder,SigmoidCalibration,
    boost,fit_stopped,split_calibration_threshold,complementary,write_predictions)


def metrics(name,frame,p,t,**extras):
    return {'candidate':name,**extras,**binary_metrics(frame.default,p),**threshold_metrics(frame.default,p,t)}


def main(argv=None):
    import xgboost,sklearn,joblib
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/preprocessing_ensemble_602020')
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    contract=PROJECT_DIR/'outputs/experiments/split_602020/frozen_split_contract.json'
    if not contract.exists():raise FileNotFoundError('Existing frozen split contract required')
    manifest={'status':'registered_before_fitting','modes':list(MODES),'families':list(FAMILIES),'seed':5106,
        'scope':'Existing sources, target and frozen 60/20/20 cohort; numerical preparation only; grade proxy retained',
        'training':'Two forward folds inside training60%; inner latest-month early stopping; refit fold fit block at fixed rounds',
        'calibration':'Split each subsequent threshold block by whole months: earlier half fits sigmoid on logit score; later half chooses F1 threshold; score on separate later block',
        'candidates':'For each of 3 modes: calibrated xgb, calibrated hgb, and conditional equal blend (max9); no hyperparameter grid',
        'ensemble_gate':'Both training folds: disagreement>=1% and each member uniquely correct>=0.5% on later fold score block',
        'choice':'Highest mean forward F1, then AP, then lower log loss; freeze one candidate before validation',
        'final_fit':'Refit constituent models on entire60%, rounds=rounded median training fold stopping rounds; sigmoid fitted on pooled out-of-sample training calibration predictions',
        'final_choice':'Compare one training-selected candidate with fully rerun current XGBoost by validation F1, AP, lower log loss; freeze threshold before scoring test',
        'test':'Selected choice only plus rerun reference; previous test reuse makes assessment exploratory',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
        'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),
        'source_hashes':{str(p.relative_to(PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PROJECT_DIR/'src/preprocessing_experiment.py',PROJECT_DIR/'src/predictive_modelling.py',PROJECT_DIR/'src/predictive_extension.py',PROJECT_DIR/'src/xgboost_benchmark.py',PROJECT_DIR/'src/scoped_tuning.py']}}
    path=out/'experiment_manifest.json'
    def save():path.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    frame=pd.read_csv(CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz',usecols=READ_COLUMNS,low_memory=False)
    train,valid,test=frozen_602020(frame,contract)
    print('Frozen split verified:',len(train),len(valid),len(test),flush=True)
    local=CLEANED_DATA_DIR/'preprocessing_ensemble_602020_handoff';local.mkdir(parents=True,exist_ok=True)
    art=PROJECT_DIR/'artifacts/preprocessing_ensemble_602020';art.mkdir(parents=True,exist_ok=True)
    # Full reference refit, independent of cached prediction values.
    dates=pd.to_datetime(train.issue_date);cutoff=dates.iloc[int(.9*len(train))]
    early,stop=train.loc[dates.lt(cutoff)],train.loc[dates.ge(cutoff)]
    enc,reference,rounds,(reference_pv,)=fit_stopped('xgb','current',train,early,stop,valid)
    old_manifest=json.loads((PROJECT_DIR/'outputs/experiments/split_602020/experiment_manifest.json').read_text())
    if rounds!=old_manifest['xgb_selected_rounds']:raise ValueError('Reference stopping rounds changed unexpectedly')
    tr=f1_threshold(valid.default,reference_pv)
    reference_pt=reference.predict_proba(enc.transform(test))[:,1]
    baseline_hgb=estimator(next(s for s in SPECS if s['name']=='expanded_grade_31'),5106,enc.mask)
    baseline_hgb.fit(enc.transform(train),train.default)
    reproduction=[];old=CLEANED_DATA_DIR/'split_602020_handoff'
    for name,m in [('xgb_depth5_double',reference),('expanded_grade_31',baseline_hgb)]:
        for cohort,f in [('validation',valid),('test',test)]:
            p=m.predict_proba(enc.transform(f))[:,1]
            cache=check_handoff(pd.read_csv(old/f'{name}_{cohort}.csv.gz'),f)
            if not np.allclose(p,cache,rtol=1e-7,atol=5e-8):raise ValueError('Reference prediction reproduction failed')
            t=f1_threshold(valid.default,m.predict_proba(enc.transform(valid))[:,1])
            flags=p>=t;s=threshold_metrics(f.default,p,t)
            if not np.allclose([s['f1'],s['precision'],s['recall']],
                [f1_score(f.default,flags),precision_score(f.default,flags),recall_score(f.default,flags)],rtol=0,atol=1e-12):raise ValueError('Independent metric parity failed')
            reproduction.append({'candidate':name,'cohort':cohort,'rows':len(f),'max_probability_difference':float(np.max(np.abs(p-cache))),
                'metric_parity':True,'identity_sha256':cohort_signature(f),**s})
    pd.DataFrame(reproduction).to_csv(out/'reference_reproduction.csv',index=False)
    print('Both reference models fully refitted; all four prediction/metric comparisons passed',flush=True)
    reference.save_model(art/'reference_xgb.ubj');joblib.dump(enc,art/'reference_encoder.joblib')
    del baseline_hgb
    foldrows=[];diversity=[];calibration_data={};round_data={};blocks=[]
    for fold,fit,threshold_block,score,early,stop in training_folds(train):
        calibration,threshold=split_calibration_threshold(threshold_block)
        for block,f in [('fit',fit),('early',early),('stop',stop),('calibration',calibration),('threshold',threshold),('score',score)]:
            blocks.append({'fold':fold,'block':block,'rows':len(f),'min_date':f.issue_date.min(),'max_date':f.issue_date.max(),'identity_sha256':cohort_signature(f)})
        pd.DataFrame(blocks).to_csv(out/'training_blocks.csv',index=False)
        for mode in MODES:
            predictions={};thresholds={}
            for family in FAMILIES:
                started=time.perf_counter()
                e,m,n,(pc,ph,ps)=fit_stopped(family,mode,fit,early,stop,calibration,threshold,score)
                c=SigmoidCalibration().fit(calibration.default,pc)
                ph,ps=c.transform(ph),c.transform(ps)
                t=f1_threshold(threshold.default,ph)
                name=f'{mode}_{family}';predictions[family]=(ph,ps);thresholds[family]=t
                calibration_data.setdefault(name,[]).append((calibration.default.to_numpy(),pc.astype(float)))
                round_data.setdefault(name,[]).append(n)
                r=metrics(name,score,ps,t,fold=fold,rounds=n,seconds=round(time.perf_counter()-started,2))
                foldrows.append(r)
                print(f"{fold} {name}: F1={r['f1']:.5f}, AP={r['pr_auc']:.5f}, rounds={n}, {r['seconds']}s",flush=True)
                del m,e
            d=complementary(score.default,predictions['xgb'][1],predictions['hgb'][1],thresholds['xgb'],thresholds['hgb'])
            diversity.append({'mode':mode,'fold':fold,**d})
            ph=(predictions['xgb'][0]+predictions['hgb'][0])/2;ps=(predictions['xgb'][1]+predictions['hgb'][1])/2
            foldrows.append(metrics(f'{mode}_blend',score,ps,f1_threshold(threshold.default,ph),fold=fold,rounds=0,seconds=0))
            pd.DataFrame(foldrows).to_csv(out/'training_comparison.csv',index=False)
            pd.DataFrame(diversity).to_csv(out/'ensemble_diversity.csv',index=False)
    diversity_frame=pd.DataFrame(diversity)
    eligible={m:bool(g.ensemble_eligible.all()) for m,g in diversity_frame.groupby('mode')}
    table=pd.DataFrame(foldrows)
    ranking=table.groupby('candidate',as_index=False).agg(mean_f1=('f1','mean'),mean_pr_auc=('pr_auc','mean'),mean_log_loss=('log_loss','mean'),folds=('fold','nunique'))
    ranking['eligible']=ranking.candidate.map(lambda n:eligible[n.removesuffix('_blend')] if n.endswith('_blend') else True)
    ranking=ranking.sort_values(['mean_f1','mean_pr_auc','mean_log_loss','candidate'],ascending=[False,False,True,True])
    ranking.to_csv(out/'training_ranking.csv',index=False)
    winner=ranking.loc[ranking.eligible].iloc[0].candidate
    mode=next(m for m in MODES if winner.startswith(m+'_') and winner[len(m)+1:] in {'xgb','hgb','blend'})
    families=FAMILIES if winner.endswith('_blend') else (winner.rsplit('_',1)[1],)
    manifest.update(status='training_choice_frozen',training_selected=winner,ensemble_eligibility=eligible)
    save();print('Training-only choice:',winner,flush=True)
    full_encoder=AblationEncoder(mode).fit(train);xt=full_encoder.transform(train);xv=full_encoder.transform(valid)
    fitted={};calibrators={};vprobs=[]
    for family in families:
        name=f'{mode}_{family}';n=int(np.rint(np.median(round_data[name])))
        m=boost(family,n,full_encoder.mask,train.default);m.fit(xt,train.default)
        y=np.concatenate([v[0] for v in calibration_data[name]]);p=np.concatenate([v[1] for v in calibration_data[name]])
        c=SigmoidCalibration().fit(y,p)
        fitted[family]=m;calibrators[family]=c;vprobs.append(c.transform(m.predict_proba(xv)[:,1]))
    pv=np.mean(vprobs,axis=0);t=f1_threshold(valid.default,pv)
    vrows=[metrics('rerun_current_xgb',valid,reference_pv,tr),metrics(winner,valid,pv,t)]
    vtable=pd.DataFrame(vrows);vtable.to_csv(out/'validation_comparison.csv',index=False)
    choice=vtable.sort_values(['f1','pr_auc','log_loss'],ascending=[False,False,True],kind='stable').iloc[0].candidate
    chosen_threshold=tr if choice=='rerun_current_xgb' else t
    manifest.update(status='choice_and_threshold_frozen_before_test',selected_on_validation=choice,threshold=chosen_threshold,
        predictor_columns=full_encoder.columns,rounds={f:int(np.rint(np.median(round_data[f'{mode}_{f}']))) for f in families},
        calibration_parameters={f:{'intercept':float(c.model.intercept_[0]),'slope':float(c.model.coef_[0,0])} for f,c in calibrators.items()})
    save();print('Frozen validation choice:',choice,flush=True)
    if choice=='rerun_current_xgb':p=reference_pt;chosen_pv=reference_pv
    else:
        xx=full_encoder.transform(test)
        p=np.mean([calibrators[f].transform(m.predict_proba(xx)[:,1]) for f,m in fitted.items()],axis=0);chosen_pv=pv
    rows=[metrics('rerun_current_xgb',test,reference_pt,tr)]
    if choice!='rerun_current_xgb':rows.append(metrics(choice,test,p,chosen_threshold))
    pd.DataFrame(rows).to_csv(out/'test_comparison.csv',index=False)
    for name,f,probs in [('validation',valid,chosen_pv),('test',test,p)]:
        target=local/f'{name}.csv.gz';write_predictions(f,probs,target,choice)
        replay=pd.read_csv(target,float_precision='round_trip').predicted_probability.to_numpy()
        if not np.array_equal(np.asarray(probs,dtype=float),replay):raise ValueError('Prediction export lost precision')
    joblib.dump({'encoder':full_encoder,'models':fitted,'calibrators':calibrators,'reference':reference,'reference_encoder':enc,
        'selected':choice,'threshold':chosen_threshold},art/'fitted_bundle.joblib')
    replay=joblib.load(art/'fitted_bundle.joblib')
    if choice=='rerun_current_xgb':rp=replay['reference'].predict_proba(replay['reference_encoder'].transform(test))[:,1]
    else:
        xx=replay['encoder'].transform(test)
        rp=np.mean([replay['calibrators'][f].transform(m.predict_proba(xx)[:,1]) for f,m in replay['models'].items()],axis=0)
    if not np.array_equal(p,rp):raise ValueError('Fitted bundle reload mismatch')
    # Conditional paired bootstrap for the frozen selection only.
    rng=np.random.default_rng(5106);y=test.default.to_numpy(dtype=bool);a=reference_pt>=tr;b=p>=chosen_threshold;draws=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),size=len(y));yi=y[i]
        def f1(flags):return 2*np.count_nonzero(flags&yi)/(np.count_nonzero(flags)+np.count_nonzero(yi))
        draws.append(f1(b[i])-f1(a[i]))
    lo,hi=np.quantile(draws,[.025,.975]);delta=threshold_metrics(y,p,chosen_threshold)['f1']-rows[0]['f1']
    pd.DataFrame([{'metric':'f1','delta':delta,'ci_low':lo,'ci_high':hi,'draws':1000}]).to_csv(out/'paired_f1_difference.csv',index=False)
    manifest.update(status='completed',reference_reproduced=True,prediction_export_exact=True,fitted_bundle_reload_exact=True,
        current_model_retained=choice=='rerun_current_xgb',f1_delta=delta,f1_difference_ci=[float(lo),float(hi)])
    save();print(vtable[['candidate','precision','recall','f1','pr_auc','brier_score']].to_string(index=False),flush=True)
    print(pd.DataFrame(rows)[['candidate','precision','recall','f1','pr_auc','brier_score']].to_string(index=False),flush=True)
    print(f'Paired F1 difference {delta:.6f} [{lo:.6f}, {hi:.6f}]',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
