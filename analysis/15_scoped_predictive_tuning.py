"""Small training-only forward search, frozen 60/20/20 split and existing origination fields."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR,CLEANED_DATA_DIR
from src.proportional_split import frozen_602020
from src.predictive_extension import READ_COLUMNS,features
from src.predictive_modelling import binary_metrics
from src.predictive_improvement import threshold_metrics
from src.xgboost_benchmark import f1_threshold,check_handoff
from src.operating_policy import validation_options
from src.scoped_tuning import CANDIDATES,training_folds,encode,model,rank_candidates,cohort_signature


def audit_sources(frame, train, valid, out):
    """Document availability; outer test features/outcomes do not guide feature selection."""
    dictionary=pd.read_csv(PROJECT_DIR/'outputs/tables/data_dictionary.csv').set_index('column')
    rows=[]
    for column in frame.columns:
        d=dictionary.loc[column] if column in dictionary.index else None
        rows.append({'column':column,'used_by_experiment':column in READ_COLUMNS,
            'role':'audit/target only' if column in ['id','issue_date','default'] else 'registered feature source' if column in READ_COLUMNS else 'not added',
            'primary_dictionary_permission':d['permitted_as_predictor'] if d is not None else '',
            'description':d['description'] if d is not None else '',
            'training_missing_fraction':float(train[column].isna().mean()),
            'validation_missing_fraction':float(valid[column].isna().mean()),
            'note':'Existing predictive extension; not an amendment of the primary inference contract' if column in ['installment','revol_bal','total_acc','addr_state','sub_grade','mths_since_last_delinq','mths_since_last_record'] else ''})
    pd.DataFrame(rows).to_csv(out/'feature_source_audit.csv',index=False)


def diagnostics(frame, probabilities, threshold, candidate):
    rows=[]
    for grouping in ['issue_year','grade','income_verification','structural_history_missing']:
        if grouping=='issue_year':groups=pd.to_datetime(frame.issue_date).dt.year.astype(str)
        elif grouping=='grade':groups=frame.grade.astype('string').fillna('MISSING')
        elif grouping=='income_verification':groups=frame.verification_status.astype('string').fillna('MISSING')
        else:groups=(frame.mths_since_last_delinq.isna()|frame.mths_since_last_record.isna()).map({True:'at_least_one_missing',False:'both_observed'})
        for value in sorted(groups.unique()):
            mask=groups.eq(value).to_numpy();y=frame.default.to_numpy()[mask];p=probabilities[mask]
            row={'candidate':candidate,'grouping':grouping,'group':value,'rows':len(y),'defaults':int(y.sum())}
            if np.unique(y).size==2:row.update(binary_metrics(y,p))
            row.update(threshold_metrics(y,p,threshold));rows.append(row)
    return rows


def main(argv=None):
    import xgboost
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROJECT_DIR/'outputs/experiments/scoped_tuning_602020')
    args=parser.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    contract=PROJECT_DIR/'outputs/experiments/split_602020/frozen_split_contract.json'
    if not contract.exists():raise FileNotFoundError('Frozen split contract required; do not create a new split')
    manifest={'status':'registered_before_search','candidates':CANDIDATES,'outer_split_contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),
        'scope':'Same resolved-loan cohort and existing registered origination-only feature extension; no new raw fields or inference changes',
        'training_search':'Two whole-month forward folds inside training60%; early stopping uses latest whole months of each fit block; thresholds use separate subsequent block; score uses later block',
        'search_choice':'Mean forward F1, then mean AP, then lower mean log loss; no outer validation/test input',
        'rounds':'Median of selected candidate training-only fold stopping rounds, rounded to nearest integer',
        'final_choice':'Compare one training-selected candidate with cached current XGBoost on outer validation F1, then AP, then lower log loss; threshold maximizes outer validation F1',
        'assessment':'Freeze final choice and threshold before selected model test scoring; previously examined test remains exploratory',
        'max_rounds':800,'seed':5106,'versions':{'python':platform.python_version(),'xgboost':xgboost.__version__,'pandas':pd.__version__,'numpy':np.__version__},
        'source_hashes':{str(p.relative_to(PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PROJECT_DIR/'src/scoped_tuning.py',PROJECT_DIR/'src/proportional_split.py',PROJECT_DIR/'src/predictive_extension.py']}}
    path=out/'experiment_manifest.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    frame=pd.read_csv(CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz',low_memory=False)
    train,valid,test=frozen_602020(frame,contract)
    audit_sources(frame,train,valid,out)
    print('Frozen split verified:',len(train),len(valid),len(test),flush=True)
    folds=[];results=[]
    for fold,fit,threshold_set,score,early,stop in training_folds(train):
        for block,p in [('fit',fit),('threshold',threshold_set),('score',score),('early_fit',early),('early_stop',stop)]:
            folds.append({'fold':fold,'block':block,'rows':len(p),'min_date':p.issue_date.min(),'max_date':p.issue_date.max(),'identity_sha256':cohort_signature(p)})
        pd.DataFrame(folds).to_csv(out/'training_folds.csv',index=False)
        print(f'{fold}: fit={len(fit)}, threshold={len(threshold_set)}, score={len(score)}',flush=True)
        for proxy in ['grade','sub_grade']:
            enc0,(xe,xs)=encode(early,stop,proxy=proxy)
            enc,(xt,xh,xx)=encode(fit,threshold_set,score,proxy=proxy)
            for spec in [s for s in CANDIDATES if s['proxy']==proxy]:
                started=time.perf_counter()
                m=model(spec,800,enc0.mask,early.default,early_stopping=True)
                m.fit(xe,early.default,eval_set=[(xs,stop.default)],verbose=False)
                rounds=int(m.best_iteration)+1
                m=model(spec,rounds,enc.mask,fit.default);m.fit(xt,fit.default,verbose=False)
                t=f1_threshold(threshold_set.default,m.predict_proba(xh)[:,1])
                p=m.predict_proba(xx)[:,1]
                row={'candidate':spec['name'],'fold':fold,'rounds':rounds,'seconds':round(time.perf_counter()-started,2),
                    **binary_metrics(score.default,p),**threshold_metrics(score.default,p,t)}
                results.append(row);pd.DataFrame(results).to_csv(out/'training_search.csv',index=False)
                print(f"{fold} {spec['name']}: F1={row['f1']:.5f}, AP={row['pr_auc']:.5f}, rounds={rounds}, {row['seconds']}s",flush=True)
                del m
            del xe,xs,xt,xh,xx
    ranked=rank_candidates(pd.DataFrame(results));ranked.to_csv(out/'training_ranking.csv',index=False)
    winner=ranked.iloc[0];spec=next(s for s in CANDIDATES if s['name']==winner.candidate)
    rounds=int(np.rint(winner.median_rounds))
    manifest.update(status='training_choice_frozen',training_selected=winner.candidate,final_rounds=rounds)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Training-only choice: {winner.candidate}, rounds={rounds}',flush=True)
    enc,(xt,xv)=encode(train,valid,proxy=spec['proxy'])
    selected=model(spec,rounds,enc.mask,train.default);selected.fit(xt,train.default,verbose=False)
    pv=selected.predict_proba(xv)[:,1];t=f1_threshold(valid.default,pv)
    handoff=CLEANED_DATA_DIR/'split_602020_handoff'
    baseline='current_xgb_depth5_double';p0v=check_handoff(pd.read_csv(handoff/'xgb_depth5_double_validation.csv.gz'),valid)
    t0=f1_threshold(valid.default,p0v)
    vrows=[]
    for name,p,threshold in [(baseline,p0v,t0),(winner.candidate,pv,t)]:
        vrows.append({'candidate':name,**binary_metrics(valid.default,p),**threshold_metrics(valid.default,p,threshold)})
    vtable=pd.DataFrame(vrows);vtable.to_csv(out/'validation_comparison.csv',index=False)
    choice=vtable.sort_values(['f1','pr_auc','log_loss'],ascending=[False,False,True],kind='stable').iloc[0].candidate
    chosen_threshold=t0 if choice==baseline else t
    manifest.update(status='choice_and_threshold_frozen_before_test',selected_on_validation=choice,selected_threshold=chosen_threshold,predictors=enc.columns)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Validation-frozen choice:',choice,flush=True)
    optionrows=[];diag=[]
    for name,p,threshold in [(baseline,p0v,t0),(winner.candidate,pv,t)]:
        optionrows.extend({'candidate':name,'cohort':'validation_only',**r} for r in validation_options(valid.default,p))
        diag.extend(diagnostics(valid,p,threshold,name))
    pd.DataFrame(optionrows).to_csv(out/'validation_operating_options.csv',index=False)
    pd.DataFrame(diag).to_csv(out/'validation_diagnostics.csv',index=False)
    # Test scores do not choose among search candidates or operating rules.
    p0=check_handoff(pd.read_csv(handoff/'xgb_depth5_double_test.csv.gz'),test)
    if choice==baseline:p=p0
    else:p=selected.predict_proba(enc.transform(features(test)))[:,1]
    test_rows=[]
    for name,probs,threshold in [(baseline,p0,t0),(choice,p,chosen_threshold)]:
        if any(r['candidate']==name for r in test_rows):continue
        test_rows.append({'candidate':name,'selected_on_validation':name==choice,**binary_metrics(test.default,probs),**threshold_metrics(test.default,probs,threshold)})
    pd.DataFrame(test_rows).to_csv(out/'test_comparison.csv',index=False)
    pd.DataFrame(diagnostics(test,p,chosen_threshold,choice)).to_csv(out/'selected_test_diagnostics.csv',index=False)
    local=CLEANED_DATA_DIR/'scoped_tuning_602020_handoff';local.mkdir(parents=True,exist_ok=True)
    for name,f,probs in [('validation',valid,p0v if choice==baseline else pv),('test',test,p)]:
        f[['id','issue_date','default']].assign(model=choice,predicted_probability=probs).to_csv(local/f'{name}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    if choice!=baseline:
        artifact=PROJECT_DIR/'artifacts/scoped_tuning_602020';artifact.mkdir(parents=True,exist_ok=True)
        selected.save_model(artifact/'selected_model.ubj')
    # Conditional paired F1 uncertainty; no retraining/search uncertainty is included.
    rng=np.random.default_rng(5106);y=test.default.to_numpy();a=p0>=t0;b=p>=chosen_threshold
    deltas=[]
    for _ in range(1000):
        i=rng.integers(0,len(y),size=len(y));yi=y[i].astype(bool)
        def f1(flags):
            tp=np.count_nonzero(flags & yi);denom=np.count_nonzero(flags)+np.count_nonzero(yi)
            return 2*tp/denom if denom else 0.
        deltas.append(f1(b[i])-f1(a[i]))
    lo,hi=np.quantile(deltas,[.025,.975]);delta=threshold_metrics(y,p,chosen_threshold)['f1']-threshold_metrics(y,p0,t0)['f1']
    pd.DataFrame([{'metric':'f1','delta':delta,'ci_low':lo,'ci_high':hi,'draws':1000}]).to_csv(out/'paired_f1_difference.csv',index=False)
    manifest.update(status='completed',current_model_retained=bool(choice==baseline),paired_f1_delta=delta,paired_f1_ci=[float(lo),float(hi)])
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(vtable[['candidate','precision','recall','f1','pr_auc']].to_string(index=False),flush=True)
    print(pd.DataFrame(test_rows)[['candidate','precision','recall','f1','pr_auc']].to_string(index=False),flush=True)
    print(f'Paired test F1 difference {delta:.6f} [{lo:.6f}, {hi:.6f}]',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
