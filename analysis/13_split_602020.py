"""Exploratory chronological 60/20/20 refits; preserve all earlier study outputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR, CLEANED_DATA_DIR
from src.proportional_split import frozen_602020
from src.predictive_modelling import TrainOnlyPreprocessor, binary_metrics
from src.predictive_extension import READ_COLUMNS, SPECS, features, ExtendedEncoding
from src.predictive_improvement import estimator, threshold_metrics
from src.xgboost_benchmark import make_model, feature_types, f1_threshold


def encode(train, *others):
    f = features(train)
    prep = TrainOnlyPreprocessor().fit(f)
    enc = ExtendedEncoding().fit(prep, f, 'grade')
    return enc, [enc.transform(f), *[enc.transform(features(p)) for p in others]]


def main():
    import xgboost
    out = PROJECT_DIR/'outputs/experiments/split_602020'
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CLEANED_DATA_DIR/'clean_v1/lending_club_clean.csv.gz', usecols=READ_COLUMNS, low_memory=False)
    train, valid, test = frozen_602020(frame, out/'frozen_split_contract.json')
    handoff = CLEANED_DATA_DIR/'split_602020_handoff'
    handoff.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{'partition': name, 'rows': len(p), 'fraction': len(p)/len(frame),
        'minimum_issue_date': p.issue_date.min(), 'maximum_issue_date': p.issue_date.max(),
        'default_count': int(p.default.sum()), 'default_rate': p.default.mean()}
        for name, p in [('train', train), ('validation', valid), ('test', test)]])
    summary.to_csv(out/'split.csv', index=False)
    manifest = {'status': 'planned', 'split': 'Chronological 60/20/20 by rows; floor(.6*N) and floor(.8*N) boundaries',
        'month_ties': 'Fixed pandas hash of loan ID then ID; independent of outcomes; a boundary month can span partitions',
        'seed': 5106, 'python': platform.python_version(), 'pandas': pd.__version__, 'xgboost': xgboost.__version__,
        'selection': 'Two fixed previously explored architectures; choose model and maximum-F1 threshold on validation only',
        'xgb_rounds': 'Before outer validation: early stop on latest whole months covering approximately 10% of training; max800, patience40; refit full60%',
        'limitations': 'Exploratory repartition of previously used data; no fresh independent holdout; prior architecture choices used overlapping cohorts',
        'primary_outputs_replaced': False}
    path = out/'experiment_manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.to_string(index=False), flush=True)
    # Whole-month stopping block stays entirely inside the outer training set.
    dates = pd.to_datetime(train.issue_date)
    cutoff = dates.iloc[int(.9*len(train))]
    early, stop = train.loc[dates.lt(cutoff)], train.loc[dates.ge(cutoff)]
    if early.default.nunique()!=2 or stop.default.nunique()!=2:
        raise ValueError('Inner training/early-stop blocks need both classes')
    enc0, (xe, xs) = encode(early, stop)
    spec = {'name': 'xgb_depth5_double', 'max_depth': 5, 'weight': 'double'}
    xgb = make_model(spec, 800, feature_types(enc0.mask), early.default, early_stopping=True)
    xgb.fit(xe, early.default, eval_set=[(xs, stop.default)], verbose=False)
    rounds = int(xgb.best_iteration)+1
    print(f'XGBoost training-only early stopping chose {rounds} rounds', flush=True)
    del xgb, xe, xs
    enc, (xt, xv, xx) = encode(train, valid, test)
    hgb_spec = next(s for s in SPECS if s['name']=='expanded_grade_31')
    models = [('expanded_grade_31', estimator(hgb_spec, 5106, enc.mask)),
              ('xgb_depth5_double', make_model(spec, rounds, feature_types(enc.mask), train.default))]
    rows, predictions = [], {}
    for name, model in models:
        print(f'Fitting {name}', flush=True)
        model.fit(xt, train.default)
        pv = model.predict_proba(xv)[:, 1]
        valid[['id','issue_date','default']].assign(model=name, predicted_probability=pv.astype('float64')).to_csv(
            handoff/f'{name}_validation.csv.gz', index=False, float_format='%.17g', compression={'method':'gzip','mtime':0})
        threshold = f1_threshold(valid.default, pv)
        rows.append({'candidate': name, **binary_metrics(valid.default, pv), **threshold_metrics(valid.default, pv, threshold)})
        predictions[name] = (model, threshold)
    validation = pd.DataFrame(rows)
    validation.to_csv(out/'validation_comparison.csv', index=False)
    winner = validation.sort_values(['f1','pr_auc','log_loss'], ascending=[False,False,True]).iloc[0].candidate
    manifest.update(status='frozen_before_test', selected_on_validation=winner,
        xgb_selected_rounds=rounds, inner_stop_cutoff=str(cutoff.date()),
        predictors=enc.columns, thresholds={k: float(v[1]) for k,v in predictions.items()},
        source_hashes={str(p.relative_to(PROJECT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), PROJECT_DIR/'src/proportional_split.py', PROJECT_DIR/'src/predictive_extension.py']})
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Frozen validation choice: {winner}', flush=True)
    rows = []
    for name, (model, threshold) in predictions.items():
        p = model.predict_proba(xx)[:, 1]
        rows.append({'candidate': name, 'selected_on_validation': name==winner,
            **binary_metrics(test.default, p), **threshold_metrics(test.default, p, threshold)})
        test[['id','issue_date','default']].assign(model=name, predicted_probability=p.astype('float64')).to_csv(
            handoff/f'{name}_test.csv.gz', index=False, float_format='%.17g', compression={'method':'gzip','mtime':0})
    result = pd.DataFrame(rows)
    result.to_csv(out/'test_comparison.csv', index=False)
    manifest['status'] = 'completed'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    report = ['# Chronological 60/20/20 experiment', '',
        'The original study and Ivy outputs remain preserved. This is a separate exploratory split comparison, not the final group report.', '',
        'Earliest 60% of resolved loans train the models; next 20% choose the F1 threshold and model; latest 20% assess performance. Rounded row counts are exhaustive and disjoint. Monthly date ties are ordered by a fixed hash of loan ID and then ID, without using outcomes. Boundary months may span adjacent partitions; this is not a strict whole-month holdout.', '',
        summary.to_csv(index=False), '',
        f'Both models use the same 27 expanded predictors and training-fitted preprocessing. XGBoost depth 5 and positive weight 2 were fixed before this run; training-only early stopping chose {rounds} rounds. HGB uses the prior expanded specification and its internal training-only stopping subset. No outer validation/test records fit either encoder or model.', '',
        f'Model chosen by validation F1 (AP then lower log loss break ties): **{winner}**. Thresholds maximize validation F1. Test results did not determine this choice.', '',
        '## Test results', '', result.to_csv(index=False), '',
        'Do not directly attribute changes from the previous split to a better model: the test period, sample size and default prevalence changed. This experiment repartitions data already used in previous searches, and even the architecture choices were informed by overlapping cohorts. It supplies no fresh confirmatory test. Resolved-loan selection and incomplete follow-up limitations remain.', '',
        'Reproduce: install requirements-xgboost.txt (macOS needs libomp), then run `python analysis/13_split_602020.py`. Aggregate results are in outputs/experiments/split_602020; loan-level handoffs remain ignored under data/cleaned. Earlier dashboard results retain their original split. Code and execution assisted by Codex.']
    (PROJECT_DIR/'report/split_602020_experiment.md').write_text('\n'.join(report)+'\n')
    print(result[['candidate','precision','recall','f1','pr_auc','roc_auc']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
