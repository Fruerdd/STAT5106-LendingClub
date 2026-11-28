"""Export aggregate dashboard data from the project's existing verified outputs.

Run from the repository root: .venv/bin/python dashboard/build_data.py
Individual loan IDs, borrower records and row-level predictions are never exported.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'dashboard/dist/assets'
TABLES = ROOT/'outputs/tables'


def records(name):
    frame = pd.read_csv(TABLES/name)
    return json.loads(frame.to_json(orient='records'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = {
        'grade': 'default_rate_by_grade.csv', 'vintage': 'default_rate_by_issue_quarter.csv',
        'fico': 'default_rate_by_fico_score_bins.csv', 'income': 'default_rate_by_annual_inc_bins.csv',
        'interest': 'default_rate_by_int_rate_bins.csv', 'purpose': 'default_rate_by_purpose.csv',
        'missingness': 'missingness.csv', 'numeric': 'eda_numeric_summary.csv',
        'inference': 'inference_odds_ratios.csv', 'inferenceSamples': 'inference_sample_summary.csv',
        'modelComparison': 'model_comparison.csv', 'split': 'temporal_split.csv',
        'metrics': 'validation_overall_metrics.csv', 'calibration': 'validation_calibration_bins.csv',
        'subgroups': 'validation_subgroup_performance.csv', 'thresholds': 'validation_threshold_performance.csv',
        'thresholdCurve': 'validation_threshold_curve_validation.csv',
        'paired': 'validation_paired_model_deltas.csv', 'robustness': 'validation_robustness_checks.csv',
        'reproduction': 'reproducibility_checks.csv',
    }
    data = {key: records(name) for key, name in names.items()}
    data['cleaning'] = json.loads((TABLES/'cleaning_validation.json').read_text())
    audit = json.loads((TABLES/'reproducibility_audit.json').read_text())
    data['audit'] = {key: audit[key] for key in ['complete', 'dataset_sha256', 'source_commit', 'csv_rtol', 'csv_atol']}
    test_log = Path(audit['work_directory'])/'logs/tests.log'
    match = re.search(r'(\d+) passed', test_log.read_text()) if test_log.is_file() else None
    data['audit']['test_count'] = int(match.group(1)) if match else None
    data['audit']['test_count_source'] = 'Member 6 fresh-environment test log'
    data['audit']['checks'] = len(audit['checks'])
    data['audit']['passed'] = sum(c['passed'] for c in audit['checks'])
    data['audit']['scope'] = 'Separate local clone; isolated environment; prior-result comparison; additional EDA repeat. External remote cloning and a second complete pipeline repeat were not tested.'
    data['sourceDate'] = '26 November 2026'
    data['sources'] = names
    data['curves'] = {}
    predictions = pd.read_csv(TABLES/'out_of_time_predictions.csv', usecols=['model','default','predicted_probability'])
    for model, group in predictions.groupby('model', sort=True):
        x,y,_ = roc_curve(group.default, group.predicted_probability)
        precision,recall,_ = precision_recall_curve(group.default, group.predicted_probability)
        def sample(a,b):
            indices = np.unique(np.linspace(0,len(a)-1,min(240,len(a))).astype(int))
            return [[round(float(a[i]),6),round(float(b[i]),6)] for i in indices]
        data['curves'][model] = {'roc': sample(x,y), 'pr': sample(recall,precision)}
    target = OUT/'tables';target.mkdir(exist_ok=True)
    for name in names.values():
        shutil.copy2(TABLES/name, target/name)
    extension = ROOT/'outputs/experiments/feature_extension'
    if (extension/'validation_audit.json').is_file():
        files = {'comparison':'comparison_2018.csv', 'metrics':'validation_overall_metrics.csv',
                 'policy':'validated_thresholds.csv', 'paired':'paired_threshold_deltas.csv',
                 'subgroups':'validation_subgroups.csv', 'calibration':'validation_calibration_bins.csv',
                 'tuning':'tuning_comparison.csv'}
        improvement = {key: json.loads(pd.read_csv(extension/name).to_json(orient='records'))
                       for key,name in files.items()}
        metadata = json.loads((extension/'experiment_manifest.json').read_text())
        improvement['candidate'] = metadata['selected_on_2016']
        improvement['ap_delta'] = metadata['ap_delta']
        improvement['ap_delta_ci'] = metadata['ap_delta_ci']
        improvement['sources'] = {key:'improvement_'+name for key,name in files.items()}
        proof = {}
        for key,name in files.items():
            alias = improvement['sources'][key]
            shutil.copy2(extension/name,target/alias)
            proof[alias] = {'source':str((extension/name).relative_to(ROOT)),
                            'sha256':hashlib.sha256((extension/name).read_bytes()).hexdigest()}
        handoff = ROOT/'data/cleaned/feature_extension_handoff/test_predictions.csv.gz'
        if handoff.is_file():
            values = pd.read_csv(handoff,usecols=['default','predicted_probability'])
            x,y,_ = roc_curve(values.default,values.predicted_probability)
            precision,recall,_ = precision_recall_curve(values.default,values.predicted_probability)
            improvement['curves'] = {'roc':sample(x,y),'pr':sample(recall,precision)}
        data['improvement'] = improvement
        (OUT/'improvement-source-manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    xgb_dir = ROOT/'outputs/experiments/xgboost_benchmark'
    xgb_manifest = xgb_dir/'experiment_manifest.json'
    if xgb_manifest.is_file() and json.loads(xgb_manifest.read_text()).get('status') == 'completed':
        meta = json.loads(xgb_manifest.read_text())
        files = {'comparison':'comparison_2018.csv','paired':'paired_differences.csv',
                 'development':'development_comparison.csv','reference':'development_reference.csv',
                 'calibration':'calibration.csv'}
        bundle = {key:json.loads(pd.read_csv(xgb_dir/name).to_json(orient='records')) for key,name in files.items()}
        bundle['candidate'] = meta['selected_candidate']
        bundle['rounds'] = meta['selected_rounds']
        bundle['weight'] = meta['final_training_positive_weight']
        bundle['version'] = meta['versions']['xgboost']
        bundle['reload_matches'] = meta['model_reload_predictions_match']
        bundle['sources'] = {key:'xgboost_'+name for key,name in files.items()}
        proof = {}
        for key,name in files.items():
            alias = bundle['sources'][key]
            shutil.copy2(xgb_dir/name,target/alias)
            proof[alias] = {'source':str((xgb_dir/name).relative_to(ROOT)),
                            'sha256':hashlib.sha256((xgb_dir/name).read_bytes()).hexdigest()}
        data['xgboost'] = bundle
        (OUT/'xgboost-source-manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    # Strict JSON: nonfinite CSV reference thresholds are represented as null.
    encoded = json.dumps(data, allow_nan=False, separators=(',',':'))
    (OUT/'results.json').write_text(encoded+'\n')
    (OUT/'results.js').write_text('window.RESEARCH_DATA = '+encoded+';\n')
    manifest = {name: hashlib.sha256((TABLES/name).read_bytes()).hexdigest() for name in names.values()}
    (OUT/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Exported {len(names)} aggregate tables and downsampled ROC/PR curves. No row-level records.')

if __name__ == '__main__':
    main()
