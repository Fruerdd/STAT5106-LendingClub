"""Member 6: reproduce the six-member analysis in a separate local checkout.

Run: python analysis/06_reproducibility_member6.py --run
Creates a local clone with the current source changes overlaid, a new virtual environment,
and a read-only-by-convention symlink to the frozen raw file. All rerun outputs stay there.
The main checkout receives only the small reproduction report, JSON audit and check table.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR, RAW_DATA_PATH, TABLES_DIR


def sha256(path, compressed=False):
    digest = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare_csv(expected, actual, rtol=1e-6, atol=1e-10):
    """Compare rows/schema exactly and finite numeric results within stated tolerance."""
    a, b = pd.read_csv(expected), pd.read_csv(actual)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False, f'schema/shape mismatch: {a.shape} vs {b.shape}'
    maximum = 0.
    for column in a:
        x, y = a[column], b[column]
        if not x.isna().equals(y.isna()):
            return False, f'missingness differs: {column}'
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if pd.api.types.is_integer_dtype(x) and pd.api.types.is_integer_dtype(y):
                same = x.equals(y)
            else:
                same = np.allclose(x, y, equal_nan=True, rtol=rtol, atol=atol)
                difference = np.abs(x.to_numpy(float) - y.to_numpy(float))
                finite = difference[np.isfinite(difference)]
                if len(finite):
                    maximum = max(maximum, float(finite.max()))
            if not same:
                return False, f'numeric mismatch: {column}'
        elif not x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str)):
            return False, f'values differ: {column}'
    return True, f'maximum absolute numeric difference {maximum:.3g}'


def command(args, cwd, log, env):
    started = time.perf_counter()
    print('Running:', ' '.join(str(x) for x in args), flush=True)
    with log.open('w') as handle:
        completed = subprocess.run([str(x) for x in args], cwd=cwd, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
    return {'command': ' '.join(str(x) for x in args), 'log': log.name,
            'exit_code': completed.returncode, 'seconds': round(time.perf_counter()-started, 2)}


def source_files():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                                    cwd=PROJECT_DIR, text=True).splitlines()
    return [n for n in names if n.endswith('.py') or n in
            ['requirements.txt', 'requirements-reproducibility.txt', 'README.md', 'CONTRIBUTING.md', '.gitignore']]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run the full fresh-environment reproduction')
    parser.add_argument('--work-dir', type=Path, help='New empty directory; defaults to a temporary directory')
    parser.add_argument('--reuse-environment', type=Path, help='Resume using a prepared fresh environment; recorded explicitly')
    args = parser.parse_args(argv)
    if not args.run:
        parser.error('Use --run to execute the reproduction; no successful result is inferred from existing files')
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f'Frozen raw file not found: {RAW_DATA_PATH}')
    work = args.work_dir or Path(tempfile.mkdtemp(prefix='stat5106-reproduction-'))
    if args.work_dir:
        if work.exists() and any(work.iterdir()):
            raise ValueError('--work-dir must be empty')
        work.mkdir(parents=True, exist_ok=True)
    clone, logs = work/'checkout', work/'logs'
    logs.mkdir()
    env = os.environ.copy()
    env.update({'PYTHONUNBUFFERED': '1', 'MPLCONFIGDIR': str(work/'matplotlib'),
                'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    checks, commands = [], []
    def check(name, passed, detail):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})
        print(f"{'PASS' if passed else 'FAIL'}: {name}: {detail}", flush=True)
    audit = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT_DIR, text=True).strip(),
             'source_has_uncommitted_changes': bool(subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT_DIR,text=True)),
             'platform': platform.platform(), 'python_launcher': platform.python_version(),
             'dataset_file': RAW_DATA_PATH.name, 'dataset_sha256': sha256(RAW_DATA_PATH),
             'csv_rtol': 1e-6, 'csv_atol': 1e-10, 'commands': commands,
             'checkout_method': 'Local clone plus current source overlay; frozen data linked locally',
             'external_remote_clone_tested': False, 'repeat_scope': 'Fresh pipeline compared with prior saved results; EDA repeated byte-for-byte',
             'complete': False}
    failed = None
    try:
        record = command(['git', 'clone', '--no-hardlinks', '--quiet', PROJECT_DIR, clone], PROJECT_DIR, logs/'clone.log', env)
        commands.append(record)
        if record['exit_code']:
            raise RuntimeError('Local clone failed')
        files = source_files()
        for name in files:
            source = PROJECT_DIR/name
            if source.is_file():
                destination = clone/name; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        audit['source_sha256'] = {name: sha256(PROJECT_DIR/name) for name in files if (PROJECT_DIR/name).is_file()}
        # Remove tracked results from the clone before running; no stale result can pass as a rerun.
        for directory in [clone/'outputs/tables', clone/'outputs/figures']:
            for path in directory.iterdir():
                if path.is_file() and path.name not in {'.gitkeep', 'feature_specification.csv'}:
                    path.unlink()
        raw = clone/'data/raw'/RAW_DATA_PATH.name
        raw.symlink_to(RAW_DATA_PATH.resolve())
        if args.reuse_environment:
            python = args.reuse_environment.resolve()/'bin/python'
            if not python.is_file():
                raise FileNotFoundError('Prepared environment has no bin/python')
            audit['environment_setup'] = 'New isolated environment prepared before the reproduction command'
        else:
            record = command([sys.executable,'-m','venv',work/'venv'], work, logs/'venv.log', env); commands.append(record)
            if record['exit_code']:
                raise RuntimeError('Virtual environment creation failed')
            python = work/'venv/bin/python'
            record = command([python,'-m','pip','install','-r',clone/'requirements-reproducibility.txt'],clone,logs/'install.log',env);commands.append(record)
            if record['exit_code']:
                raise RuntimeError('Fresh dependency installation failed; see install.log')
            audit['environment_setup'] = 'Created isolated venv and installed requirements-reproducibility.txt'
        audit['software'] = subprocess.check_output([python,'-m','pip','freeze'],cwd=clone,env=env,text=True).splitlines()
        check('isolated_environment', subprocess.check_output([python,'-c','import sys; print(sys.prefix != sys.base_prefix)'],env=env,text=True).strip() == 'True', 'Separate virtual environment')
        stages = [('tests', ['-m','pytest','-q']),
                  ('cleaning', ['src/lending_club_data_cleaning.py','--input','data/raw/'+RAW_DATA_PATH.name,'--output-dir','data/cleaned/clean_v1']),
                  ('member1', ['analysis/01_cleaning_member2.py']),
                  ('member2', ['analysis/02_visualization_member3.py']),
                  ('member3', ['analysis/03_inference_member4.py']),
                  ('member4', ['analysis/04_modelling_pavel.py']),
                  ('member5', ['analysis/05_validation_member5.py'])]
        for name, arguments in stages:
            record = command([python,*arguments],clone,logs/f'{name}.log',env); commands.append(record)
            check(f'command_{name}', record['exit_code'] == 0, f"exit {record['exit_code']}; {record['log']}")
            if record['exit_code']:
                raise RuntimeError(f'{name} failed; see {record["log"]}')
        expected = PROJECT_DIR/'data/cleaned/clean_v1'
        actual = clone/'data/cleaned/clean_v1'
        check('cleaned_csv_content', sha256(expected/'lending_club_clean.csv.gz', True)==sha256(actual/'lending_club_clean.csv.gz', True), 'SHA-256 of decompressed CSV (gzip timestamps ignored)')
        check('cleaning_report',json.loads((expected/'cleaning_report.json').read_text())==json.loads((actual/'cleaning_report.json').read_text()), 'Exact rows, exclusions, duplicates, outcomes, dates, invalid values and missingness')
        for path in sorted((PROJECT_DIR/'outputs/tables').glob('*.csv')):
            if path.name.startswith('reproducibility_'):
                continue
            target = clone/'outputs/tables'/path.name
            if target.is_file():
                passed, detail = compare_csv(path,target)
                check('table_'+path.name,passed,detail)
            else:
                check('table_'+path.name,False,'No regenerated table')
        validation_manifest = json.loads((clone/'outputs/tables/validation_output_manifest.json').read_text())
        for name, digest in validation_manifest['files'].items():
            check('manifest_'+name,sha256(clone/name)==digest,'Generated output matches Ivy manifest')
        robustness = pd.read_csv(clone/'outputs/tables/validation_robustness_checks.csv')
        failures = robustness.loc[robustness.status.astype(str).str.contains('fail|error|not completed',case=False)]
        check('robustness_no_suppressed_failures',failures.empty,f'{len(failures)} failed robustness rows')
        # A same-input EDA repeat should reproduce the figures and tables exactly.
        before = {str(p.relative_to(clone)): sha256(p) for p in (clone/'outputs').rglob('*') if p.is_file()}
        record = command([python,'analysis/02_visualization_member3.py'],clone,logs/'member2_repeat.log',env);commands.append(record)
        check('repeat_eda_command',record['exit_code']==0,'Same input and environment')
        changed = [name for name,digest in before.items() if sha256(clone/name)!=digest]
        check('repeat_eda_outputs',not changed,'Changed files: '+str(changed))
        tracked = subprocess.check_output(['git','ls-files'],cwd=clone,text=True).splitlines()
        bad = [n for n in tracked if (n.startswith('data/raw/') and n not in ['data/raw/.gitkeep','data/raw/README.md']) or
               (n.startswith('data/cleaned/') and n!='data/cleaned/.gitkeep') or n.endswith(('.pkl','.joblib','.env'))]
        check('no_tracked_data_or_models',not bad,str(bad))
        mod = importlib.util.spec_from_file_location('ivy_validation_contract', clone/'analysis/05_validation_member5.py')
        module = importlib.util.module_from_spec(mod);mod.loader.exec_module(module);module.check_predictor_contract()
        check('predictor_leakage_contract',True,'Ivy contract rejects outcome/audit and post-origination fields')
        model_metadata = json.loads((clone/'outputs/tables/modelling_metadata.json').read_text())
        check('training_only_preprocessing_recorded',model_metadata['preprocessing_fitted_on']=='training partition only',model_metadata['selection_rule'])
    except Exception as exc:
        failed = f'{type(exc).__name__}: {exc}'
        check('pipeline_completion',False,failed)
    finally:
        check('raw_data_unchanged',sha256(RAW_DATA_PATH)==audit['dataset_sha256'],'Frozen raw checksum before/after')
        audit['complete'] = failed is None and all(r['passed'] for r in checks)
        audit['failure'] = failed
        audit['work_directory'] = str(work)
        audit['checks'] = checks
        audit['working_tree_status'] = subprocess.check_output(['git','status','--short'],cwd=PROJECT_DIR,text=True).splitlines()
        for record in commands:
            record['command'] = record['command'].replace(str(PROJECT_DIR), '<project>').replace(str(work), '<workdir>')
        TABLES_DIR.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(checks).to_csv(TABLES_DIR/'reproducibility_checks.csv',index=False)
        (TABLES_DIR/'reproducibility_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        lines = ['# Member 6 Reproducibility', '',
                 '**Result: '+('PASS' if audit['complete'] else 'INCOMPLETE OR FAILED')+'**', '',
                 'A separate local clone with the current source overlay and an isolated Python environment was used. The external GitHub clone/download setup was not tested. All generated analyses remained in the separate checkout; Ivy’s original validation script and saved results in the main checkout were preserved.', '',
                 f"Frozen dataset SHA-256: `{audit['dataset_sha256']}`.", '',
                 '## Checks', '', '| Check | Passed | Detail |', '| --- | :---: | --- |']
        lines += [f"| {r['check']} | {'Yes' if r['passed'] else 'No'} | {r['detail'].replace('|','/')} |" for r in checks]
        lines += ['', '## Commands actually run', '', '```text']
        lines += [f"{r['command']}  # exit={r['exit_code']}, seconds={r['seconds']}" for r in commands]
        lines += ['```', '', '## Environment and comparison rules', '',
                  'Integer counts, column order, row order and categorical values must match exactly. Numeric CSV results use relative tolerance 1e-6 and absolute tolerance 1e-10. The cleaned CSV is compared after decompression. EDA repeat outputs are compared byte-for-byte. Bootstrap uncertainty describes sampling variation conditional on the fitted models; package/plot rendering differences can prevent binary agreement across environments.', '',
                  'The fresh run is compared with earlier saved results. Only EDA receives an additional immediate repeat; a second full pipeline execution is not claimed. Automated contract checks and synthetic tests support the leakage review; the metadata alone does not prove absence of leakage.', '',
                  'This is a member reproducibility note, not the final group report. Code and execution were assisted by Codex. Course approach: Week 1 reproducible seeds; Week 8 seeded simulation/bootstrap; Week 9 training-only preprocessing and validation/test separation.', '', '```text', *audit.get('software',[]), '```', '',
                  'Detailed audit: `outputs/tables/reproducibility_audit.json`. Per-command logs remain in the local work directory recorded there.', '']
        (PROJECT_DIR/'report/member6_reproducibility_report.md').write_text('\n'.join(lines))
        print('Reproduction result:',audit['complete'],'work directory:',work,flush=True)
    return 0 if audit['complete'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
