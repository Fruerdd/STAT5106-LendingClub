"""Reproducible chronological row proportions, with outcome-independent month ties."""
import numpy as np
import pandas as pd
import hashlib
import json
from pathlib import Path


def chronological_602020(frame):
    dates = pd.to_datetime(frame['issue_date'], errors='coerce')
    ids = frame['id'].astype('string')
    if dates.isna().any() or ids.isna().any() or ids.duplicated().any():
        raise ValueError('Split requires valid issue dates and unique nonmissing loan IDs')
    # IDs only break month ties; neither labels nor predictors influence assignment.
    ties = pd.util.hash_pandas_object(ids, index=False).to_numpy()
    order = np.lexsort((ids.to_numpy(dtype=str), ties, dates.to_numpy()))
    n = len(frame)
    boundaries = (int(.6*n), int(.8*n))
    parts = tuple(frame.iloc[i].copy() for i in np.split(order, boundaries))
    if any(p.empty or p.default.nunique() != 2 for p in parts):
        raise ValueError('Each partition must contain both binary outcome classes')
    if not frame.default.isin([0, 1]).all():
        raise ValueError('Outcomes must be binary')
    return parts


def frozen_602020(frame, contract_path):
    """Create once, then require exactly the same ordered IDs, dates and outcomes."""
    parts = chronological_602020(frame)
    contract = {'version': 1, 'method': 'chronological_602020_ID_hash_month_ties', 'partitions': {}}
    for name, part in zip(('train', 'validation', 'test'), parts):
        records = part[['id', 'issue_date', 'default']].astype('string').to_csv(index=False, lineterminator='\n')
        contract['partitions'][name] = {'rows': len(part), 'identity_date_outcome_sha256': hashlib.sha256(records.encode()).hexdigest()}
    path = Path(contract_path)
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError('Frozen 60/20/20 split mismatch: cohort, assignment, dates or outcomes changed. Do not overwrite the split contract.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement of an established lock.
        with path.open('x') as handle:
            handle.write(json.dumps(contract, indent=2)+'\n')
    return parts
