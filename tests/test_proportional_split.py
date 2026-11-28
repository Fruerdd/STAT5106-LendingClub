import numpy as np
import pandas as pd
import pytest
from src.proportional_split import chronological_602020, frozen_602020


def test_split_counts_coverage_chronology_and_order_independence():
    frame = pd.DataFrame({'id': np.arange(103), 'default': np.arange(103)%2,
                          'issue_date': np.repeat(pd.date_range('2010-01-01', periods=11, freq='MS'), 10)[:103]})
    parts = chronological_602020(frame)
    assert [len(p) for p in parts] == [61, 21, 21]
    assert len(set().union(*(set(p.id) for p in parts))) == len(frame)
    assert parts[0].issue_date.max() <= parts[1].issue_date.min()
    assert parts[1].issue_date.max() <= parts[2].issue_date.min()
    again = chronological_602020(frame.sample(frac=1, random_state=5))
    assert all(a.id.tolist() == b.id.tolist() for a, b in zip(parts, again))
    swapped = chronological_602020(frame.assign(default=1-frame.default))
    assert all(a.id.tolist() == b.id.tolist() for a, b in zip(parts, swapped))


def test_split_rejects_invalid_dates_and_duplicate_ids():
    frame = pd.DataFrame({'id': range(100), 'default': np.arange(100)%2, 'issue_date': '2015-01-01'})
    for column, value in [('issue_date', 'invalid'), ('id', 1)]:
        bad = frame.copy(); bad.loc[0, column] = value
        with pytest.raises(ValueError): chronological_602020(bad)


def test_frozen_split_refuses_changed_cohort_or_labels(tmp_path):
    frame = pd.DataFrame({'id': range(100), 'default': np.arange(100)%2,
                          'issue_date': np.repeat(pd.date_range('2010-01-01', periods=10, freq='MS'), 10)})
    path = tmp_path/'contract.json'
    frozen_602020(frame, path)
    original = path.read_bytes()
    frozen_602020(frame.sample(frac=1, random_state=3), path)
    for changed in [frame.iloc[1:], frame.assign(default=1-frame.default), frame.assign(issue_date='2012-01-01')]:
        with pytest.raises(ValueError, match='Frozen'):
            frozen_602020(changed, path)
        assert path.read_bytes() == original
