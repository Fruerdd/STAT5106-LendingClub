"""Validation-only threshold trade-offs on the frozen 60/20/20 split. Never opens test predictions."""
import sys
import hashlib
import json
from pathlib import Path
import pandas as pd
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT_DIR, CLEANED_DATA_DIR
from src.operating_policy import validation_options


def main():
    rows = []
    handoff = CLEANED_DATA_DIR/'split_602020_handoff'
    contract = json.loads((PROJECT_DIR/'outputs/experiments/split_602020/frozen_split_contract.json').read_text())
    for candidate in ['expanded_grade_31', 'xgb_depth5_double']:
        frame = pd.read_csv(handoff/f'{candidate}_validation.csv.gz')
        if len(frame)!=contract['partitions']['validation']['rows'] or frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError('Unexpected validation cohort')
        records = frame[['id','issue_date','default']].astype('string').to_csv(index=False, lineterminator='\n')
        if hashlib.sha256(records.encode()).hexdigest()!=contract['partitions']['validation']['identity_date_outcome_sha256']:
            raise ValueError('Validation handoff differs from frozen split contract')
        rows.extend({'candidate': candidate, 'cohort': 'validation_only', **r}
                    for r in validation_options(frame.default, frame.predicted_probability))
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT_DIR/'outputs/experiments/split_602020/validation_operating_options.csv', index=False)
    print(result[['candidate','rule','constraint_met','threshold','precision','recall','f1','fp','fn']].to_string(index=False))
    print('Validation estimates only. No new policy was selected or assessed on test data; minimum 1000 flagged validation loans for constrained rules.')


if __name__ == '__main__':
    main()
