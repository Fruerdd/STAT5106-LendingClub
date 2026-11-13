"""Member 2: validate cleaning, write a data dictionary, and audit missingness/leakage.

Input: data/cleaned/clean_v1/lending_club_clean.csv.gz (override with --input).
Outputs: data_dictionary.csv, missingness.csv, and cleaning_validation.json in outputs/tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lending_club_data_cleaning import LEAKAGE_FIELDS
from src.paths import CLEANED_DATA_DIR, TABLES_DIR

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
DERIVED_FIELD_DESCRIPTIONS = {
    "default": "Resolved outcome: Fully Paid=0; Charged Off or Default=1.",
    "loan_status": "Original Lending Club loan status retained for auditing.",
    "term_months": "Original loan term parsed into months.",
    "int_rate": "Origination interest rate in numeric percentage points.",
    "revol_util": "Revolving-line utilization in numeric percentage points.",
    "emp_length_years": "Employment length in years; <1 year=0 and 10+ years=10.",
    "issue_date": "Loan issue month parsed from issue_d.",
    "earliest_cr_line_date": "Earliest credit-line month with two-digit-year century correction.",
    "credit_history_months": "Whole months between earliest credit line and issue date.",
    "fico_score": "Midpoint of origination fico_range_low and fico_range_high.",
}


def require_input(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned data required at {path}. Run the documented cleaning command first "
            "or pass --input."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    require_input(args.input)
    frame = pd.read_csv(args.input, compression="infer")
    if "default" not in frame or "loan_status" not in frame:
        raise ValueError("Cleaned data must contain both default and original loan_status")
    if not set(frame["default"].dropna().unique()).issubset({0, 1}):
        raise ValueError("default must contain only 0 and 1")

    leakage = sorted(set(frame.columns).intersection(LEAKAGE_FIELDS))
    if leakage:
        raise ValueError(f"Post-outcome leakage fields found: {', '.join(leakage)}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    dictionary = pd.DataFrame(
        {
            "column": frame.columns,
            "pandas_dtype": [str(frame[column].dtype) for column in frame.columns],
            "description": [
                DERIVED_FIELD_DESCRIPTIONS.get(
                    column,
                    "Missingness indicator derived before modelling."
                    if column.endswith("_was_missing")
                    else "Application/origination-time field from the frozen source data.",
                )
                for column in frame.columns
            ],
        }
    )
    dictionary.to_csv(TABLES_DIR / "data_dictionary.csv", index=False)
    missingness = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
            "missing_proportion": [float(frame[column].isna().mean()) for column in frame.columns],
        }
    )
    missingness.to_csv(TABLES_DIR / "missingness.csv", index=False)
    validation = {
        "row_count": len(frame),
        "outcome_counts": {
            str(key): int(value) for key, value in frame["default"].value_counts().items()
        },
        "leakage_fields_present": leakage,
        "duplicate_nonmissing_ids": (
            int(frame.loc[frame["id"].notna(), "id"].duplicated().sum()) if "id" in frame else None
        ),
    }
    with (TABLES_DIR / "cleaning_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Cleaning audits written to {TABLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
