"""Member 6: end-to-end reproducibility and consistency-check entry point.

This check requires the frozen raw file and an existing cleaned output/report pair. It verifies
cohort counts, outcomes, paths, and report consistency; it does not fabricate downstream outputs.
Use a fresh clone and repeat the documented commands for the final full-team reproduction.
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

from src.paths import CLEANED_DATA_DIR, PROJECT_DIR, RAW_DATA_PATH

DEFAULT_CLEANED = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
DEFAULT_REPORT = CLEANED_DATA_DIR / "clean_v1" / "cleaning_report.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    for label, path in (("Raw dataset", args.raw), ("Cleaned data", args.cleaned), ("Cleaning report", args.report)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}. Follow the README execution order.")

    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    required_report_fields = {
        "raw_row_count",
        "final_row_count",
        "status_exclusions",
        "duplicates",
        "invalid_values",
        "missingness",
        "outcome_counts",
        "default_rate",
        "issue_date_range",
    }
    missing_fields = sorted(required_report_fields.difference(report))
    if missing_fields:
        raise ValueError(f"Cleaning report is missing fields: {', '.join(missing_fields)}")

    cleaned = pd.read_csv(args.cleaned, usecols=["default"], compression="infer")
    if len(cleaned) != report["final_row_count"]:
        raise ValueError("Cleaned row count does not match cleaning_report.json")
    observed_rate = float(cleaned["default"].mean())
    if abs(observed_rate - report["default_rate"]) > 1e-12:
        raise ValueError("Observed default rate does not match cleaning_report.json")
    if not Path(args.cleaned).resolve().is_relative_to(PROJECT_DIR):
        raise ValueError("The reproducibility check expects cleaned data inside the project")
    print("Reproducibility consistency checks passed for the existing real cleaned output.")
    print("No downstream analysis artefacts were inferred or created by this check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
