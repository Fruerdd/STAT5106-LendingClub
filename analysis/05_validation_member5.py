"""Member 5: out-of-time model validation and limitations entry point.

Inputs: the cleaned cohort and a real out-of-time predictions table containing loan id,
issue_date, observed default, and model-predicted probability. The scaffold validates those
inputs only. ROC-AUC, PR-AUC, calibration, thresholds, bootstrap intervals, subgroup and
robustness analyses must be computed from those predictions; no placeholder metrics are made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, TABLES_DIR

DEFAULT_CLEANED = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
DEFAULT_PREDICTIONS = TABLES_DIR / "out_of_time_predictions.csv"
REQUIRED_PREDICTION_COLUMNS = {"id", "issue_date", "default", "predicted_probability"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args(argv)
    for label, path in (("Cleaned cohort", args.cleaned), ("Predictions", args.predictions)):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} file not found: {path}. Complete cleaning and out-of-time modelling first."
            )
    predictions = pd.read_csv(args.predictions)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {', '.join(missing)}")
    if predictions["predicted_probability"].isna().any() or not predictions[
        "predicted_probability"
    ].between(0, 1).all():
        raise ValueError("predicted_probability must contain finite probabilities between 0 and 1")
    print(
        f"Validated {len(predictions):,} real prediction rows. Validation metrics, figures, "
        "bootstrap intervals, subgroup checks, robustness analyses, and limitations still "
        "need to be computed; no validation results have been generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
