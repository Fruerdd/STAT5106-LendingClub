"""Pavel: temporal splitting, train-only preprocessing, and logistic-model entry point.

Required input is the cleaned cohort. The reusable split and pipeline functions are provided;
this scaffold checks their inputs but does not fit or report a model until the modelling plan
and comparison candidates are finalized. Imputation is always inside the training pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, TABLES_DIR

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
RANDOM_STATE = 42


def temporal_train_test_split(
    frame: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole issue dates so no date appears in both train and test sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    dates = pd.to_datetime(frame["issue_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Temporal splitting requires a valid issue_date for every row")
    unique_dates = dates.sort_values().unique()
    split_position = int(len(unique_dates) * (1 - test_size))
    if len(unique_dates) < 2 or not 0 < split_position < len(unique_dates):
        raise ValueError("Not enough distinct issue dates for the requested temporal split")
    cutoff = unique_dates[split_position]
    train = frame.loc[dates < cutoff].copy()
    test = frame.loc[dates >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set")
    return train, test


def build_logistic_pipeline(
    numeric_columns: list[str], categorical_columns: list[str], class_weight: str | None = None
) -> Pipeline:
    """Build preprocessing fitted only on training data, followed by logistic regression."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Modelling input not found: {args.input}. Run the cleaning command first."
        )
    header = pd.read_csv(args.input, nrows=0, compression="infer")
    required = {"default", "issue_date"}
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Modelling requires cleaned columns: {', '.join(missing)}")
    print(
        "Modelling scaffold ready: use temporal_train_test_split and build_logistic_pipeline; "
        "compare prespecified candidates on validation data and interpret fitted coefficients. "
        f"No model has been fitted and no results have been written to {TABLES_DIR}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
