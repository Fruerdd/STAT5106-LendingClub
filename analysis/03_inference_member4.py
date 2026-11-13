"""Member 4: statistical inference and feature-engineering entry point.

Required input is the cleaned cohort. This initial scaffold validates the input and derived
features; it deliberately does not report coefficients, p-values, or significance until the
team specifies estimands, controls, standard errors, and multiple-testing treatment.
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

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Inference input not found: {args.input}. Run the cleaning command first."
        )
    frame = pd.read_csv(args.input, compression="infer")
    required = {"default", "issue_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Inference requires cleaned columns: {', '.join(missing)}")
    if {"loan_amnt", "annual_inc"}.issubset(frame.columns):
        denominator = frame["annual_inc"].where(frame["annual_inc"].gt(0))
        frame["loan_to_annual_income"] = frame["loan_amnt"] / denominator
    print(
        f"Loaded {len(frame):,} cleaned rows. Feature checks are ready in {Path(__file__).name}; "
        f"no inference results have been generated or written to {TABLES_DIR}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
