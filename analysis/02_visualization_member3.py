"""Member 3: exploratory plots using only the resolved, cleaned cohort.

Input: data/cleaned/clean_v1/lending_club_clean.csv.gz (override with --input).
Outputs: real default-rate tables and time/grade figures under outputs/. No synthetic plots
are produced. Run Member 2's cleaning pipeline before this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, FIGURES_DIR, TABLES_DIR

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Visualization input not found: {args.input}. Run the cleaning command first."
        )

    header = pd.read_csv(args.input, nrows=0, compression="infer")
    required = {"default", "issue_date"}
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"Visualization requires columns: {', '.join(missing)}")
    optional = [column for column in ("grade",) if column in header.columns]
    frame = pd.read_csv(
        args.input,
        usecols=["default", "issue_date", *optional],
        compression="infer",
    )
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="coerce")
    frame = frame.dropna(subset=["issue_date", "default"])
    if frame.empty:
        raise ValueError("No rows have both a valid issue_date and default outcome")
    frame["issue_quarter"] = frame["issue_date"].dt.to_period("Q").astype(str)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    over_time = frame.groupby("issue_quarter")["default"].agg(["mean", "size"]).reset_index()
    over_time.columns = ["issue_quarter", "default_rate", "loan_count"]
    over_time.to_csv(TABLES_DIR / "default_rate_by_issue_quarter.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(over_time["issue_quarter"], over_time["default_rate"])
    ax.set(title="Observed default rate by issue quarter", xlabel="Issue quarter", ylabel="Default rate")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "default_rate_over_time.png", dpi=160)
    plt.close(fig)

    if "grade" in frame:
        by_grade = frame.groupby("grade")["default"].agg(["mean", "size"]).reset_index()
        by_grade.columns = ["grade", "default_rate", "loan_count"]
        by_grade.to_csv(TABLES_DIR / "default_rate_by_grade.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=by_grade, x="grade", y="default_rate", ax=ax)
        ax.set(title="Observed default rate by grade", xlabel="Grade", ylabel="Default rate")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "default_rate_by_grade.png", dpi=160)
        plt.close(fig)

    print(f"Exploratory tables written to {TABLES_DIR}")
    print(f"Exploratory figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
