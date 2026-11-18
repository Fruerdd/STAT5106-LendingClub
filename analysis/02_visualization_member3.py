"""Member 3: exploratory plots using only the resolved, cleaned cohort.

Input: data/cleaned/clean_v1/lending_club_clean.csv.gz (override with --input).
Outputs: real default-rate tables and time/grade figures under outputs/. No synthetic plots
are produced. Run Member 2's cleaning pipeline before this script.

Updated on Nov 18th 2026

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, FIGURES_DIR, TABLES_DIR

DEFAULT_INPUT = (
    CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {filename}")


def summarize_default(frame, group_col, filename):
    """Calculate observed default rates and sample counts."""
    result = (
        frame.groupby(group_col, observed=True)["default"]
        .agg(default_rate="mean", loan_count="size",
             default_count="sum")
        .reset_index()
    )
    result.to_csv(TABLES_DIR / filename, index=False)
    return result


def plot_loan_amount(df):
    """Figure 1: Loan amount distribution."""
    if "loan_amnt" not in df:
        return

    values = pd.to_numeric(df["loan_amnt"], errors="coerce").dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(values, bins=40, ax=ax, color="steelblue")

    ax.set(
        xlabel="Loan amount (USD)",
        ylabel="Number of loans",
        title="Distribution of loan amounts"
    )
    save_figure(fig, "loan_amount_distribution.png")


def plot_grade(df):
    """Figure 2: Default rate by loan grade."""
    if "grade" not in df:
        return

    data = df.dropna(subset=["grade"])
    if data.empty:
        return

    result = summarize_default(
        data, "grade", "default_rate_by_grade.csv"
    )

    order = [
        grade for grade in list("ABCDEFG")
        if grade in result["grade"].values
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=result,
        x="grade",
        y="default_rate",
        order=order,
        color="steelblue",
        errorbar=None,
        ax=ax
    )

    ax.set(
        xlabel="Loan grade",
        ylabel="Observed default rate",
        title="Default rate by loan grade"
    )
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )
    save_figure(fig, "default_rate_by_grade.png")


def plot_binned_default(df, column, bins, filename, xlabel):
    """Default rate across fixed numerical intervals."""
    if column not in df:
        return

    values = pd.to_numeric(df[column], errors="coerce")
    subset = df[["default"]].copy()
    subset["group"] = pd.cut(values, bins=bins)

    subset = subset.dropna(subset=["group", "default"])
    if subset.empty:
        return

    result = summarize_default(
        subset,
        "group",
        f"default_rate_by_{column}_bins.csv"
    )

    # Convert interval labels into strings for charting.
    result["label"] = result["group"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=result,
        x="label",
        y="default_rate",
        color="steelblue",
        errorbar=None,
        ax=ax
    )

    ax.set(
        xlabel=xlabel,
        ylabel="Observed default rate",
        title=f"Default rate by {xlabel.lower()}"
    )
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )

    save_figure(fig, filename)


def plot_interest_rate(df):
    """Figure 3: Default rate by interest-rate interval."""
    plot_binned_default(
        df,
        column="int_rate",
        bins=[0, 5, 10, 15, 20, 25, 30, float("inf")],
        filename="default_rate_by_interest_rate.png",
        xlabel="Interest rate (%)"
    )


def plot_income(df):
    """Figure 4: Default rate by annual income."""
    plot_binned_default(
        df,
        column="annual_inc",
        bins=[
            0, 25000, 50000, 75000, 100000,
            150000, 200000, float("inf")
        ],
        filename="default_rate_by_annual_income.png",
        xlabel="Annual income (USD)"
    )


def plot_fico(df):
    """Figure 5: Default rate by origination FICO score."""
    plot_binned_default(
        df,
        column="fico_score",
        bins=[300, 580, 620, 660, 700, 740, 780, 850],
        filename="default_rate_by_fico.png",
        xlabel="FICO score"
    )


def plot_purpose(df):
    """Figure 6: Default rate by loan purpose."""
    if "purpose" not in df:
        return

    data = df.dropna(subset=["purpose"])
    if data.empty:
        return

    result = summarize_default(
        data, "purpose", "default_rate_by_purpose.csv"
    )

    result = result.sort_values("default_rate")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=result,
        x="default_rate",
        y="purpose",
        color="steelblue",
        errorbar=None,
        ax=ax
    )

    ax.set(
        xlabel="Observed default rate",
        ylabel="Loan purpose",
        title="Default rate by loan purpose"
    )
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )

    save_figure(fig, "default_rate_by_purpose.png")


def plot_time(df):
    """Figure 7: Observed default rate by issue quarter."""
    if "issue_date" not in df:
        return

    data = df[["issue_date", "default"]].copy()
    data["issue_date"] = pd.to_datetime(
        data["issue_date"], errors="coerce"
    )
    data = data.dropna(subset=["issue_date"])

    if data.empty:
        return

    data["issue_quarter"] = (
        data["issue_date"].dt.to_period("Q").astype(str)
    )

    result = summarize_default(
        data,
        "issue_quarter",
        "default_rate_by_issue_quarter.csv"
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        range(len(result)),
        result["default_rate"],
        marker="o",
        markersize=3,
        linewidth=1.5
    )

    step = max(1, len(result) // 12)
    positions = list(range(0, len(result), step))

    ax.set_xticks(positions)
    ax.set_xticklabels(
        result["issue_quarter"].iloc[positions],
        rotation=45,
        ha="right"
    )

    ax.set(
        xlabel="Loan issue quarter",
        ylabel="Observed default rate",
        title="Default rate by loan origination quarter"
    )
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )

    save_figure(fig, "default_rate_over_time.png")


def plot_correlation(df):
    """Figure 8: Correlation among selected numeric fields."""
    candidates = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "fico_score",
        "revol_util",
        "credit_history_months",
        "term_months"
    ]

    columns = [c for c in candidates if c in df.columns]
    if len(columns) < 2:
        return

    numeric = df[columns].apply(
        pd.to_numeric, errors="coerce"
    )

    corr = numeric.corr()
    corr.to_csv(TABLES_DIR / "eda_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )

    ax.set_title("Correlations among loan characteristics")
    save_figure(fig, "correlation_heatmap.png")


def plot_missingness(df):
    """Figure 9: Missingness of the most incomplete columns."""
    missing_path = TABLES_DIR / "missingness.csv"

    if missing_path.is_file():
        missing = pd.read_csv(missing_path)
    else:
        # Fallback for independent EDA execution.
        missing = pd.DataFrame({
            "column": df.columns,
            "missing_proportion": df.isna().mean().values
        })

    missing = (
        missing.sort_values("missing_proportion", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=missing,
        x="missing_proportion",
        y="column",
        color="steelblue",
        ax=ax
    )

    ax.set(
        xlabel="Proportion missing",
        ylabel="Variable",
        title="Variables with highest missingness"
    )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1)
    )

    save_figure(fig, "missingness_top15.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {args.input}. "
            "Run the cleaning pipeline first."
        )

    # Read selected origination-time fields only.
    requested = [
        "default", "issue_date", "loan_amnt", "grade",
        "int_rate", "annual_inc", "fico_score",
        "purpose", "dti", "revol_util",
        "credit_history_months", "term_months"
    ]

    header = pd.read_csv(args.input, nrows=0)
    available = [c for c in requested if c in header.columns]

    required = {"default", "issue_date"}
    missing_required = required - set(available)
    if missing_required:
        raise ValueError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    df = pd.read_csv(
        args.input,
        usecols=available,
        compression="infer",
        low_memory=False
    )

    df["default"] = pd.to_numeric(
        df["default"], errors="coerce"
    )
    if df["default"].isna().any() or not df["default"].isin([0, 1]).all():
        raise ValueError("Default outcome must contain only 0 and 1.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Observations: {len(df):,}")
    print(f"Observed default rate: {df['default'].mean():.2%}")
    print(f"Available variables: {available}")

    # Descriptive summary for selected numerical variables.
    numeric_cols = [
        c for c in [
            "loan_amnt", "int_rate", "annual_inc",
            "fico_score", "dti", "revol_util",
            "credit_history_months", "term_months"
        ] if c in df.columns
    ]

    numeric = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric.describe().T.to_csv(
        TABLES_DIR / "eda_numeric_summary.csv"
    )

    # Generate all figures and summary tables.
    plot_loan_amount(df)
    plot_grade(df)
    plot_interest_rate(df)
    plot_income(df)
    plot_fico(df)
    plot_purpose(df)
    plot_time(df)
    plot_correlation(df)
    plot_missingness(df)

    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Tables saved in: {TABLES_DIR}")
    print("Member 3 EDA completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
