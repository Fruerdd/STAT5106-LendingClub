"""Build a leakage-safe, resolved-outcome Lending Club modelling cohort.

Run from the repository root:
    python src/lending_club_data_cleaning.py --input data/raw/accepted_2007_to_2018Q4.csv.gz --output-dir data/cleaned/clean_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, RAW_DATA_PATH

MINIMUM_INPUT_COLUMNS = frozenset({"loan_status", "issue_d"})
ORIGINATION_COLUMNS = (
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
)
LEAKAGE_FIELDS = frozenset(
    {
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "last_fico_range_low",
        "last_fico_range_high",
        "last_credit_pull_d",
        "out_prncp",
        "out_prncp_inv",
        "next_pymnt_d",
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
        "hardship_flag",
        "hardship_status",
        "hardship_loan_status",
        "disbursement_method",
    }
)
IMPORTANT_FIELDS = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length_years",
    "revol_util",
    "credit_history_months",
    "fico_score",
)
NUMERIC_BOUNDS = {
    "loan_amnt": {"strictly_positive": True},
    "installment": {"strictly_positive": True},
    "annual_inc": {"minimum": 0},
    "dti": {"minimum": 0},
    "delinq_2yrs": {"minimum": 0},
    "inq_last_6mths": {"minimum": 0},
    "mths_since_last_delinq": {"minimum": 0},
    "mths_since_last_record": {"minimum": 0},
    "open_acc": {"minimum": 0},
    "pub_rec": {"minimum": 0},
    "revol_bal": {"minimum": 0},
    "total_acc": {"minimum": 0},
    "fico_range_low": {"minimum": 300, "maximum": 850},
    "fico_range_high": {"minimum": 300, "maximum": 850},
}


def construct_default(loan_status: pd.Series) -> pd.Series:
    """Map resolved outcomes to 0/1 and unresolved or unknown outcomes to missing."""
    normalized = loan_status.astype("string").str.strip().str.lower()
    return normalized.map(
        {"fully paid": 0, "charged off": 1, "default": 1}
    ).astype("Int64")


def parse_percentage(series: pd.Series) -> pd.Series:
    """Convert percentage strings to numeric percentage points without rescaling."""
    cleaned = series.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def parse_term(series: pd.Series) -> pd.Series:
    """Extract the number of months from values such as '36 months'."""
    months = series.astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(months, errors="coerce")


def parse_employment_length(series: pd.Series) -> pd.Series:
    """Convert employment length to years, mapping '< 1 year' to 0 and '10+' to 10."""
    normalized = series.astype("string").str.strip().str.lower()
    years = pd.to_numeric(
        normalized.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce"
    )
    return years.mask(normalized.str.contains(r"^<\s*1", na=False), 0)


def _present_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def _coerce_numeric(
    frame: pd.DataFrame,
    column: str,
    invalid_values: dict[str, int],
    *,
    parser: Callable[[pd.Series], pd.Series] | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> None:
    source = frame[column]
    numeric = parser(source) if parser is not None else pd.to_numeric(source, errors="coerce")
    numeric = numeric.astype("float64")
    invalid = _present_mask(source) & numeric.isna()

    impossible = pd.Series(False, index=frame.index)
    if minimum is not None:
        impossible |= numeric.lt(minimum).fillna(False)
    if maximum is not None:
        impossible |= numeric.gt(maximum).fillna(False)
    if strictly_positive:
        impossible |= numeric.le(0).fillna(False)

    invalid_values[column] = int((invalid | impossible).sum())
    frame[column] = numeric.mask(impossible)


def _parse_dates(frame: pd.DataFrame, invalid_values: dict[str, int]) -> None:
    issue_source = frame["issue_d"]
    issue_date = pd.to_datetime(issue_source, format="%b-%Y", errors="coerce")
    invalid_values["issue_d"] = int((_present_mask(issue_source) & issue_date.isna()).sum())
    frame["issue_date"] = issue_date

    if "earliest_cr_line" not in frame:
        return

    credit_source = frame["earliest_cr_line"]
    normalized = credit_source.astype("string").str.strip()
    two_digit_year = normalized.str.contains(r"^[A-Za-z]{3}-\d{2}$", na=False)
    credit_date = pd.to_datetime(credit_source, format="%b-%Y", errors="coerce")
    two_digit_dates = pd.to_datetime(
        credit_source.where(two_digit_year), format="%b-%y", errors="coerce"
    )
    credit_date = credit_date.fillna(two_digit_dates)
    invalid_credit = _present_mask(credit_source) & credit_date.isna()

    correct_century = (
        two_digit_year
        & credit_date.notna()
        & issue_date.notna()
        & credit_date.gt(issue_date)
    )
    if correct_century.any():
        credit_date.loc[correct_century] = credit_date.loc[correct_century].map(
            lambda value: value - pd.DateOffset(years=100)
        )

    impossible_credit = credit_date.notna() & issue_date.notna() & credit_date.gt(issue_date)
    invalid_values["earliest_cr_line"] = int((invalid_credit | impossible_credit).sum())
    credit_date = credit_date.mask(impossible_credit)
    frame["earliest_cr_line_date"] = credit_date

    months = (issue_date.dt.year - credit_date.dt.year) * 12 + (
        issue_date.dt.month - credit_date.dt.month
    )
    frame["credit_history_months"] = months.astype("float64").mask(months.lt(0))


def _status_exclusion_counts(status: pd.Series) -> dict[str, int]:
    display_status = status.astype("string").fillna("<missing>").str.strip()
    display_status = display_status.mask(display_status.eq(""), "<blank>")
    counts = display_status.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean an allowlisted raw dataframe and return the cohort plus its audit report."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame")
    if raw.empty:
        raise ValueError("The input dataset contains no rows")

    missing_required = sorted(MINIMUM_INPUT_COLUMNS.difference(raw.columns))
    if missing_required:
        raise ValueError(f"Missing required input columns: {', '.join(missing_required)}")

    retained_columns = [column for column in ORIGINATION_COLUMNS if column in raw.columns]
    working = raw.loc[:, retained_columns].copy()
    raw_row_count = len(working)

    exact_duplicates = working.duplicated(keep="first")
    working = working.loc[~exact_duplicates].copy()

    duplicate_ids = pd.Series(False, index=working.index)
    if "id" in working:
        usable_id = _present_mask(working["id"])
        duplicate_ids = usable_id & working.duplicated(subset=["id"], keep="first")
        working = working.loc[~duplicate_ids].copy()

    outcome = construct_default(working["loan_status"])
    unresolved = outcome.isna()
    status_exclusions = _status_exclusion_counts(working.loc[unresolved, "loan_status"])
    cleaned = working.loc[~unresolved].copy()
    cleaned["default"] = outcome.loc[~unresolved].astype("int8")

    invalid_values: dict[str, int] = {}
    for column, bounds in NUMERIC_BOUNDS.items():
        if column in cleaned:
            _coerce_numeric(cleaned, column, invalid_values, **bounds)

    percentage_parsers = {"int_rate": parse_percentage, "revol_util": parse_percentage}
    for column, parser in percentage_parsers.items():
        if column in cleaned:
            _coerce_numeric(cleaned, column, invalid_values, parser=parser, minimum=0)

    if "term" in cleaned:
        _coerce_numeric(
            cleaned,
            "term",
            invalid_values,
            parser=parse_term,
            strictly_positive=True,
        )
        cleaned["term_months"] = cleaned.pop("term")
    if "emp_length" in cleaned:
        _coerce_numeric(
            cleaned,
            "emp_length",
            invalid_values,
            parser=parse_employment_length,
            minimum=0,
        )
        cleaned["emp_length_years"] = cleaned.pop("emp_length")

    _parse_dates(cleaned, invalid_values)
    if {"fico_range_low", "fico_range_high"}.issubset(cleaned.columns):
        cleaned["fico_score"] = cleaned[["fico_range_low", "fico_range_high"]].mean(
            axis=1, skipna=False
        )

    for column in IMPORTANT_FIELDS:
        if column in cleaned:
            cleaned[f"{column}_was_missing"] = cleaned[column].isna().astype("int8")

    expected_final_count = (
        raw_row_count - int(exact_duplicates.sum()) - int(duplicate_ids.sum()) - int(unresolved.sum())
    )
    assert len(cleaned) == expected_final_count, "Row-count reconciliation failed"
    assert cleaned["default"].notna().all(), "Unresolved outcomes remain in the cohort"
    assert set(cleaned["default"].unique()).issubset({0, 1}), "Invalid outcome encoding"
    assert not cleaned.columns.intersection(LEAKAGE_FIELDS).any(), "Leakage field retained"
    assert not cleaned.duplicated().any(), "Exact duplicate rows remain after cleaning"
    if "id" in cleaned:
        usable_id = _present_mask(cleaned["id"])
        assert not cleaned.loc[usable_id, "id"].duplicated().any(), "Duplicate IDs remain"
    if "credit_history_months" in cleaned:
        assert not cleaned["credit_history_months"].lt(0).any(), "Negative credit history"

    outcome_counts = cleaned["default"].value_counts()
    issue_dates = cleaned["issue_date"].dropna()
    missingness = {
        column: {
            "count": int(cleaned[column].isna().sum()),
            "proportion": (
                float(cleaned[column].isna().mean()) if len(cleaned) else None
            ),
        }
        for column in cleaned.columns
    }
    report = {
        "raw_row_count": raw_row_count,
        "final_row_count": len(cleaned),
        "status_exclusions": {
            "total": int(unresolved.sum()),
            "by_status": status_exclusions,
        },
        "duplicates": {
            "exact_rows_removed": int(exact_duplicates.sum()),
            "duplicate_id_rows_removed": int(duplicate_ids.sum()),
        },
        "invalid_values": invalid_values,
        "missingness": missingness,
        "outcome_counts": {
            "fully_paid": int(outcome_counts.get(0, 0)),
            "charged_off_or_default": int(outcome_counts.get(1, 0)),
        },
        "default_rate": float(cleaned["default"].mean()) if len(cleaned) else None,
        "issue_date_range": {
            "minimum": issue_dates.min().strftime("%Y-%m-%d") if not issue_dates.empty else None,
            "maximum": issue_dates.max().strftime("%Y-%m-%d") if not issue_dates.empty else None,
        },
    }
    return cleaned, report


def read_raw_data(input_path: Path) -> pd.DataFrame:
    """Read only allowlisted columns, inferring gzip compression from the filename."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}. Place the frozen dataset at "
            "data/raw/accepted_2007_to_2018Q4.csv.gz or pass --input."
        )
    try:
        header = pd.read_csv(input_path, nrows=0, compression="infer")
    except Exception as exc:
        raise ValueError(f"Could not read the input header from {input_path}: {exc}") from exc

    missing_required = sorted(MINIMUM_INPUT_COLUMNS.difference(header.columns))
    if missing_required:
        raise ValueError(f"Missing required input columns: {', '.join(missing_required)}")
    usecols = [column for column in ORIGINATION_COLUMNS if column in header.columns]
    return pd.read_csv(
        input_path,
        usecols=usecols,
        compression="infer",
        low_memory=False,
    )


def run_cleaning(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read, clean, and save the cohort and report without modifying the input."""
    output_data_path = output_dir / "lending_club_clean.csv.gz"
    output_report_path = output_dir / "cleaning_report.json"
    if input_path.resolve() in {output_data_path.resolve(), output_report_path.resolve()}:
        raise ValueError("The output paths must not overwrite the input dataset")

    raw = read_raw_data(input_path)
    cleaned, report = clean_dataframe(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(
        output_data_path,
        index=False,
        compression="infer",
        date_format="%Y-%m-%d",
    )
    with output_report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return cleaned, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Raw CSV or CSV.GZ file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLEANED_DATA_DIR,
        help="Directory for lending_club_clean.csv.gz and cleaning_report.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cleaned, report = run_cleaning(args.input, args.output_dir)
    default_rate = report["default_rate"]
    rate_display = f"{default_rate:.4%}" if default_rate is not None else "n/a"
    print(
        f"Cleaned {report['raw_row_count']:,} raw rows to {report['final_row_count']:,} "
        f"resolved-outcome rows; default rate={rate_display}"
    )
    print(f"Wrote {args.output_dir / 'lending_club_clean.csv.gz'}")
    print(f"Wrote {args.output_dir / 'cleaning_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
