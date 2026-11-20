"""Member 1: independently audit the cleaned cohort and write a data dictionary.

Run from the repository root after cleaning. The audit uses the frozen raw data, cleaned cohort,
and cleaning report; it writes cleaning_validation.json, missingness.csv, and data_dictionary.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lending_club_data_cleaning import (
    IMPORTANT_FIELDS,
    LEAKAGE_FIELDS,
    NUMERIC_BOUNDS,
    ORIGINATION_COLUMNS,
    construct_default,
    read_raw_data,
)
from src.paths import CLEANED_DATA_DIR, RAW_DATA_PATH, TABLES_DIR

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
DEFAULT_REPORT = DEFAULT_INPUT.with_name("cleaning_report.json")

SOURCE_INFO = {
    "id": ("Unique Lending Club loan identifier.", "identifier"),
    "loan_amnt": ("Listed loan amount applied for by the borrower.", "USD"),
    "int_rate": ("Origination interest rate.", "percentage points"),
    "installment": ("Monthly payment owed after origination.", "USD per month"),
    "grade": ("Lending Club assigned loan grade.", "category"),
    "sub_grade": ("Lending Club assigned loan subgrade.", "category"),
    "home_ownership": ("Borrower-reported home-ownership status.", "category"),
    "annual_inc": ("Self-reported annual income.", "USD per year"),
    "verification_status": ("Income verification status.", "category"),
    "issue_d": ("Origination month retained for temporal auditing and splitting.", "month"),
    "loan_status": ("Original loan status retained for outcome auditing.", "category"),
    "purpose": ("Borrower-provided loan purpose.", "category"),
    "addr_state": ("Borrower-provided state.", "category"),
    "dti": ("Debt-to-income ratio at application.", "percentage points"),
    "delinq_2yrs": ("Delinquencies in the previous two years.", "events"),
    "earliest_cr_line": ("Month of the borrower's earliest reported credit line.", "month"),
    "fico_range_low": ("Lower origination FICO range boundary.", "FICO points"),
    "fico_range_high": ("Upper origination FICO range boundary.", "FICO points"),
    "inq_last_6mths": ("Credit inquiries in the previous six months.", "inquiries"),
    "mths_since_last_delinq": ("Months since the last delinquency.", "months"),
    "mths_since_last_record": ("Months since the last public record.", "months"),
    "open_acc": ("Open credit lines in the borrower's credit file.", "accounts"),
    "pub_rec": ("Derogatory public records.", "records"),
    "revol_bal": ("Total revolving credit balance.", "USD"),
    "revol_util": ("Revolving-line utilization rate.", "percentage points"),
    "total_acc": ("Total credit lines in the borrower's credit file.", "accounts"),
    "application_type": ("Individual or joint application.", "category"),
}
DERIVED_INFO = {
    "default": ("Resolved outcome: Fully Paid=0; Charged Off/Default=1.", "binary", "Derived from loan_status"),
    "term_months": ("Original loan term parsed into months.", "months", "Derived from term"),
    "emp_length_years": ("Employment length; <1 year=0 and 10+ years=10.", "years", "Derived from emp_length"),
    "issue_date": ("Loan issue month.", "date", "Parsed from issue_d"),
    "earliest_cr_line_date": ("Earliest credit-line month with century correction.", "date", "Parsed from earliest_cr_line and issue_date"),
    "credit_history_months": ("Whole months between earliest credit line and issue date.", "months", "Derived from issue_date and earliest_cr_line_date"),
    "fico_score": ("Midpoint of origination FICO bounds.", "FICO points", "Mean of fico_range_low and fico_range_high"),
}
APPROVED_DIRECT = {
    "grade", "home_ownership", "verification_status", "purpose", "application_type",
    "term_months", "emp_length_years", "revol_util", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "int_rate", "dti", "fico_score", "credit_history_months",
}
SOURCE_ONLY = {"loan_amnt", "annual_inc", "fico_range_low", "fico_range_high"}
SENSITIVITY_ONLY = {"mths_since_last_delinq", "mths_since_last_record"}


def _numeric_mismatches(actual: pd.Series, expected: pd.Series) -> int:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype="float64")
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype="float64")
    return int((~np.isclose(a, e, rtol=1e-8, atol=1e-8, equal_nan=True)).sum())


def _date_mismatches(actual: pd.Series, expected: pd.Series) -> int:
    a = pd.to_datetime(actual, errors="coerce")
    e = pd.to_datetime(expected, errors="coerce")
    return int(((a.ne(e)) & ~(a.isna() & e.isna())).sum())


def _parsing_checks(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, int]:
    raw_indexed = raw.assign(id=pd.to_numeric(raw["id"], errors="coerce")).set_index("id")
    cleaned_indexed = cleaned.assign(
        id=pd.to_numeric(cleaned["id"], errors="coerce")
    ).set_index("id")
    joined = cleaned_indexed.join(raw_indexed, rsuffix="_raw", how="left")
    checks: dict[str, int] = {}
    checks["term_months"] = _numeric_mismatches(
        joined["term_months"],
        pd.to_numeric(joined["term"].astype("string").str.extract(r"(\d+)", expand=False)),
    )
    for column in ("int_rate", "revol_util"):
        expected = pd.to_numeric(
            joined[f"{column}_raw"].astype("string").str.replace("%", "", regex=False),
            errors="coerce",
        )
        checks[column] = _numeric_mismatches(joined[column], expected)
    employment = joined["emp_length"].astype("string").str.strip().str.lower()
    expected_employment = pd.to_numeric(
        employment.str.extract(r"(\d+)", expand=False), errors="coerce"
    ).mask(employment.str.contains(r"^<\s*1", na=False), 0)
    checks["emp_length_years"] = _numeric_mismatches(
        joined["emp_length_years"], expected_employment
    )
    expected_issue = pd.to_datetime(joined["issue_d"], format="%b-%Y", errors="coerce")
    checks["issue_date"] = _date_mismatches(joined["issue_date"], expected_issue)
    checks["fico_score"] = _numeric_mismatches(
        joined["fico_score"],
        (joined["fico_range_low"] + joined["fico_range_high"]) / 2,
    )

    credit_raw = joined["earliest_cr_line_raw"].astype("string").str.strip()
    two_digit = credit_raw.str.contains(r"^[A-Za-z]{3}-\d{2}$", na=False)
    expected_credit = pd.to_datetime(credit_raw, format="%b-%Y", errors="coerce")
    expected_credit = expected_credit.fillna(
        pd.to_datetime(credit_raw.where(two_digit), format="%b-%y", errors="coerce")
    )
    wrong_century = two_digit & expected_credit.gt(expected_issue)
    if wrong_century.any():
        expected_credit.loc[wrong_century] = expected_credit.loc[wrong_century].map(
            lambda value: value - pd.DateOffset(years=100)
        )
    checks["earliest_cr_line_date"] = _date_mismatches(
        joined["earliest_cr_line_date"], expected_credit
    )
    expected_months = (
        (expected_issue.dt.year - expected_credit.dt.year) * 12
        + expected_issue.dt.month
        - expected_credit.dt.month
    )
    checks["credit_history_months"] = _numeric_mismatches(
        joined["credit_history_months"], expected_months
    )
    return checks


def _dictionary(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in cleaned.columns:
        if column in SOURCE_INFO:
            description, unit = SOURCE_INFO[column]
            source = "Frozen raw dataset"
        elif column in DERIVED_INFO:
            description, unit, source = DERIVED_INFO[column]
        elif column.endswith("_was_missing"):
            original = column.removesuffix("_was_missing")
            description = f"Indicator that {original} was missing after cleaning."
            unit = "binary"
            source = f"Derived from missingness of {original}"
        else:
            raise ValueError(f"No data-dictionary metadata for cleaned column: {column}")

        if column in APPROVED_DIRECT:
            permitted, use = "yes", "Approved origination-time predictor"
        elif column in SOURCE_ONLY:
            permitted, use = "no", "Source for approved derived predictors only"
        elif column in SENSITIVITY_ONLY:
            permitted, use = "no", "Only missingness indicators may be sensitivity features"
        elif column == "default":
            permitted, use = "no", "Prediction/inference target"
        else:
            permitted, use = "no", "Identifier, audit, raw date, or excluded field"
        rows.append(
            {
                "column": column,
                "pandas_dtype": str(cleaned[column].dtype),
                "description": description,
                "unit": unit,
                "source_or_derivation": source,
                "permitted_as_predictor": permitted,
                "approved_use": use,
            }
        )
    return pd.DataFrame(rows)


def run_audit(raw_path: Path, input_path: Path, report_path: Path, output_dir: Path) -> dict:
    for path, label in ((raw_path, "Raw data"), (input_path, "Cleaned data"), (report_path, "Cleaning report")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    raw = read_raw_data(raw_path)
    cleaned = pd.read_csv(input_path, compression="infer", low_memory=False)
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    exact_duplicates = raw.duplicated()
    after_exact = raw.loc[~exact_duplicates].copy()
    usable_id = after_exact["id"].notna() & after_exact["id"].astype("string").str.strip().ne("")
    duplicate_ids = usable_id & after_exact.duplicated("id", keep="first")
    unique_raw = after_exact.loc[~duplicate_ids].copy()
    expected_default = construct_default(unique_raw["loan_status"])
    unresolved = unique_raw.loc[expected_default.isna(), "loan_status"]
    status_display = unresolved.astype("string").fillna("<missing>").str.strip()
    status_display = status_display.mask(status_display.eq(""), "<blank>")
    status_counts = {str(k): int(v) for k, v in status_display.value_counts().sort_index().items()}

    required = {"id", "default", "loan_status", "issue_date"}
    missing_required = sorted(required - set(cleaned.columns))
    if missing_required:
        raise ValueError(f"Cleaned data is missing required columns: {', '.join(missing_required)}")
    expected_schema = set(ORIGINATION_COLUMNS) | set(DERIVED_INFO) | {
        f"{column}_was_missing" for column in IMPORTANT_FIELDS
    }
    unexpected_columns = sorted(set(cleaned.columns) - expected_schema)
    outcome = pd.to_numeric(cleaned["default"], errors="coerce")
    outcome_counts = outcome.value_counts()
    issue_dates = pd.to_datetime(cleaned["issue_date"], errors="coerce")
    mapped_outcome = construct_default(cleaned["loan_status"])

    missingness = pd.DataFrame(
        {
            "column": cleaned.columns,
            "missing_count": [int(cleaned[c].isna().sum()) for c in cleaned.columns],
            "missing_proportion": [float(cleaned[c].isna().mean()) for c in cleaned.columns],
        }
    )
    report_missing_mismatches = {
        row.column: int(row.missing_count)
        for row in missingness.itertuples(index=False)
        if int(report["missingness"].get(row.column, {}).get("count", -1)) != int(row.missing_count)
    }
    indicator_mismatches = {
        column: int(
            (cleaned[f"{column}_was_missing"] != cleaned[column].isna().astype("int8")).sum()
        )
        for column in IMPORTANT_FIELDS
        if column in cleaned and f"{column}_was_missing" in cleaned
        and (cleaned[f"{column}_was_missing"] != cleaned[column].isna()).any()
    }
    invalid_remaining = {}
    for column, bounds in NUMERIC_BOUNDS.items():
        if column not in cleaned:
            continue
        values = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = pd.Series(False, index=cleaned.index)
        if "minimum" in bounds:
            invalid |= values.lt(bounds["minimum"]).fillna(False)
        if "maximum" in bounds:
            invalid |= values.gt(bounds["maximum"]).fillna(False)
        if bounds.get("strictly_positive"):
            invalid |= values.le(0).fillna(False)
        if invalid.any():
            invalid_remaining[column] = int(invalid.sum())

    parsing = _parsing_checks(unique_raw, cleaned)
    actual = {
        "raw_row_count": len(raw),
        "final_row_count": len(cleaned),
        "status_exclusions": {"total": len(unresolved), "by_status": status_counts},
        "duplicates": {
            "exact_rows_removed": int(exact_duplicates.sum()),
            "duplicate_id_rows_removed": int(duplicate_ids.sum()),
        },
        "outcome_counts": {
            "fully_paid": int(outcome_counts.get(0, 0)),
            "charged_off_or_default": int(outcome_counts.get(1, 0)),
        },
        "default_rate": float(outcome.mean()),
        "issue_date_range": {
            "minimum": issue_dates.min().strftime("%Y-%m-%d"),
            "maximum": issue_dates.max().strftime("%Y-%m-%d"),
        },
    }
    report_checks = {
        key: actual[key] == report.get(key)
        for key in ("raw_row_count", "final_row_count", "status_exclusions", "duplicates", "outcome_counts", "issue_date_range")
    }
    report_checks["default_rate"] = bool(np.isclose(actual["default_rate"], report["default_rate"]))
    checks = {
        **report_checks,
        "outcome_binary_and_nonmissing": bool(outcome.notna().all() and outcome.isin([0, 1]).all()),
        "outcome_matches_original_status": bool(outcome.astype("Int64").equals(mapped_outcome)),
        "ids_nonmissing_and_unique": bool(cleaned["id"].notna().all() and not cleaned["id"].duplicated().any()),
        "issue_dates_valid": bool(issue_dates.notna().all()),
        "no_post_outcome_fields": not bool(set(cleaned.columns) & LEAKAGE_FIELDS),
        "schema_expected": not unexpected_columns,
        "missingness_matches_report": not report_missing_mismatches,
        "missing_indicators_consistent": not indicator_mismatches,
        "no_impossible_values_remain": not invalid_remaining,
        "real_value_parsing_matches": not any(parsing.values()),
    }
    extremes = {
        column: float(pd.to_numeric(cleaned[column], errors="coerce").max())
        for column in ("loan_amnt", "annual_inc", "dti", "revol_util", "credit_history_months")
    }
    audit = {
        "passed": all(checks.values()),
        "checks": checks,
        "actual": actual,
        "unexpected_columns": unexpected_columns,
        "leakage_fields_present": sorted(set(cleaned.columns) & LEAKAGE_FIELDS),
        "missingness_report_mismatches": report_missing_mismatches,
        "missing_indicator_mismatches": indicator_mismatches,
        "invalid_values_remaining": invalid_remaining,
        "parsing_mismatch_counts": parsing,
        "parsing_rows_checked": len(cleaned),
        "retained_extreme_values": extremes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(output_dir / "missingness.csv", index=False)
    _dictionary(cleaned).to_csv(output_dir / "data_dictionary.csv", index=False)
    with (output_dir / "cleaning_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Cleaning audit failed: {', '.join(failed)}. See cleaning_validation.json")
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args(argv)
    audit = run_audit(args.raw, args.input, args.report, args.output_dir)
    print(f"Cleaning audit passed for {audit['actual']['final_row_count']:,} rows")
    print(f"Audit tables written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
