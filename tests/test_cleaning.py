from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import src.paths as paths
from src.lending_club_data_cleaning import (
    LEAKAGE_FIELDS,
    clean_dataframe,
    construct_default,
    main,
    parse_employment_length,
    parse_percentage,
)


def resolved_row(loan_id: str, status: str = "Fully Paid", **overrides) -> dict:
    row = {
        "id": loan_id,
        "loan_status": status,
        "issue_d": "Jan-2020",
        "earliest_cr_line": "Jan-68",
        "term": "36 months",
        "int_rate": "12.5%",
        "revol_util": "40%",
        "emp_length": "5 years",
        "loan_amnt": 10000,
        "installment": 250,
        "annual_inc": 60000,
        "dti": 15,
        "fico_range_low": 700,
        "fico_range_high": 704,
    }
    row.update(overrides)
    return row


def test_outcome_construction() -> None:
    statuses = pd.Series(["Fully Paid", "Charged Off", "Default", " charged off "])
    result = construct_default(statuses)
    expected = pd.Series([0, 1, 1, 1], dtype="Int64")
    pd.testing.assert_series_equal(result, expected)


def test_unresolved_and_unknown_outcomes_are_excluded() -> None:
    raw = pd.DataFrame(
        [
            resolved_row("paid"),
            resolved_row("off", "Charged Off"),
            resolved_row("current", "Current"),
            resolved_row("late", "Late (31-120 days)"),
            resolved_row("grace", "In Grace Period"),
            resolved_row("unknown", "Unexpected status"),
        ]
    )
    cleaned, report = clean_dataframe(raw)
    assert cleaned["id"].tolist() == ["paid", "off"]
    assert cleaned["default"].tolist() == [0, 1]
    assert report["status_exclusions"]["total"] == 4
    assert report["status_exclusions"]["by_status"]["Current"] == 1
    assert "loan_status" in cleaned


def test_percentage_and_employment_length_parsing() -> None:
    percentages = parse_percentage(pd.Series(["12.5%", " 7 % ", "bad", None]))
    assert percentages.iloc[0] == pytest.approx(12.5)
    assert percentages.iloc[1] == pytest.approx(7)
    assert percentages.iloc[2:].isna().all()

    employment = parse_employment_length(
        pd.Series(["< 1 year", "1 year", "10+ years", "n/a"])
    )
    assert employment.iloc[:3].tolist() == [0, 1, 10]
    assert pd.isna(employment.iloc[3])


def test_exact_duplicates_and_duplicate_ids_are_removed_and_counted() -> None:
    original = resolved_row("A")
    changed_same_id = resolved_row("A", loan_amnt=20000)
    raw = pd.DataFrame(
        [
            original,
            resolved_row("B", "Charged Off"),
            changed_same_id,
            original.copy(),
            resolved_row("C", "Current"),
        ]
    )
    cleaned, report = clean_dataframe(raw)
    assert cleaned["id"].tolist() == ["A", "B"]
    assert report["raw_row_count"] == 5
    assert report["final_row_count"] == 2
    assert report["duplicates"] == {
        "exact_rows_removed": 1,
        "duplicate_id_rows_removed": 1,
    }


def test_impossible_values_become_missing_but_extreme_rows_are_retained() -> None:
    impossible = resolved_row(
        "bad",
        loan_amnt=0,
        annual_inc=-1,
        dti=-2,
        int_rate="-5%",
        revol_util="-10%",
        fico_range_low=200,
        fico_range_high=900,
    )
    extreme = resolved_row(
        "extreme",
        "Charged Off",
        loan_amnt=1_000_000_000,
        annual_inc=1_000_000_000_000,
        dti=500,
        revol_util="150%",
    )
    cleaned, report = clean_dataframe(pd.DataFrame([impossible, extreme]))
    bad = cleaned.loc[cleaned["id"] == "bad"].iloc[0]
    for column in (
        "loan_amnt",
        "annual_inc",
        "dti",
        "int_rate",
        "revol_util",
        "fico_range_low",
        "fico_range_high",
        "fico_score",
    ):
        assert pd.isna(bad[column]), column
    assert len(cleaned) == 2
    assert cleaned.loc[cleaned["id"] == "extreme", "revol_util"].item() == 150
    assert cleaned.loc[cleaned["id"] == "extreme", "annual_inc"].item() == 1_000_000_000_000
    assert report["invalid_values"]["annual_inc"] == 1
    assert bad["annual_inc_was_missing"] == 1


def test_dates_credit_history_fico_and_term_are_derived() -> None:
    cleaned, report = clean_dataframe(pd.DataFrame([resolved_row("A", emp_length="< 1 year")]))
    row = cleaned.iloc[0]
    assert row["term_months"] == 36
    assert row["emp_length_years"] == 0
    assert row["issue_date"] == pd.Timestamp("2020-01-01")
    assert row["earliest_cr_line_date"] == pd.Timestamp("1968-01-01")
    assert row["credit_history_months"] == 624
    assert row["fico_score"] == 702
    assert report["issue_date_range"] == {"minimum": "2020-01-01", "maximum": "2020-01-01"}


def test_leakage_fields_do_not_appear_in_cleaned_output() -> None:
    row = resolved_row("A")
    row.update({field: "post-outcome" for field in LEAKAGE_FIELDS})
    cleaned, _ = clean_dataframe(pd.DataFrame([row]))
    assert not set(cleaned.columns).intersection(LEAKAGE_FIELDS)


def test_paths_are_derived_from_repository_location() -> None:
    assert paths.PROJECT_DIR == Path(paths.__file__).resolve().parents[1]
    for path in (
        paths.RAW_DATA_DIR,
        paths.RAW_DATA_PATH,
        paths.CLEANED_DATA_DIR,
        paths.FIGURES_DIR,
        paths.TABLES_DIR,
    ):
        assert path.relative_to(paths.PROJECT_DIR) is not None
    assert paths.RAW_DATA_PATH.relative_to(paths.PROJECT_DIR) == Path(
        "data/raw/accepted_2007_to_2018Q4.csv.gz"
    )


def test_gzip_cli_outputs_report_and_preserves_input(tmp_path: Path) -> None:
    input_path = tmp_path / "synthetic.csv.gz"
    output_dir = tmp_path / "nested" / "clean"
    raw = pd.DataFrame(
        [
            {**resolved_row("A"), "total_pymnt": 10000, "recoveries": 0},
            {**resolved_row("B", "Default"), "total_pymnt": 100, "recoveries": 5},
        ]
    )
    raw.to_csv(input_path, index=False, compression="infer")
    digest_before = hashlib.sha256(input_path.read_bytes()).hexdigest()

    assert main(["--input", str(input_path), "--output-dir", str(output_dir)]) == 0

    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == digest_before
    output_path = output_dir / "lending_club_clean.csv.gz"
    report_path = output_dir / "cleaning_report.json"
    cleaned = pd.read_csv(output_path, compression="infer")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(cleaned) == 2
    assert not set(cleaned.columns).intersection(LEAKAGE_FIELDS)
    assert report["raw_row_count"] == 2
    assert report["final_row_count"] == 2
    assert report["outcome_counts"] == {"fully_paid": 1, "charged_off_or_default": 1}
    assert report["default_rate"] == 0.5
