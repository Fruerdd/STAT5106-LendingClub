"""Prespecified feature preparation for Member 4 statistical inference.

This module implements the frozen contract in
``report/member4_inference_specification.md``. It does not select features from
outcomes and does not expose full-cohort preprocessing values to predictive
modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Scenario = Literal[
    "primary",
    "mature",
    "unwinsorized",
    "employment_complete_case",
]

OUTCOME = "default"
RARE_LEVEL = "OTHER_RARE"
RARE_MIN_COUNT = 1_000
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995
MATURITY_CUTOFF = pd.Period("2018-12", freq="M")

REQUIRED_INPUT_COLUMNS = (
    "default",
    "loan_amnt",
    "annual_inc",
    "fico_score",
    "dti",
    "dti_was_missing",
    "revol_util",
    "credit_history_months",
    "emp_length_years",
    "emp_length_years_was_missing",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "term_months",
    "purpose",
    "home_ownership",
    "verification_status",
    "application_type",
    "issue_date",
    "grade",
    "int_rate",
    "mths_since_last_delinq",
    "mths_since_last_record",
)

NUMERIC_INPUT_COLUMNS = (
    "default",
    "loan_amnt",
    "annual_inc",
    "fico_score",
    "dti",
    "dti_was_missing",
    "revol_util",
    "credit_history_months",
    "emp_length_years",
    "emp_length_years_was_missing",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "term_months",
    "int_rate",
    "mths_since_last_delinq",
    "mths_since_last_record",
)

LOW_MISSING_COMPLETE_CASE_COLUMNS = (
    "revol_util",
    "credit_history_months",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
)

PRIMARY_WINSOR_COLUMNS = (
    "log2_annual_inc",
    "log2_loan_to_annual_income",
    "dti_per_10",
    "revol_util_per_10",
    "credit_history_years",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc_per_5",
    "pub_rec",
)

REFERENCE_LEVELS = {
    "term_months": "36",
    "purpose_grouped": "debt_consolidation",
    "home_ownership_grouped": "MORTGAGE",
    "verification_status": "Not Verified",
    "application_type": "Individual",
    "issue_year": "2015",
    "grade": "A",
}

ENGINEERED_MODEL_COLUMNS = (
    "default",
    "fico_per_10",
    "log2_annual_inc",
    "log2_loan_to_annual_income",
    "income_or_dti_missing",
    "dti_per_10",
    "revol_util_per_10",
    "credit_history_years",
    "emp_length_years",
    "emp_length_years_was_missing",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc_per_5",
    "pub_rec",
    "term_months",
    "purpose_grouped",
    "home_ownership_grouped",
    "verification_status",
    "application_type",
    "issue_year",
    "grade",
    "int_rate_per_5",
    "has_last_delinq_record",
    "has_last_public_record_date",
)

BASE_MODEL_TERMS = (
    "fico_per_10",
    "log2_annual_inc",
    "log2_loan_to_annual_income",
    "income_or_dti_missing",
    "dti_per_10",
    "revol_util_per_10",
    "credit_history_years",
    "emp_length_years",
    "emp_length_years_was_missing",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc_per_5",
    "pub_rec",
    "C(term_months, Treatment(reference='36'))",
    "C(purpose_grouped, Treatment(reference='debt_consolidation'))",
    "C(home_ownership_grouped, Treatment(reference='MORTGAGE'))",
    "C(verification_status, Treatment(reference='Not Verified'))",
    "C(application_type, Treatment(reference='Individual'))",
    "C(issue_year, Treatment(reference='2015'))",
)

MODEL_EXTRA_TERMS = {
    "A": (),
    "B": ("C(grade, Treatment(reference='A'))",),
    "C": ("int_rate_per_5",),
}


@dataclass
class PreparedInferenceData:
    """Engineered inference frame and a JSON-serializable preparation audit."""

    frame: pd.DataFrame
    summary: dict[str, object]


def load_inference_input(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read required columns and optionally take a deterministic development sample."""
    if not path.is_file():
        raise FileNotFoundError(f"Inference input not found: {path}")

    header = pd.read_csv(path, nrows=0, compression="infer")
    missing = sorted(set(REQUIRED_INPUT_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"Inference input is missing columns: {', '.join(missing)}")

    frame = pd.read_csv(
        path,
        usecols=list(REQUIRED_INPUT_COLUMNS),
        compression="infer",
        low_memory=False,
    )
    if nrows is not None and len(frame) > nrows:
        frame = (
            frame.sample(n=nrows, random_state=42, replace=False)
            .sort_index()
            .reset_index(drop=True)
        )
    return frame


def _ordered_category(series: pd.Series, reference: str, name: str) -> pd.Series:
    values = series.astype("string")
    observed = sorted(str(value) for value in values.dropna().unique())
    if reference not in observed:
        raise ValueError(f"Reference level {reference!r} is absent from {name}")
    categories = [reference, *(value for value in observed if value != reference)]
    return pd.Series(
        pd.Categorical(values, categories=categories),
        index=series.index,
        name=series.name,
    )


def _group_rare_levels(
    series: pd.Series,
    *,
    reference: str,
    minimum_count: int,
) -> tuple[pd.Series, list[str], dict[str, int]]:
    values = series.astype("string").fillna(RARE_LEVEL)
    counts = values.value_counts(dropna=False)
    rare_levels = sorted(
        str(level)
        for level, count in counts.items()
        if int(count) < minimum_count and str(level) != reference
    )
    grouped = values.mask(values.isin(rare_levels), RARE_LEVEL)
    return (
        grouped,
        rare_levels,
        {str(level): int(count) for level, count in counts.sort_index().items()},
    )


def _validate_inputs(frame: pd.DataFrame) -> None:
    outcome = pd.to_numeric(frame[OUTCOME], errors="coerce")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError("default must contain only nonmissing 0 and 1 values")

    required_nonmissing = (
        "loan_amnt",
        "fico_score",
        "term_months",
        "purpose",
        "home_ownership",
        "verification_status",
        "application_type",
        "issue_date",
        "grade",
        "int_rate",
    )
    missing = {
        column: int(frame[column].isna().sum())
        for column in required_nonmissing
        if frame[column].isna().any()
    }
    if missing:
        raise ValueError(f"Unexpected missing values in required inference fields: {missing}")

    terms = set(pd.to_numeric(frame["term_months"], errors="coerce").dropna().unique())
    if not terms.issubset({36.0, 60.0}):
        raise ValueError(f"Unexpected loan terms: {sorted(terms)}")


def _maturity_mask(issue_date: pd.Series, term_months: pd.Series) -> pd.Series:
    issue_period = issue_date.dt.to_period("M")
    maturity_ordinal = issue_period.astype("int64") + term_months.astype("int64")
    return maturity_ordinal.le(MATURITY_CUTOFF.ordinal)


def _winsorize(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for column in columns:
        lower, upper = frame[column].quantile(
            [lower_quantile, upper_quantile],
            interpolation="linear",
        )
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError(f"Could not determine winsorization bounds for {column}")
        frame[column] = frame[column].clip(lower=lower, upper=upper)
        bounds[column] = {"lower": float(lower), "upper": float(upper)}
    return bounds


def prepare_inference_data(
    raw: pd.DataFrame,
    scenario: Scenario = "primary",
    *,
    rare_min_count: int = RARE_MIN_COUNT,
    winsor_lower: float = WINSOR_LOWER,
    winsor_upper: float = WINSOR_UPPER,
) -> PreparedInferenceData:
    """Apply the frozen feature contract for one prespecified scenario."""
    valid_scenarios = {
        "primary",
        "mature",
        "unwinsorized",
        "employment_complete_case",
    }
    if scenario not in valid_scenarios:
        raise ValueError(f"Unknown scenario {scenario!r}")
    if rare_min_count < 1:
        raise ValueError("rare_min_count must be positive")
    if not 0 <= winsor_lower < winsor_upper <= 1:
        raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1")

    frame = raw.loc[:, REQUIRED_INPUT_COLUMNS].copy()
    for column in NUMERIC_INPUT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="coerce")
    _validate_inputs(frame)

    input_rows = len(frame)
    if scenario == "mature":
        frame = frame.loc[
            _maturity_mask(frame["issue_date"], frame["term_months"])
        ].copy()
    rows_after_maturity_filter = len(frame)

    complete_case_before = len(frame)
    frame = frame.dropna(subset=list(LOW_MISSING_COMPLETE_CASE_COLUMNS)).copy()
    low_missing_rows_excluded = complete_case_before - len(frame)

    observed_dti_missing = frame["dti"].isna()
    supplied_dti_missing = frame["dti_was_missing"].eq(1)
    if not observed_dti_missing.equals(supplied_dti_missing):
        raise ValueError("dti_was_missing is inconsistent with dti")
    dti_median = float(frame["dti"].median())
    frame["dti"] = frame["dti"].fillna(dti_median)

    observed_employment_missing = frame["emp_length_years"].isna()
    supplied_employment_missing = frame["emp_length_years_was_missing"].eq(1)
    if not observed_employment_missing.equals(supplied_employment_missing):
        raise ValueError("emp_length_years_was_missing is inconsistent with emp_length_years")

    employment_rows_excluded = 0
    if scenario == "employment_complete_case":
        before = len(frame)
        frame = frame.loc[~observed_employment_missing].copy()
        employment_rows_excluded = before - len(frame)
        frame["emp_length_years_was_missing"] = 0
        employment_median: float | None = None
    else:
        employment_median = float(frame["emp_length_years"].median())
        frame["emp_length_years"] = frame["emp_length_years"].fillna(employment_median)

    positive_income = frame["annual_inc"].gt(0) & frame["annual_inc"].notna()
    frame["income_or_dti_missing"] = (
        (~positive_income) | observed_dti_missing
    ).astype("int8")
    frame["log2_annual_inc"] = np.log2(frame["annual_inc"].where(positive_income))
    ratio = frame["loan_amnt"] / frame["annual_inc"].where(positive_income)
    frame["log2_loan_to_annual_income"] = np.log2(1 + ratio)

    income_medians: dict[str, float] = {}
    for column in ("log2_annual_inc", "log2_loan_to_annual_income"):
        median = float(frame[column].median())
        if not np.isfinite(median):
            raise ValueError(f"No finite values available to impute {column}")
        frame[column] = frame[column].fillna(median)
        income_medians[column] = median

    frame["fico_per_10"] = frame["fico_score"] / 10
    frame["dti_per_10"] = frame["dti"] / 10
    frame["revol_util_per_10"] = frame["revol_util"] / 10
    frame["credit_history_years"] = frame["credit_history_months"] / 12
    frame["open_acc_per_5"] = frame["open_acc"] / 5
    frame["int_rate_per_5"] = frame["int_rate"] / 5
    frame["issue_year"] = frame["issue_date"].dt.year.astype("Int64").astype("string")
    frame["term_months"] = frame["term_months"].astype("int64").astype("string")
    frame["has_last_delinq_record"] = frame["mths_since_last_delinq"].notna().astype("int8")
    frame["has_last_public_record_date"] = (
        frame["mths_since_last_record"].notna().astype("int8")
    )

    (
        frame["purpose_grouped"],
        rare_purpose_levels,
        purpose_counts,
    ) = _group_rare_levels(
        frame["purpose"],
        reference=REFERENCE_LEVELS["purpose_grouped"],
        minimum_count=rare_min_count,
    )
    (
        frame["home_ownership_grouped"],
        rare_home_levels,
        home_counts,
    ) = _group_rare_levels(
        frame["home_ownership"],
        reference=REFERENCE_LEVELS["home_ownership_grouped"],
        minimum_count=rare_min_count,
    )

    for column, reference in REFERENCE_LEVELS.items():
        frame[column] = _ordered_category(frame[column], reference, column)

    should_winsorize = scenario != "unwinsorized"
    winsor_bounds = (
        _winsorize(
            frame,
            PRIMARY_WINSOR_COLUMNS,
            winsor_lower,
            winsor_upper,
        )
        if should_winsorize
        else {}
    )

    engineered = frame.loc[:, ENGINEERED_MODEL_COLUMNS].copy()
    if engineered.isna().any().any():
        missing = {
            column: int(count)
            for column, count in engineered.isna().sum().items()
            if count
        }
        raise ValueError(f"Engineered inference frame contains missing values: {missing}")

    category_counts = {
        column: {
            str(level): int(count)
            for level, count in engineered[column].value_counts().sort_index().items()
        }
        for column in REFERENCE_LEVELS
    }
    summary: dict[str, object] = {
        "scenario": scenario,
        "input_rows": int(input_rows),
        "rows_after_maturity_filter": int(rows_after_maturity_filter),
        "low_missing_rows_excluded": int(low_missing_rows_excluded),
        "employment_rows_excluded": int(employment_rows_excluded),
        "final_rows": int(len(engineered)),
        "default_count": int(engineered[OUTCOME].sum()),
        "default_rate": float(engineered[OUTCOME].mean()),
        "annual_income_nonpositive_or_missing": int((~positive_income).sum()),
        "dti_missing_count": int(supplied_dti_missing.sum()),
        "income_or_dti_missing_count": int(engineered["income_or_dti_missing"].sum()),
        "employment_missing_count": int(supplied_employment_missing.sum()),
        "employment_median": employment_median,
        "dti_median": dti_median,
        "income_transform_medians": income_medians,
        "rare_min_count": int(rare_min_count),
        "rare_purpose_levels": rare_purpose_levels,
        "rare_home_ownership_levels": rare_home_levels,
        "original_purpose_counts": purpose_counts,
        "original_home_ownership_counts": home_counts,
        "category_counts": category_counts,
        "winsorized": should_winsorize,
        "winsor_quantiles": {
            "lower": float(winsor_lower),
            "upper": float(winsor_upper),
        },
        "winsor_bounds": winsor_bounds,
        "maturity_cutoff": str(MATURITY_CUTOFF),
    }
    return PreparedInferenceData(frame=engineered, summary=summary)


def build_model_formula(
    model_name: Literal["A", "B", "C"],
    *,
    include_employment_missing_indicator: bool = True,
    include_income_or_dti_missing_indicator: bool = True,
) -> str:
    """Return one of the three frozen model formulas."""
    if model_name not in MODEL_EXTRA_TERMS:
        raise ValueError(f"Unknown inference model {model_name!r}")
    terms = list(BASE_MODEL_TERMS)
    if not include_employment_missing_indicator:
        terms.remove("emp_length_years_was_missing")
    if not include_income_or_dti_missing_indicator:
        terms.remove("income_or_dti_missing")
    terms.extend(MODEL_EXTRA_TERMS[model_name])
    return f"{OUTCOME} ~ " + " + ".join(terms)
