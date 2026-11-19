from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.inference_features import (
    PRIMARY_WINSOR_COLUMNS,
    RARE_LEVEL,
    build_model_formula,
    prepare_inference_data,
)
from src.inference_models import (
    _formula_design_info,
    fit_inference_model,
    glm_score_sandwich_covariance,
)


def synthetic_cleaned_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5106)
    issue_year = rng.choice([2012, 2015, 2017], size=n, p=[0.25, 0.5, 0.25])
    term = rng.choice([36.0, 60.0], size=n, p=[0.75, 0.25])
    fico = np.clip(rng.normal(700, 30, n), 620, 820)
    annual_inc = rng.lognormal(np.log(65_000), 0.45, n)
    dti = np.clip(rng.normal(18, 7, n), 0, 60)
    revol_util = np.clip(rng.normal(52, 22, n), 0, 130)
    employment = rng.integers(0, 11, n).astype("float64")

    income_problem = np.arange(0, 8)
    annual_inc[income_problem[:6]] = 0
    annual_inc[income_problem[6:]] = np.nan
    dti[income_problem[:5]] = np.nan
    dti_missing = np.isnan(dti).astype("int8")

    employment_missing = np.arange(10, n, 17)
    employment[employment_missing] = np.nan
    employment_was_missing = np.isnan(employment).astype("int8")

    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "small_business"],
        size=n,
        p=[0.65, 0.30, 0.05],
    ).astype(object)
    purpose[:8] = "educational"
    home = rng.choice(["MORTGAGE", "RENT", "OWN"], size=n).astype(object)
    home[:6] = "NONE"
    application = rng.choice(["Individual", "Joint App"], size=n, p=[0.9, 0.1])
    grade = rng.choice(list("ABCDEFG"), size=n)
    int_rate = np.clip(8 + (760 - fico) * 0.08 + rng.normal(0, 1.5, n), 5, 31)
    loan_amnt = rng.integers(1_000, 40_001, n).astype("float64")

    linear_predictor = (
        -1.3
        - 0.025 * (fico - 700)
        + 0.025 * np.nan_to_num(dti, nan=18)
        + 0.55 * (term == 60)
        + 0.25 * (purpose == "small_business")
    )
    probability = 1 / (1 + np.exp(-linear_predictor))
    default = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "default": default,
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "fico_score": fico,
            "dti": dti,
            "dti_was_missing": dti_missing,
            "revol_util": revol_util,
            "credit_history_months": rng.integers(24, 480, n).astype("float64"),
            "emp_length_years": employment,
            "emp_length_years_was_missing": employment_was_missing,
            "delinq_2yrs": rng.poisson(0.3, n).astype("float64"),
            "inq_last_6mths": rng.poisson(1.0, n).astype("float64"),
            "open_acc": rng.integers(1, 30, n).astype("float64"),
            "pub_rec": rng.poisson(0.15, n).astype("float64"),
            "term_months": term,
            "purpose": purpose,
            "home_ownership": home,
            "verification_status": rng.choice(
                ["Not Verified", "Source Verified", "Verified"],
                size=n,
            ),
            "application_type": application,
            "issue_date": pd.to_datetime(
                pd.Series(issue_year).astype(str) + "-06-01"
            ),
            "grade": grade,
            "int_rate": int_rate,
            "mths_since_last_delinq": np.where(
                rng.random(n) < 0.5,
                rng.integers(1, 120, n),
                np.nan,
            ),
            "mths_since_last_record": np.where(
                rng.random(n) < 0.2,
                rng.integers(1, 120, n),
                np.nan,
            ),
        }
    )
    frame.loc[20, "revol_util"] = np.nan
    return frame


def test_feature_contract_handles_missingness_rare_levels_and_winsorization() -> None:
    raw = synthetic_cleaned_frame()
    prepared = prepare_inference_data(raw, "primary", rare_min_count=20)

    assert len(prepared.frame) == len(raw) - 1
    assert not prepared.frame.isna().any().any()
    assert int(prepared.frame["income_or_dti_missing"].sum()) == 8
    assert RARE_LEVEL in prepared.frame["purpose_grouped"].cat.categories
    assert RARE_LEVEL in prepared.frame["home_ownership_grouped"].cat.categories
    assert prepared.summary["rare_purpose_levels"] == ["educational"]
    assert prepared.summary["rare_home_ownership_levels"] == ["NONE"]
    assert set(prepared.summary["winsor_bounds"]) == set(PRIMARY_WINSOR_COLUMNS)
    assert prepared.summary["low_missing_rows_excluded"] == 1
    assert prepared.summary["dti_missing_count"] == 5


def test_maturity_and_employment_complete_case_scenarios() -> None:
    raw = synthetic_cleaned_frame()
    mature = prepare_inference_data(raw, "mature", rare_min_count=10)
    employment = prepare_inference_data(
        raw,
        "employment_complete_case",
        rare_min_count=10,
    )

    expected_mature = (
        ((raw["issue_date"].dt.year <= 2015) & raw["term_months"].eq(36))
        | ((raw["issue_date"].dt.year <= 2013) & raw["term_months"].eq(60))
    )
    expected_mature &= raw["revol_util"].notna()
    assert len(mature.frame) == int(expected_mature.sum())
    assert employment.summary["employment_rows_excluded"] == int(
        raw.loc[raw["revol_util"].notna(), "emp_length_years"].isna().sum()
    )
    assert employment.frame["emp_length_years_was_missing"].eq(0).all()


def test_prespecified_formulas_keep_lender_proxies_separate() -> None:
    formula_a = build_model_formula("A")
    formula_b = build_model_formula("B")
    formula_c = build_model_formula("C")
    complete_case = build_model_formula(
        "A",
        include_employment_missing_indicator=False,
        include_income_or_dti_missing_indicator=False,
    )

    assert "grade" not in formula_a and "int_rate_per_5" not in formula_a
    assert "grade" in formula_b and "int_rate_per_5" not in formula_b
    assert "grade" not in formula_c and "int_rate_per_5" in formula_c
    assert "emp_length_years_was_missing" not in complete_case
    assert "income_or_dti_missing" not in complete_case


def test_custom_hc0_matches_statsmodels_and_hc3_fit_is_auditable() -> None:
    rng = np.random.default_rng(2026)
    n = 800
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    group = rng.choice(["base", "other"], size=n)
    probability = 1 / (
        1 + np.exp(-(-0.4 + 0.7 * x - 0.3 * z + 0.25 * (group == "other")))
    )
    frame = pd.DataFrame(
        {
            "default": rng.binomial(1, probability),
            "x": x,
            "z": z,
            "group": group,
        }
    )
    formula = "default ~ x + z + C(group, Treatment(reference='base'))"
    model = smf.glm(formula, data=frame, family=sm.families.Binomial())
    nonrobust = model.fit()
    custom_hc0, _ = glm_score_sandwich_covariance(
        nonrobust,
        adjustment="HC0",
        chunk_size=127,
    )
    builtin_hc0 = model.fit(cov_type="HC0").cov_params().to_numpy()
    np.testing.assert_allclose(custom_hc0, builtin_hc0, rtol=1e-6, atol=1e-9)

    fitted = fit_inference_model(
        frame,
        formula,
        scenario="synthetic",
        model_name="A",
        hc3_chunk_size=127,
    )
    assert fitted.audit["covariance_type"] == "HC3"
    assert fitted.audit["converged"] is True
    assert fitted.audit["leverage_denominator_clipped_count"] == 0
    assert fitted.coefficients["robust_se"].gt(0).all()
    assert fitted.coefficients.loc[
        fitted.coefficients["parameter"].ne("Intercept"),
        "q_value",
    ].notna().all()
    assert set(fitted.global_tests["term"]) == {
        "Intercept",
        "C(group, Treatment(reference='base'))",
        "x",
        "z",
    }


def test_formula_metadata_supports_statsmodels_014_and_015_layouts() -> None:
    design_info = SimpleNamespace(term_name_slices={})
    model_014 = SimpleNamespace(data=SimpleNamespace(design_info=design_info))
    model_015 = SimpleNamespace(data=SimpleNamespace(model_spec=design_info))

    assert _formula_design_info(model_014) is design_info
    assert _formula_design_info(model_015) is design_info


def test_partial_scenarios_cannot_overwrite_reviewed_default_outputs() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "analysis/03_inference_member4.py",
            "--scenarios",
            "mature",
        ],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "partial scenario run requires a non-default --output-dir" in completed.stderr
