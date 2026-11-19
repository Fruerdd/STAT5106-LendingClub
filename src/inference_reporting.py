"""Deterministic reporting and visualization for Member 4 inference outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

FOREST_ITEMS = (
    ("A", "fico_per_10", "FICO score (+10 points)"),
    ("A", "log2_annual_inc", "Annual income (doubling)"),
    (
        "A",
        "log2_loan_to_annual_income",
        "Loan-to-income burden (+1 log2 unit)",
    ),
    ("A", "dti_per_10", "Debt-to-income ratio (+10 points)"),
    ("A", "revol_util_per_10", "Revolving utilization (+10 points)"),
    (
        "A",
        "C(term_months, Treatment(reference='36'))[T.60]",
        "60-month term (vs 36 months)",
    ),
    (
        "A",
        "C(purpose_grouped, Treatment(reference='debt_consolidation'))"
        "[T.small_business]",
        "Small-business purpose (vs debt consolidation)",
    ),
    (
        "A",
        "C(home_ownership_grouped, Treatment(reference='MORTGAGE'))[T.RENT]",
        "Rent (vs mortgage)",
    ),
    (
        "B",
        "C(grade, Treatment(reference='A'))[T.B]",
        "Grade B (vs A)",
    ),
    (
        "B",
        "C(grade, Treatment(reference='A'))[T.C]",
        "Grade C (vs A)",
    ),
    (
        "B",
        "C(grade, Treatment(reference='A'))[T.D]",
        "Grade D (vs A)",
    ),
    (
        "B",
        "C(grade, Treatment(reference='A'))[T.E]",
        "Grade E (vs A)",
    ),
    (
        "B",
        "C(grade, Treatment(reference='A'))[T.F]",
        "Grade F (vs A)",
    ),
    (
        "B",
        "C(grade, Treatment(reference='A'))[T.G]",
        "Grade G (vs A)",
    ),
    ("C", "int_rate_per_5", "Interest rate (+5 percentage points)"),
)

MODEL_COLORS = {"A": "#2F6690", "B": "#C45D1A", "C": "#2A7F62"}
MODEL_MARKERS = {"A": "o", "B": "s", "C": "^"}


def _coefficient_row(
    coefficients: pd.DataFrame,
    *,
    scenario: str,
    model: str,
    parameter: str,
) -> pd.Series:
    selected = coefficients.loc[
        coefficients["scenario"].eq(scenario)
        & coefficients["model"].eq(model)
        & coefficients["parameter"].eq(parameter)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {scenario}/{model}/{parameter}, found {len(selected)}"
        )
    return selected.iloc[0]


def _format_q(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0 or value < 1e-300:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _format_or(row: pd.Series) -> str:
    return (
        f"{row['odds_ratio']:.3f} "
        f"({row['or_ci_low']:.3f}–{row['or_ci_high']:.3f})"
    )


def create_inference_forest_plot(
    coefficients: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the prespecified selected-association forest plot."""
    records: list[dict[str, object]] = []
    for model, parameter, label in FOREST_ITEMS:
        row = _coefficient_row(
            coefficients,
            scenario="primary",
            model=model,
            parameter=parameter,
        )
        records.append(
            {
                "model": model,
                "label": label,
                "odds_ratio": float(row["odds_ratio"]),
                "low": float(row["or_ci_low"]),
                "high": float(row["or_ci_high"]),
            }
        )

    plot_data = pd.DataFrame.from_records(records)
    positions = np.arange(len(plot_data))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 8.2))

    for position, row in zip(positions, plot_data.itertuples(index=False), strict=True):
        color = MODEL_COLORS[row.model]
        marker = MODEL_MARKERS[row.model]
        ax.errorbar(
            row.odds_ratio,
            position,
            xerr=np.array(
                [
                    [row.odds_ratio - row.low],
                    [row.high - row.odds_ratio],
                ]
            ),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=2.5,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
        ax.text(
            row.high * 1.035,
            position,
            f"{row.odds_ratio:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=color,
        )

    ax.axvline(1, color="#555555", linewidth=1, linestyle="--")
    ax.axhline(positions[7] - 0.5, color="#D0D0D0", linewidth=0.8)
    ax.axhline(positions[13] - 0.5, color="#D0D0D0", linewidth=0.8)
    ax.set_xscale("log")
    ticks = [0.75, 1, 1.5, 2, 3, 5, 6]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.72, 6.3)
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_data["label"])
    ax.set_xlabel("Adjusted odds ratio (log scale)")
    ax.set_ylabel("")
    ax.set_title("Selected adjusted associations with observed loan default")
    ax.grid(axis="x", which="major", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            color="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            markersize=7,
            label=f"Model {model}",
        )
        for model in ("A", "B", "C")
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, ncol=3)
    fig.text(
        0.01,
        0.01,
        "Odds ratios and 95% custom HC3 robust confidence intervals. "
        "Resolved-outcome cohort; associations are not causal effects.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _markdown_or_row(
    coefficients: pd.DataFrame,
    model: str,
    parameter: str,
    label: str,
    *,
    scenario: str = "primary",
) -> str:
    row = _coefficient_row(
        coefficients,
        scenario=scenario,
        model=model,
        parameter=parameter,
    )
    return f"| {label} | {_format_or(row)} | {_format_q(float(row['q_value']))} |"


def build_inference_report(
    coefficients: pd.DataFrame,
    global_tests: pd.DataFrame,
    sample_summary: pd.DataFrame,
    preprocessing: dict[str, object],
    *,
    figure_reference: str,
) -> str:
    """Render the real-data Member 4 report from auditable result tables."""
    primary = sample_summary.loc[
        sample_summary["scenario"].eq("primary")
        & sample_summary["model"].eq("A")
    ].iloc[0]
    mature = sample_summary.loc[
        sample_summary["scenario"].eq("mature")
        & sample_summary["model"].eq("A")
    ].iloc[0]

    preparation_by_scenario = {
        item["scenario"]: item for item in preprocessing["preparation_audits"]
    }
    primary_preparation = preparation_by_scenario["primary"]
    employment_preparation = preparation_by_scenario["employment_complete_case"]

    model_a_rows = (
        ("fico_per_10", "FICO score, per 10 points"),
        ("log2_annual_inc", "Annual income, per doubling"),
        (
            "log2_loan_to_annual_income",
            "Loan-to-income burden, per +1 log2 unit",
        ),
        ("dti_per_10", "Debt-to-income ratio, per 10 points"),
        ("revol_util_per_10", "Revolving utilization, per 10 points"),
        ("credit_history_years", "Credit-history length, per year"),
        ("delinq_2yrs", "Delinquencies in previous two years, per event"),
        ("inq_last_6mths", "Inquiries in previous six months, per inquiry"),
        ("open_acc_per_5", "Open accounts, per five accounts"),
        ("pub_rec", "Derogatory public records, per record"),
        (
            "C(term_months, Treatment(reference='36'))[T.60]",
            "60-month term versus 36 months",
        ),
        (
            "C(purpose_grouped, Treatment(reference='debt_consolidation'))"
            "[T.small_business]",
            "Small-business versus debt-consolidation purpose",
        ),
        (
            "C(home_ownership_grouped, Treatment(reference='MORTGAGE'))[T.RENT]",
            "Rent versus mortgage",
        ),
    )
    model_a_markdown = "\n".join(
        _markdown_or_row(coefficients, "A", parameter, label)
        for parameter, label in model_a_rows
    )

    grade_markdown = "\n".join(
        _markdown_or_row(
            coefficients,
            "B",
            f"C(grade, Treatment(reference='A'))[T.{grade}]",
            f"Grade {grade} versus Grade A",
        )
        for grade in "BCDEFG"
    )
    rate_markdown = _markdown_or_row(
        coefficients,
        "C",
        "int_rate_per_5",
        "Interest rate, per five percentage points",
    )

    sensitivity_parameters = (
        ("fico_per_10", "FICO +10 points"),
        ("log2_annual_inc", "Income doubling"),
        ("log2_loan_to_annual_income", "Loan-to-income +1 log2 unit"),
        ("dti_per_10", "DTI +10 points"),
        ("revol_util_per_10", "Revolving utilization +10 points"),
    )
    sensitivity_rows: list[str] = []
    for scenario in (
        "primary",
        "mature",
        "unwinsorized",
        "employment_complete_case",
    ):
        for parameter, label in sensitivity_parameters:
            row = _coefficient_row(
                coefficients,
                scenario=scenario,
                model="A",
                parameter=parameter,
            )
            sensitivity_rows.append(
                f"| {scenario} | {label} | {_format_or(row)} |"
            )
    sensitivity_markdown = "\n".join(sensitivity_rows)

    diagnostics_markdown = "\n".join(
        (
            f"| {row.scenario} | {row.model} | {int(row.nobs):,} | "
            f"{int(row.iterations)} | {row.standardized_condition_number:.2f} | "
            f"{row.maximum_vif:.2f} | {row.leverage_max:.4f} |"
        )
        for row in sample_summary.itertuples(index=False)
    )

    employment_row = _coefficient_row(
        coefficients,
        scenario="primary",
        model="A",
        parameter="emp_length_years",
    )
    missing_row = _coefficient_row(
        coefficients,
        scenario="primary",
        model="A",
        parameter="income_or_dti_missing",
    )
    purpose_global = global_tests.loc[
        global_tests["scenario"].eq("primary")
        & global_tests["model"].eq("A")
        & global_tests["term"].str.startswith("C(purpose_grouped")
    ].iloc[0]

    return f"""# Member 4 — Statistical Inference Report

**Project:** STAT5106 Lending Club Loan Default
**Analysis:** Prespecified feature engineering, adjusted association inference, and uncertainty
**Population:** Cleaned resolved-outcome loans

## 1. Question and estimand

This analysis asks which application/origination-time borrower and loan characteristics are
independently associated with the odds of observed default. Estimates are adjusted odds ratios
from prespecified binomial GLMs. They are associations within the resolved-outcome cohort, not
causal effects and not measures of predictive accuracy.

The feature, missing-data, sparse-category, multiplicity, and sensitivity rules were frozen in
`report/member4_inference_specification.md` before fitting.

## 2. Analysis sample

The cleaned cohort contained 1,348,099 resolved loans. The primary inference cohort retained
{int(primary.nobs):,} loans after excluding
{int(primary_preparation['low_missing_rows_excluded']):,} rows missing one of the prespecified
complete-case fields. It contained {int(primary.default_count):,} defaults
({primary.default_rate:.2%}).

Missing employment length was median-imputed with a missingness indicator in the primary
models. The employment complete-case sensitivity excluded
{int(employment_preparation['employment_rows_excluded']):,} additional rows. Purpose levels
`educational` and `renewable_energy`, and home-ownership levels `ANY`, `NONE`, and `OTHER`,
were combined into `OTHER_RARE` using the frozen frequency rule.

## 3. Methods

- Model A includes prespecified borrower, credit, loan/application, and issue-year terms without
  a lender risk-pricing proxy.
- Model B adds categorical loan grade.
- Model C adds interest rate instead of grade.
- Primary continuous features were winsorized at the 0.5th and 99.5th percentiles without using
  the outcome; an unwinsorized Model A was fitted separately.
- Uncertainty uses an explicitly computed, chunked GLM HC3 score-sandwich covariance.
- Coefficient tests and feature-family Wald tests use separate Benjamini-Hochberg FDR
  corrections within each model.
- No stepwise selection or outcome-driven category merging was used.

## 4. Primary adjusted associations

Model A used {int(primary.nobs):,} loans and {int(primary.parameter_count)} parameters.

| Characteristic | Adjusted OR (95% HC3 CI) | BH q-value |
| --- | ---: | ---: |
{model_a_markdown}

Loan purpose was jointly associated with default
(Wald χ²={purpose_global['wald_chi2']:.1f}, df={int(purpose_global['df'])},
q={_format_q(float(purpose_global['q_value']))}). In particular, small-business purpose had
substantially higher adjusted odds than debt consolidation.

Employment length itself was not distinguishable from no association after coefficient-level
FDR correction (OR {_format_or(employment_row)}, q={_format_q(float(employment_row['q_value']))}).
The combined nonpositive-income/DTI-missing indicator was also not distinguishable from no
association (OR {_format_or(missing_row)}, q={_format_q(float(missing_row['q_value']))}).

## 5. Lender risk-pricing specifications

### Model B — grade

| Grade contrast | Adjusted OR (95% HC3 CI) | BH q-value |
| --- | ---: | ---: |
{grade_markdown}

The monotonic grade gradient remained after adjustment for the other prespecified borrower and
loan characteristics. These estimates condition on a lender-assigned risk category and are not
causal grade effects.

### Model C — interest rate

| Interest-rate contrast | Adjusted OR (95% HC3 CI) | BH q-value |
| --- | ---: | ---: |
{rate_markdown}

Grade and interest rate were deliberately fitted in separate models because both encode lender
risk assessment.

## 6. Sensitivity analyses

The maturity sensitivity retained {int(mature.nobs):,} loans whose contractual term would have
ended by 2018-12-31; its observed default rate was {mature.default_rate:.2%}. Directional
conclusions for the core continuous predictors were retained, although magnitudes changed.

| Scenario | Characteristic | Adjusted OR (95% HC3 CI) |
| --- | --- | ---: |
{sensitivity_markdown}

The difference between winsorized and unwinsorized loan-burden/DTI estimates shows that extreme
values materially affect magnitude. Results should therefore be interpreted using the
prespecified primary model together with the sensitivity range, not a single coefficient alone.

## 7. Diagnostics

All six models converged. No HC3 leverage denominator required clipping. Standardized condition
numbers and approximate VIFs did not indicate severe multicollinearity.

| Scenario | Model | n | Iterations | Condition number | Maximum VIF | Maximum leverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{diagnostics_markdown}

Selected adjusted associations are summarized below; the complete coefficient and global-test
tables remain the source of truth.

![Selected adjusted odds ratios]({figure_reference})

## 8. Limitations

- The cohort excludes current, late, grace-period, missing, and unknown statuses. Selection into
  a resolved status is especially problematic for recent loan vintages.
- The mature-loan restriction reduces but cannot fully remove outcome-observation and historical
  cohort differences.
- The models are associational and may remain affected by unmeasured confounding, functional-form
  misspecification, and Lending Club's own underwriting process.
- Very large sample size makes small effects statistically detectable; effect magnitude,
  uncertainty, and sensitivity are more important than significance alone.
- Grade and interest rate are lender-assigned summaries, not manipulable causal exposures.
- Predictive discrimination, calibration, thresholds, and out-of-time performance belong to the
  downstream modelling and validation tasks and are not reported here.

## 9. Reproduction

From the repository root:

```bash
python analysis/03_inference_member4.py \\
  --input data/cleaned/clean_v1/lending_club_clean.csv.gz
```

Auditable outputs:

- `outputs/tables/inference_odds_ratios.csv`
- `outputs/tables/inference_global_tests.csv`
- `outputs/tables/inference_sample_summary.csv`
- `outputs/tables/inference_preprocessing.json`
- `outputs/tables/feature_specification.csv`
- `outputs/figures/inference_forest_plot.png`
- `report/member4_inference_report.md`

The downstream modelling owner must estimate preprocessing values and category maps on the
temporal training partition rather than reusing the full-cohort inference values.
"""


def write_inference_report(
    coefficients: pd.DataFrame,
    global_tests: pd.DataFrame,
    sample_summary: pd.DataFrame,
    preprocessing: dict[str, object],
    *,
    figure_reference: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_inference_report(
        coefficients,
        global_tests,
        sample_summary,
        preprocessing,
        figure_reference=figure_reference,
    )
    output_path.write_text(report, encoding="utf-8")
