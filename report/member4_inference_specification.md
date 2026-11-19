# Member 4 — Statistical Inference and Feature Contract

**Status:** Frozen before model fitting
**Dataset:** `data/cleaned/clean_v1/lending_club_clean.csv.gz`
**Outcome:** `default` (`0` = Fully Paid; `1` = Charged Off or Default)
**Primary population:** Loans in the cleaned resolved-outcome cohort

## 1. Scope and separation from prediction

This part estimates adjusted associations between information available at application or
origination and the odds of observed default. The estimates are not causal effects and apply
only to the resolved-outcome cohort.

Member 4 owns:

- feature definitions used for inference;
- estimands and model specifications;
- coefficient uncertainty and multiple-testing treatment;
- sensitivity analyses and an interpretable result table;
- a feature contract for the downstream modelling owner.

Member 4 does not own:

- the cleaning outcome definition or leakage allowlist;
- exploratory feature discovery beyond the reviewed EDA;
- Pavel's temporal train/test split, predictive model selection, or test-set preprocessing;
- ROC-AUC, PR-AUC, calibration, thresholds, or other out-of-time validation metrics.

No feature may be removed from the downstream candidate set solely because its full-cohort
inference p-value is large. Pavel must fit imputers, caps, scalers, encoders, and rare-level
mappings on training data only.

## 2. Primary estimands

For each prespecified continuous feature, the estimand is the adjusted odds ratio for the
meaningful increment listed in `outputs/tables/feature_specification.csv`, holding the other
terms in that model fixed.

For each categorical feature, the estimands are:

1. adjusted odds ratios for each level relative to its prespecified reference level; and
2. a joint Wald test for the feature as a whole.

All reported estimates must include the coefficient, robust standard error, odds ratio, 95%
confidence interval, raw p-value, and multiplicity-adjusted q-value. Statistical significance
must not replace discussion of effect magnitude and uncertainty.

## 3. Cohort and time handling

The primary association analysis uses the full cleaned resolved-outcome cohort and includes
categorical issue-year controls. This is an association analysis, not an out-of-time prediction
evaluation.

The following limitations must accompany the primary results:

- unresolved statuses were excluded by construction;
- recent vintages have less opportunity to reach a resolved adverse outcome;
- the sharp fall in observed 2018 default rates is not evidence of a causal calendar-time
  improvement.

A prespecified maturity sensitivity cohort will retain a loan only when its contractual term
would have ended by `2018-12-31`:

```text
issue_date + term_months <= 2018-12-31
```

This deterministic cutoff is conservative and is derived from the frozen dataset endpoint, not
from observed default rates.

## 4. Feature families

### 4.1 Borrower and credit characteristics

- FICO midpoint, reported per 10 points;
- log2 annual income, interpreted per income doubling;
- debt-to-income ratio, reported per 10 percentage-point units;
- revolving utilization, reported per 10 percentage-point units;
- credit-history length in years;
- employment length in years;
- number of delinquencies in the previous two years;
- inquiries in the previous six months;
- number of open accounts, reported per five accounts;
- number of derogatory public records.

### 4.2 Loan and application characteristics

- log2 of one plus loan-to-annual-income ratio;
- loan term, with 36 months as the reference;
- loan purpose, with debt consolidation as the reference;
- home ownership, with mortgage as the reference;
- verification status, with Not Verified as the reference;
- application type, with Individual as the reference.

### 4.3 Calendar-time control

- issue year as a categorical control, with 2015 as the reference year.

### 4.4 Lender risk-pricing proxies

- Model B adds loan grade, with Grade A as the reference;
- Model C adds interest rate, reported per five percentage points;
- grade, sub-grade, and interest rate must never appear together in the same primary model.

Grade and interest rate are assigned by the lender and encode information from multiple
application characteristics. Changes between Models A, B, and C must therefore be interpreted
as changes in conditional association, not as mediation or causal effects.

## 5. Excluded and sensitivity-only fields

The following are excluded from all primary predictor sets:

- `id` and `loan_status`;
- raw date strings and the original FICO bounds;
- post-outcome leakage fields defined by the cleaning module;
- `installment`, because it is jointly determined by amount, rate, and term;
- `sub_grade`, because grade or interest rate supplies the prespecified lender-risk proxy;
- `mths_since_last_delinq` and `mths_since_last_record` as ordinary continuous predictors,
  because their missingness is largely structural and exceeds approximately 50% and 83%.

Binary indicators for whether a prior delinquency/public-record date exists may be evaluated
only in a labelled sensitivity model. They may not replace the primary specification after
results are viewed.

## 6. Transformations and extreme values

Transformations are fixed before fitting:

- `fico_per_10 = fico_score / 10`;
- `log2_annual_inc = log2(annual_inc)` when annual income is positive;
- `log2_loan_to_annual_income = log2(1 + loan_amnt / annual_inc)` when annual income is
  positive;
- `dti_per_10 = dti / 10`;
- `revol_util_per_10 = revol_util / 10`;
- `credit_history_years = credit_history_months / 12`;
- `open_acc_per_5 = open_acc / 5`;
- `int_rate_per_5 = int_rate / 5`.

For the primary inference models, the following engineered continuous predictors are winsorized
at the 0.5th and 99.5th percentiles of the inference cohort without using the outcome:
`log2_annual_inc`, `log2_loan_to_annual_income`, `dti_per_10`, `revol_util_per_10`,
`credit_history_years`, `delinq_2yrs`, `inq_last_6mths`, `open_acc_per_5`, and `pub_rec`.
FICO, employment length, and interest rate retain their observed cleaned ranges. An otherwise
identical unwinsorized model is required as a sensitivity analysis. For predictive modelling,
the percentile cutoffs must instead be learned from the training partition only.

## 7. Missing-data contract

- Missing DTI is retained using an inference-cohort median. Nonpositive or missing annual income
  makes both income transforms undefined, so those transforms are also replaced with
  inference-cohort medians.
- The two conditions use one combined `income_or_dti_missing` indicator: 361 of the 365
  nonpositive-or-missing-income rows also have missing DTI, leaving the separate indicators
  nearly collinear. The frozen full cohort has 380 rows in their union.
- A scenario includes a binary missingness indicator only when both levels contain at least 20
  observations. Otherwise the indicator is recorded as not estimable and omitted for that
  scenario without inspecting outcomes; the mature cohort has only two flagged rows.
- Rows missing one of the remaining very-low-missingness primary fields (`revol_util`,
  `credit_history_months`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, or `pub_rec`) are
  excluded from the primary inference complete-case cohort. The excluded count must be reported.
- Missing employment length is retained using a training/inference-cohort median plus the
  existing `emp_length_years_was_missing` indicator. A complete-case employment sensitivity
  model is required.
- No full-dataset imputation value may be passed to Pavel. Prediction preprocessing must estimate
  all imputation values on the temporal training set.

## 8. Sparse-category contract

Rare-level handling is based on frequency, never on observed default rate:

- categorical levels with fewer than 1,000 observations in the relevant fitting cohort map to
  `OTHER_RARE`;
- in the frozen full cohort this combines `educational` and `renewable_energy` for purpose;
- in the frozen full cohort this combines `NONE`, `OTHER`, and `ANY` for home ownership;
- unknown future/test levels also map to `OTHER_RARE`;
- grade levels A–G remain separate;
- continuous FICO is used instead of sparse EDA bins.

Pavel must learn the training-set rare-level map and apply that same map to validation/test data.

## 9. Prespecified models

### Model A — primary borrower/application association model

Includes all borrower, credit, loan/application, missingness, and issue-year terms above, but no
grade, sub-grade, or interest rate.

### Model B — grade-adjusted model

Adds categorical grade to Model A.

### Model C — interest-rate-adjusted model

Adds continuous interest rate to Model A instead of grade.

No stepwise selection and no unplanned interaction terms are permitted in the primary analysis.
An additional term requires a documented amendment and team review before its result is viewed.

## 10. Uncertainty, multiplicity, and diagnostics

- Fit binomial GLMs/logistic regressions with HC3 robust covariance. The project computes the
  GLM score-sandwich covariance explicitly from score residuals and weighted leverage;
  `statsmodels` 0.14.x must not be called with `GLM.fit(cov_type="HC3")` because that path
  computes HC0 while retaining an HC3 label.
- Report 95% confidence intervals.
- Within each model, apply Benjamini-Hochberg FDR at 5% across displayed non-intercept
  coefficient tests.
- For feature-family joint Wald tests, apply a separate Benjamini-Hochberg correction across
  feature families within each model.
- Check convergence, separation, rank deficiency, variance inflation, and design-matrix
  condition number.
- Report category counts and the final analysis row count for every model.

The primary conclusion must be based on adjusted effect sizes, confidence intervals, robustness
across the prespecified models, and domain interpretation—not p-values alone.

## 11. Downstream handoff

`outputs/tables/feature_specification.csv` is the machine-readable handoff. It fixes source
columns, transformations, units, references, missing handling, model membership, and downstream
notes.

The modelling owner may compare predictive candidates using the temporal training/validation
process, but must not silently change the outcome, feature definitions, transformations, rare
category rule, or leakage policy. Any contract change requires a reviewed update to this
document and the CSV.
