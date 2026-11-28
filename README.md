# STAT5106 Lending Club Loan Default

A local Python/PyCharm group project for predicting Lending Club loan default.

**Research question:** Which borrower and loan characteristics significantly predict loan
default, and how accurately can default be predicted?

The cleaning pipeline restricts the main cohort to resolved outcomes: Fully Paid (`0`) and
Charged Off or Default (`1`). Current, late, grace-period, missing, and unknown outcomes are
excluded. Only application/origination-time fields are read; post-outcome fields are excluded
to reduce leakage. Statistical significance and predictive accuracy must be assessed separately.

## Prerequisites

- Python 3.12 recommended (a 64-bit Python 3.10+ environment should also work).
- PyCharm for local development; Google Colab is not used.
- Git and, for maintainers, the authenticated GitHub CLI (`gh`).
- The frozen course dataset, supplied separately rather than through GitHub.

## PyCharm and virtual-environment setup

1. Open the `STAT5106-LendingClub` directory as a PyCharm project.
2. Create a local interpreter if `.venv` does not already exist:

   macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Windows Command Prompt:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   One-line Windows PowerShell dependency setup after creating `.venv`:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. In PyCharm, open **Settings/Preferences → Project → Python Interpreter**, then select the
   existing interpreter at `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe`
   (Windows).
4. Set run-configuration working directories to the repository root. Do not store personal
   absolute paths in source files or shared run configurations.

## Exact local dataset placement

Put the frozen file manually at:

```text
data/raw/accepted_2007_to_2018Q4.csv.gz
```

Use the same dataset version on every machine. The raw file must never be edited, committed,
or pushed. The raw and cleaned data directories are ignored even though the GitHub repository
is private. Small CSV result tables under `outputs/tables` remain trackable because there is
deliberately no global `*.csv` ignore rule. See `data/raw/README.md` and `CONTRIBUTING.md`.

## Clean the data and run tests

Run all commands from the repository root. Install the dependencies, then run:

```bash
python src/lending_club_data_cleaning.py --input data/raw/accepted_2007_to_2018Q4.csv.gz --output-dir data/cleaned/clean_v1
```

This reads the gzip file directly with `usecols`, preserves the original `loan_status`, and
writes:

- `data/cleaned/clean_v1/lending_club_clean.csv.gz`
- `data/cleaned/clean_v1/cleaning_report.json`

The report records row reconciliation, status exclusions, duplicates, invalid values,
missingness, outcomes, default rate, and issue-date range. Impossible values become missing;
legitimate extreme observations are retained. No full-dataset median imputation occurs.
Modelling imputers must be fitted on training data only.

Run the synthetic tests (no real dataset is needed):

```bash
python -m pytest
```

## Team member assignments

The files under `analysis/` are member-owned analysis entry points. Each implementation must use
real project data and the requirements below. Do not create synthetic findings, placeholder
metrics, or fake figures.

### Requirements for everyone

- Use `src/paths.py` for common directories rather than personal absolute paths.
- Provide a `main()` entry point and clear command-line arguments where inputs can vary.
- Check required inputs and fail with a useful message if they are unavailable.
- Use only application/origination-time predictors. Never use post-outcome or settlement fields.
- Do not use `loan_status` as a predictor: it defines the outcome and is retained only for audit.
- Document every feature, filtering, or cohort change and notify downstream owners.
- Save real result tables to `outputs/tables` and real figures to `outputs/figures`.
- Record assumptions and limitations in the report. Test code before submitting it for review.

Complete the work in this order: **Member 1 → Member 2 → Member 3 → Member 4 (Pavel) →
Member 5 → Member 6**. Later tasks depend on earlier outputs and decisions.

### Member 1 — cleaning validation and data audit

**File:** `analysis/01_cleaning_member2.py`

**Required inputs:** the frozen raw dataset, the cleaned cohort, and `cleaning_report.json`.

**What must be done:**

1. Independently check raw/final row counts, status exclusions, duplicate removal, outcome
   counts, default rate, and issue-date range against the cleaning report.
2. Confirm Fully Paid is `0`, Charged Off/Default is `1`, and unresolved outcomes are absent.
   Verify that original `loan_status` remains available for auditing.
3. Audit the input and cleaned schemas for post-outcome leakage. Check that only expected
   origination-time fields and approved derived variables are present.
4. Verify parsing of term, percentages, employment length, issue dates, earliest credit line,
   credit-history months, and FICO midpoint using representative real observations.
5. Check impossible-value handling and confirm that legitimate extreme observations were not
   automatically removed. Reconcile missingness counts and indicators with the report.
6. Produce a data dictionary containing every cleaned field, its meaning, unit, source or
   derivation, and whether it is permitted as a predictor.

**Outputs/handoff:** cleaning-validation summary, missingness table, leakage audit, and data
dictionary in `outputs/tables`. Report any cleaning defects before EDA or modelling begins.

### Member 2 — visualization and exploratory analysis

**File:** `analysis/02_visualization_member3.py`

**Required input:** Member 1's validated cleaned cohort.

**What must be done:**

1. Describe cohort size, outcome balance, issue-time coverage, and important missingness.
2. Examine distributions of borrower and loan characteristics such as loan amount, income,
   interest rate, DTI, employment length, credit history, FICO, utilization, and inquiries.
3. Compare default rates across grade, purpose, home ownership, verification status, geography,
   and other approved categorical predictors. Include sample sizes and avoid overinterpreting
   small groups.
4. Explore numerical relationships with default using appropriate plots, transformations, and
   summaries. Keep legitimate extreme values visible or clearly explain any plot-only truncation.
5. Examine default and feature patterns over issue time to identify drift that could affect
   temporal modelling. Investigate unusual patterns rather than silently removing observations.
6. Label axes, units, cohorts, and sample sizes. Figures must come from the real cleaned data.

**Outputs/handoff:** exploratory tables and clearly named figures under `outputs/`, plus a short
written summary of patterns, possible confounders, data-quality concerns, and features that
Member 3 and Pavel should consider. Exploratory associations are not proof of significance.

### Member 3 — statistical inference and feature engineering

**File:** `analysis/03_inference_member4.py`

**Required inputs:** the validated cleaned cohort and Member 2's exploratory findings.

**What must be done:**

1. Define the inferential population, estimands, hypotheses, outcome definition, and candidate
   confounders before interpreting statistical tests. Keep inference goals distinct from pure
   predictive modelling.
2. Engineer defensible features such as loan-to-income ratios or credit-history measures where
   useful. Document formulas, units, denominator handling, and missing values. Do not create
   features from post-outcome information.
3. Select appropriate unadjusted analyses for categorical and numerical predictors, checking
   assumptions and using suitable alternatives when assumptions are not credible.
4. Fit an adjusted statistical model, such as logistic regression with statsmodels, to estimate
   associations between default and borrower/loan characteristics while accounting for agreed
   confounders.
5. Report effect estimates with confidence intervals, not p-values alone. For logistic models,
   provide interpretable odds ratios and clearly state reference categories and units.
6. Check model specification, influential observations, sparse categories, separation,
   multicollinearity, and residual/diagnostic concerns. Address multiple comparisons where
   relevant and distinguish exploratory from confirmatory findings.
7. Ensure any feature selection or fitted transformation intended for prediction is learned on
   training data only. Do not use the held-out test cohort to choose features.

**Outputs/handoff:** inference tables, diagnostics, feature definitions, and a written answer to
which characteristics are significantly associated with default and under what assumptions.
Give Pavel the approved feature list and document any transformations required by the model.

### Member 4 (Pavel) — predictive modelling and interpretation

**File:** `analysis/04_modelling_pavel.py`

**Required inputs:** the validated cleaned cohort and Member 3's approved features.

**What must be done:**

1. Create an out-of-time split using `issue_date`: train on earlier loans and test on later
   loans. State the cutoff and cohort sizes. Do not substitute a random split for the primary
   evaluation.
2. Define the target as `default`. Exclude `loan_status`, identifiers, post-outcome fields, and
   any other leakage or non-predictive audit columns from predictors.
3. Build reproducible preprocessing pipelines. Fit numerical imputation/scaling and categorical
   imputation/encoding on training data only. Keep test data untouched until final evaluation.
4. Fit a logistic-regression baseline and clearly justified comparison models. Pre-specify model
   candidates and hyperparameter/tuning procedures, use training/validation data for selection,
   and set random seeds where applicable.
5. Consider class imbalance without leaking test information. Explain any class weighting,
   resampling, regularization, or threshold choice and where it was fitted.
6. Compare models using appropriate discrimination and calibration measures, but do not choose
   the final model solely from test-set performance.
7. Interpret the final logistic model using coefficients and odds ratios with clear units and
   reference categories. Explain that predictive importance does not establish causation.
8. Save a real out-of-time predictions table for Member 5 containing at least `id`,
   `issue_date`, `default`, and `predicted_probability`, plus a model identifier if multiple
   models are evaluated. Save model-comparison and interpretation tables separately.

**Outputs/handoff:** split definition, model specifications, comparison table, final-model
interpretation, and `outputs/tables/out_of_time_predictions.csv` or a documented equivalent.
Large fitted model artefacts must remain local and ignored by Git.

### Member 5 (Ivy) — validation, robustness, and limitations

**File:** `analysis/05_validation_member5.py`

**Required inputs:** the cleaned test cohort and Pavel's real out-of-time predictions.

**What must be done:**

1. Verify prediction row identity, uniqueness, outcome consistency, date range, and probability
   validity against the held-out cohort. Ensure no training observations are included.
2. Calculate ROC-AUC and PR-AUC with confidence intervals. Include the outcome prevalence as the
   relevant baseline for PR-AUC and explain what each metric does and does not measure.
3. Assess calibration using a calibration plot and suitable quantitative measures such as Brier
   score and calibration intercept/slope. Distinguish discrimination from calibration.
4. Analyze classification thresholds using prespecified costs or operating goals. Select any
   threshold on training/validation data only, then report its untouched-test performance,
   including confusion-matrix-derived measures relevant to the project.
5. Use a documented bootstrap procedure on out-of-time predictions to obtain uncertainty
   intervals. State the number of resamples, random seed, unit of resampling, and method.
6. Evaluate performance across meaningful subgroups, such as grade or other agreed borrower and
   loan groups. Report subgroup sizes, outcome counts, uncertainty, and missing-category results;
   do not draw conclusions from inadequate samples.
7. Conduct robustness checks for reasonable alternative splits, feature specifications, class
   weighting, or modelling assumptions without cherry-picking favorable results.
8. Document temporal drift, selection bias, unresolved-outcome exclusion, missing data,
   subgroup limitations, and limits on causal interpretation and generalizability.

**Outputs/handoff:** real validation tables, ROC/PR/calibration/threshold figures, bootstrap
intervals, subgroup and robustness summaries, and a limitations section for the final report.

### Member 6 — end-to-end reproducibility

**File:** `analysis/06_reproducibility_member6.py`

**Required inputs:** the completed team scripts, frozen raw dataset, cleaning report, and real
analysis outputs.

**What must be done:**

1. Test the fresh-clone procedure below on a separate local checkout with a newly created virtual
   environment and the same frozen dataset. Record the dataset checksum and software versions.
2. Run the synthetic tests and cleaning command, then independently compare row counts, status
   exclusions, duplicates, outcome counts, default rate, date range, and cleaned-schema details.
3. Run each member's implemented script in the required order. Confirm that every input is
   available, every script has a clear entry point, and failures are informative.
4. Check that repeated runs with the same inputs and seeds produce consistent tables, metrics,
   and figures. Document acceptable numerical tolerances where exact binary equality is not
   expected.
5. Verify that preprocessing, feature selection, model selection, and threshold selection do not
   use held-out test data and that no leakage field enters a predictor matrix.
6. Confirm that no raw data, cleaned data, credentials, personal paths, or large model artefacts
   are tracked. Check the final Git diff and working-tree status.
7. Record environment details, commands actually run, discrepancies, unresolved failures, and the
   final reproduction result. Never claim successful reproduction if any required step was not
   run or failed.

**Outputs/handoff:** a reproducibility report and a small consistency-check table. Return any
problems to the responsible owner before final submission.

## Implementation priorities and optional dataset extension

The accepted-loan dataset is sufficient for the current research question and for every component
of the project rubric. Adding another dataset is not required. The highest-priority work beyond
the current predictive pipeline is:

1. Complete out-of-time validation with bootstrap confidence intervals, calibration intercept and
   slope, ROC/PR and calibration figures, and prevalence-aware interpretation of PR-AUC.
2. Select classification thresholds on training/validation predictions only, using documented
   costs or operating goals, and report untouched-test confusion matrices and derived measures.
3. Evaluate subgroup performance and reasonable alternative splits, feature specifications, and
   class-weighting assumptions. Report inadequate subgroup samples rather than overinterpreting
   them.
4. Integrate the inference, visualization, modelling, and validation findings into a coherent final
   report. Include effect sizes and uncertainty, distinguish inference from prediction, and avoid
   causal claims.
5. Complete a clean end-to-end reproduction, resolve failing tests and interface mismatches, and
   verify that all generated tables and figures come from the frozen real dataset.
6. Prepare concise slides covering the research question, data, methods, key quantitative results,
   and limitations within the 15-minute limit. Keep the functional AI-use disclosure complete.

The final report and presentation must state that conclusions apply only to accepted loans with
resolved outcomes. Recent vintages have unequal outcome follow-up, and results cannot be
generalized to rejected applicants or unresolved loans without additional assumptions and data.

### Optional rejected-applications analysis

`rejected_2007_to_2018Q4.csv` is not required by the guidelines or grading rubric. It has no
observed `loan_status` or `default` outcome, so it must not be concatenated with accepted loans for
default training, and rejected applicants must never be assigned synthetic default labels.

If the team deliberately adds a second research question about application approval or selection,
clean the rejected file separately, harmonize only defensible common application-time fields, and
analyse acceptance/selection separately from default. Document the additional cohort, outputs, and
limitations. This optional extension must not alter the accepted-loan default target, temporal
splits, or existing out-of-time predictions.

## Directory structure

```text
STAT5106-LendingClub/
├── data/
│   ├── raw/                 # Local frozen CSV.GZ; ignored
│   └── cleaned/             # Generated local cohorts/reports; ignored
├── src/                     # Shared paths and cleaning implementation
├── analysis/                # Ordered member-owned analysis entry points
├── tests/                   # Synthetic, dataset-independent tests
├── outputs/
│   ├── figures/             # Real generated figures
│   └── tables/              # Small real result/prediction tables
├── report/ai_use_log.md     # Functional AI-use disclosure
├── slides/                  # Presentation materials
├── .github/workflows/       # Dataset-independent CI tests
├── requirements.txt
├── README.md
└── CONTRIBUTING.md
```

`src/paths.py` derives all common paths from `__file__`, so shared code does not depend on a
machine-specific checkout location. Run implemented scripts from the repository root.

## Fresh-clone reproducibility procedure

1. Clone the private repository and enter its root:

   ```bash
   git clone <PRIVATE_REPOSITORY_URL>
   cd STAT5106-LendingClub
   ```

2. Create and select `.venv`, then install `requirements.txt` as described above.
3. Manually copy the team's same frozen dataset to
   `data/raw/accepted_2007_to_2018Q4.csv.gz`; it will not be downloaded by Git or CI.
4. Run `python -m pytest` and require all synthetic tests to pass.
5. Run the exact cleaning command above. Compare `cleaning_report.json` with the team's reviewed
   baseline, including row counts, exclusions, outcome counts, default rate, and date range.
6. Once each member has implemented their assigned placeholder, run the six analysis files in
   order. Resolve missing real inputs instead of substituting synthetic results.
7. Have Member 6 perform the reproducibility and consistency checks described above. Record any
   environment, dataset, feature, or output discrepancies.
8. Confirm that generated data remains local with `git status --short` and
   `git check-ignore -v data/raw/accepted_2007_to_2018Q4.csv.gz`.

The synthetic tests and GitHub Actions do not require, access, or download the private dataset.
A real-data run must still be performed and checked by the team before results are reported.

### Member 6 reproduction command

```bash
python analysis/06_reproducibility_member6.py --run
```

This creates a separate local clone with the current source changes, installs
`requirements-reproducibility.txt` in a new virtual environment, links the frozen raw file,
and runs the tests and analyses in order. Generated results stay in the temporary checkout.
The approved `feature_specification.csv` is a frozen input contract and is preserved.

Results are recorded in `report/member6_reproducibility_report.md`,
`outputs/tables/reproducibility_checks.csv`, and `outputs/tables/reproducibility_audit.json`.
Counts and categories must agree exactly; numerical CSV results use relative tolerance
`1e-6` and absolute tolerance `1e-10`. EDA receives an additional repeat with byte-for-byte
checks. A second complete pipeline run and an external GitHub clone are not claimed.

Ivy's original validation implementation, report, and outputs are retained. Her verification
tests use the current function arguments and unpack its prediction-table/summary return value.
The Member 6 note is separate from the final group report.

## Interactive research dashboard

[Open the private dashboard](https://lending-club-research-dashboard.compassionateblimp128.chatgpt.site).

The `dashboard/` folder displays the verified aggregate results across seven analysis views, including Ivy's validation and Member 6 reproduction. Open `dashboard/dist/index.html` offline, or follow `dashboard/README.md` to serve and refresh it.

Run `.venv/bin/python dashboard/build_data.py` to rebuild its aggregate data bundle and `node dashboard/check.mjs` for source-level interaction checks. The dashboard does not refit models or alter the analysis outputs.

## Exploratory predictive improvement

The follow-up scripts `analysis/07_predictive_improvement.py` through `analysis/11_validate_feature_extension.py` compare bounded tuning changes and an audited feature extension. See `report/predictive_improvement_experiments.md` for selection rules, results, uncertainty and test-reuse limitations. Results are separate under `outputs/experiments`; original primary outputs and Ivy's report remain intact. The feature extension requires the existing cleaned data and original prediction outputs. Run the five scripts in numerical order.

The [Model improvement dashboard view](https://lending-club-research-dashboard.compassionateblimp128.chatgpt.site/#improvement) presents the extension separately, including its reused-test limitation, calibration and paired threshold comparisons.

## Optional XGBoost benchmark

Install `requirements-xgboost.txt` in the project environment. On macOS, XGBoost also needs the OpenMP runtime (`brew install libomp`). Run `python analysis/12_xgboost_benchmark.py` after the feature-extension handoffs have been generated by script 11.

The benchmark uses the same 27 expanded grade predictors. Training-only early stopping sets the round count; early-2016 thresholds are frozen before late-2016 F1 candidate comparison. The selected XGBoost model is refitted before 2017, its final threshold is chosen on 2017, and 2018 is explicitly treated as a reused exploratory test. Results are isolated in `outputs/experiments/xgboost_benchmark`; model artifacts and loan-level prediction handoffs stay ignored locally. The original requirements and results are unchanged.

The completed benchmark achieved F1 0.3809 versus 0.3795 for the expanded model; the paired difference interval includes zero. Precision increased and recall decreased, while average precision and calibration worsened. The expanded model remains preferred. See [the benchmark note](report/xgboost_benchmark_report.md) and the dashboard's Model improvement view for details.

## Chronological 60/20/20 experiment

Run `python analysis/13_split_602020.py` with the optional XGBoost environment to refit the expanded HGB and XGBoost models on the earliest 60% of loans, select thresholds and a model on the next 20%, and assess the latest 20%. Assignments use issue month followed by a fixed outcome-independent ID hash; exact proportions can split a boundary month. All preprocessing is training-fitted, and XGBoost stopping uses an inner training block.

Results go to `outputs/experiments/split_602020` and [a separate experiment note](report/split_602020_experiment.md). This is an exploratory repartition of previously examined data. It preserves earlier study/Ivy outputs; the existing dashboard continues to show the earlier split. Scores across different test cohorts are not directly comparable.

The 60/20/20 assignments are now frozen for the remainder of the project in `outputs/experiments/split_602020/frozen_split_contract.json`. All future predictive experiments must call `frozen_602020` with this contract and preserve the same cohort; a mismatch stops the run. Do not regenerate or overwrite the contract to accommodate a changed split. Use training-only inner folds for fitting/tuning, validation for final operating rules, and test only for a frozen assessment. Previously examined test results remain exploratory.

Earlier predictive scripts and Ivy's validation are retained as historical results/reproduction entry points, not as the split definition for new modelling. The frozen contract above governs new work.

Script 13 also exports local ignored validation predictions. Run `python analysis/14_validation_operating_options.py` to compare maximum-F1, precision-target and recall-target thresholds on validation only. It verifies the frozen validation identity hash and does not open test predictions. Constrained rules require at least 1,000 flagged validation loans; validation precision targets are not guarantees on future data.

## Bounded improvement within project scope

`python analysis/15_scoped_predictive_tuning.py` audits the existing cleaned feature sources and compares six fixed XGBoost settings on two chronological forward folds inside the frozen training 60%. It does not add raw fields or change the research question, target or inference analysis. One training-selected candidate is compared with the current model on validation; the chosen model/threshold is frozen before test scoring. The current model is retained if it wins validation. See [the registered protocol](report/scoped_tuning_protocol.md); aggregate results are separate under `outputs/experiments/scoped_tuning_602020`.

The completed run retained the current model: validation F1 was 0.47398 versus 0.47356 for the training-selected candidate. Its test F1 therefore remains 0.45107. See [the completed results and diagnostics](report/scoped_tuning_results.md). All 49 tests passed; the locked split and Ivy's original work were preserved.

## Preprocessing comparison and full model refit

Install `requirements-preprocessing-experiment.txt`, then run `python analysis/16_preprocessing_ensemble_rerun.py`. This independently refits both current reference models, checks cached-prediction and metric parity, and compares three numerical preparations for fixed XGBoost/HGB settings inside training. A fixed equal calibrated blend is eligible only if both training folds demonstrate complementary errors. One training-selected pipeline is compared with the refitted current model on validation before any new selected-model test assessment.

The run verifies the same frozen split contract. Its [protocol and code-review details](report/preprocessing_ensemble_protocol.md) explain the all-missing numeric-column fix, full-precision prediction exports, training-only calibration and test-reuse limitations. Aggregate results are isolated under `outputs/experiments/preprocessing_ensemble_602020`; fitted bundles and loan predictions remain ignored locally. This experiment does not generate the final group report or update the dashboard.

The completed full refit reproduced both references. The native calibrated blend improved validation precision/AP/Brier score but had F1 0.473877 versus 0.473978 for current XGBoost, so the current model was retained; its test F1 remains 0.451069. All 56 tests and exact fitted-bundle/score-export reload checks passed. See [the completed results](report/preprocessing_ensemble_results.md).

## Joint accuracy and default F1 experiment

On the frozen 60/20/20 split, a single XGBoost challenger used ordered loan subgrade and three ratios from existing origination fields. Its validation-selected rule increased exploratory test accuracy from 63.80% to 64.72% and default F1 from 0.45107 to 0.45152. The paired F1 interval includes zero. The [error analysis and reproduction instructions](report/joint_accuracy_f1.md) give the precision and recall trade-off, model location, and same-loan comparison.

The [follow-up default F1 experiment](report/temporal_f1.md) compares four alternatives inside the frozen training set. Averaging the revised XGBoost with a model trained with recency weights and four additional origination-derived predictors raised test F1 to 0.45219 and accuracy to 65.09%. The paired F1 interval again includes zero. Run `.venv/bin/python analysis/18_temporal_f1.py` after experiment 17 to reproduce the comparison.

The [precision-focused follow-up](report/precision_focus.md) trades some recall for fewer false alarms on the same frozen split. The cutoff chosen on validation avoided 10,934 test false alarms and missed 3,557 additional defaults. Use `.venv/bin/python analysis/19_precision_focus.py` after experiment 18 to reproduce it.

Run `.venv/bin/python analysis/21_majority_baseline.py` after experiment 20 to reproduce the [majority-class comparison](outputs/experiments/majority_baseline_602020/comparison.csv). Fully paid was the most frequent training outcome. Predicting it for every test loan gives 78.21% accuracy and zero default recall or F1. Default precision is blank because the baseline flags no loans. The maximum-F1 model gives 65.09% accuracy, 66.12% default recall, and 0.45219 default F1 on the same 269,620 test loans.
