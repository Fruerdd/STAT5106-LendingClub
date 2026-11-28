# Preprocessing comparison and full model rerun

This is a bounded predictive experiment within the existing Lending Club research question, not the final group report. The frozen resolved-loan cohort and 60/20/20 assignments remain unchanged. Ivy's implementation and original results are retained. No new raw fields, altered outcomes, loan exclusions, test resampling, or subgroup-specific decisions are introduced.

## Code review and fixes

The review covers target polarity, confusion-matrix orientation, maximum-F1 thresholds including tied scores, loan identity/order alignment, train-only transformations and categorical coding, native missing values, class weights, saved score precision and model reloads.

Two reproducible issues were found:

1. The original median imputer can drop an entirely missing training column, breaking the expected numeric feature layout. `keep_empty_features=True` preserves the column. A synthetic regression test reproduces the old failure and checks the fix. This robustness defect is not evidence that the real fitted models used the wrong layout; the full reference refits check that separately.
2. Default CSV formatting of float32 XGBoost scores loses the exact binary score. Reloading the original validation handoff as float64 and using its stored threshold changes one validation decision (F1 0.473969 versus 0.473978). New exports promote scores to float64 and write 17 significant digits; round-trip reads preserve every score. Script 13's future exports use the same precision. Existing historical handoffs and results are preserved, not silently rewritten.

Class weighting can affect calibration and preprocessing can remove information, but these are modelling choices rather than demonstrated metric bugs. No claim is made that either discovered defect explains the overall low F1.

## Full reference reproduction

Refit the existing expanded HGB and XGBoost models from cleaned training data. Reselect XGBoost rounds using the same inner training-month stopping block and require the previous 458 rounds. Check validation/test loan identities, probabilities against the original cached predictions within serialization tolerance, and precision/recall/F1 against independent sklearn implementations. Reference test checks reproduce an already-examined result; they do not select the new pipeline.

## Bounded experiment

Three numerical preparations use the same existing source fields and grade proxy:

| Mode | Numerical treatment | Predictors |
| --- | --- | ---: |
| current | Existing training median/caps/scaling for base numbers; native added-number missingness | 27 |
| native | Preserve numerical NaNs and uncapped engineered values | 27 |
| native_indicators | Native treatment plus 18 fixed numerical missingness indicators | 45 |

Every mode retains train-fitted category maps/rare grouping and unknown-category handling. All indicator columns are defined in advance; future missingness does not select the schema. Grade, subgrade and interest rate are never jointly entered.

For each mode, compare fixed existing XGBoost and HGB settings using the two whole-month forward folds inside training from script 15. Both families use latest-month stopping data inside the fold fit block, then refit that block at the selected round count. HGB uses explicit `X_val`/`y_val`, so recent sklearn is required. This stopping design is shared across the new modes; the separately reproduced reference retains its historical stopping design.

Split each subsequent fold threshold block into earlier whole months for sigmoid calibration and later whole months for threshold selection. Choose the nonempty month boundary closest to half the rows; a first-month row median must not produce an empty earlier block. The still-later fold score block assesses the frozen rule. Calibration fits logistic regression on logit scores using only those out-of-sample training predictions; positive slopes are required.

Measure member decision disagreement and each member's uniquely correct fraction on the training score blocks. An equal-weight calibrated blend is eligible only if both folds have at least 1% disagreement and each member uniquely correct on at least 0.5% of loans. There is only one fixed blend per mode; no weight grid or stacking model is introduced. Ineligible blend diagnostics remain visible but cannot win selection.

Rank at most nine fixed pipelines by mean forward F1, then AP, then lower log loss. Freeze one pipeline before outer validation. Refit its constituent models on the entire training 60%, using rounded median training-fold stopping rounds. Fit final sigmoid maps on pooled out-of-sample calibration scores from the training folds; transfer to the full fit is an approximation and must be checked on validation.

Compare this one pipeline with the fully refitted current XGBoost on validation F1, then AP, then lower log loss. Thresholds maximize validation F1. Freeze the choice and threshold before assessing the selected pipeline on test. Rejected pipelines receive no new test assessment. Carry forward the refitted current model if it wins.

Save the fitted encoder/models/calibrators locally, reload the selected prediction path and require identical predictions. Also require exact score CSV round trips. Use 1,000 conditional paired loan bootstraps for the selected-versus-reference test F1 difference. A retained identical reference has a degenerate zero difference, not zero population uncertainty.

## Reproduction and limits

```bash
python -m pip install -r requirements-preprocessing-experiment.txt
python analysis/16_preprocessing_ensemble_rerun.py
```

macOS needs the existing libomp runtime. Aggregate artifacts are isolated in `outputs/experiments/preprocessing_ensemble_602020`. Fitted bundles and loan predictions stay ignored under artifacts/ and data/cleaned/. No full raw-data cleaning rerun is performed; models start from the frozen cleaned cohort.

All assessments remain exploratory after previous test inspection. Calibration does not itself guarantee better ranking or maximum F1. Follow-up selection, observed-outcome bias, model-selection uncertainty and historical-vintage drift remain limitations. Code and execution are assisted by Codex and acknowledged in the AI-use log. The dashboard is not changed by this experiment.

API references: [sklearn HGB stopping and native missing handling](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
