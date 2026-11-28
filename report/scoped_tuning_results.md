# Scoped predictive tuning: completed results

The bounded experiment did not improve validation F1. The current XGBoost model is retained. This is an experiment note, not the final group report.

## Work completed

- Audited all 41 cleaned columns, source permissions and training/validation missingness. Added no new raw sources. Only the existing documented predictive extension is used; primary inference permissions are unchanged.
- Verified the frozen 60/20/20 hash contract: 808,859 training, 269,620 validation and 269,620 test loans. No loan was moved, removed or resampled in validation/test.
- Completed six fixed XGBoost specifications on two chronological folds entirely within training. Preprocessing, category maps, stopping and tuning thresholds were learned in their designated training blocks.
- Compared depth, class weighting, regularization, row/column sampling and the already-documented alternative subgrade representation. Never combined lender grade, subgrade and interest rate in the predictor matrix.
- Saved validation-only operating-rule alternatives and diagnostics by issue year, grade, income verification and structural credit-history missingness.

The detailed methods were registered in `scoped_tuning_protocol.md` and the experiment manifest before search. The course Week 9 workflow and research scope remain unchanged. No external data, new outcome definition, model deployment, inference changes or final group report was introduced.

## Selection result

The training-only search selected `grade_depth5_unweighted`, with mean forward F1 0.45090 and 457 rounds. Across all six candidates, mean forward F1 ranged from 0.44945 to 0.45090. The small differences do not suggest a substantial improvement from this parameter search.

On the same outer validation 20%:

| Metric | Current XGBoost | Training-selected candidate |
| --- | ---: | ---: |
| Precision | 36.39% | 35.89% |
| Recall | 67.94% | 69.57% |
| F1 | 0.47398 | 0.47356 |
| Average precision | 0.43610 | 0.43583 |
| Brier score | 0.16017 | 0.15895 |

The candidate slightly improved Brier score and recall, but precision, F1 and average precision decreased. The registered F1 selection rule retained the current model. The rejected candidate was not scored on test. No further search was added after seeing these outcomes.

## Retained test result

The existing current-model test predictions were verified against the frozen cohort and carried forward. Precision remains 33.68%, recall 68.25%, F1 0.45107 and average precision 0.39087.

| Actual / Predicted | No default | Default |
| --- | ---: | ---: |
| No default | 131,903 | 78,959 |
| Default | 18,653 | 40,105 |

The saved paired F1 difference is zero because the selected model is the identical reference; its degenerate interval is not evidence of zero population uncertainty or a new improvement.

## Error diagnostics and practical implications

These are descriptive checks, not criteria for another round of tuning:

- Employment length is missing in 5.10% of training versus 6.40% of validation. Months since last delinquency is missing in 51.39% versus 47.59%; months since last public record in 84.47% versus 80.34%. The existing pipeline retains structural missingness for the added history features. Missing dates must not simply be interpreted as zero months.
- Within the fixed test set, 2017 loans have 23.13% observed defaults, 35.53% precision and 68.69% recall. The 2018 subset has 15.76% observed defaults, 25.81% precision and 67.59% recall. Similar recall and lower precision coincide with the changed default prevalence; these groups also differ in vintage and follow-up. The comparison does not establish causation.
- The global threshold catches few Grade A/B defaults and flags most Grade E/F/G loans. Subgroup metrics and sample counts are saved for interpretation. No grade-specific threshold or subgroup model was fitted from these test diagnostics.
- A higher-precision operating rule is possible, but is not an improvement in both error types. On validation, targeting 50% precision for the current model lowers recall to 28.80%. These validation estimates do not guarantee the same precision on test or future loans, and the maximum-F1 rule remains selected.

The scoped tuning does not fix the false-alarm problem. Report the limitation rather than substituting accuracy, changing the split, excluding difficult loans, or presenting a threshold trade-off as a model upgrade. Additional reliable information or genuinely independent follow-up data may be necessary for a substantial gain; the existing frozen-data project cannot guarantee one.

## Verification and artifacts

All 49 tests passed, including forward-fold separation, bounded candidate checks, ranking rules, frozen split checks, train-only feature preparation and optional XGBoost categorical/model-reload checks. The completed experiment's source hashes and frozen split hash match. Ivy's unchanged original implementation and all 13 original output hashes were preserved. There is no claim of human review or an independently repeated complete search.

Run `python analysis/15_scoped_predictive_tuning.py` in the optional XGBoost environment after script 13 has generated its handoffs. Tables, settings, source hashes and diagnostics are under `outputs/experiments/scoped_tuning_602020`. Loan-level predictions stay ignored locally. The dashboard retains its earlier published experiments. Code, audit, execution and documentation were assisted by Codex and recorded in the AI-use log.

All findings remain exploratory because previous work examined overlapping validation/test records. The accepted/resolved-loan cohort, observational design, incomplete follow-up and historical-vintage limits still apply.
