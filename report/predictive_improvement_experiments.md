# Predictive improvement experiments

The expanded grade-based model improves ranking and probability accuracy on the same 56,318 resolved 2018 loans. This is an exploratory follow-up after the original validation and test results were inspected. The primary analysis, inference report, original dashboard results and Ivy's original validation are preserved.

[Open the model improvement dashboard](https://lending-club-research-dashboard.compassionateblimp128.chatgpt.site/#improvement).

## Search and selection

Thirteen candidate specifications were evaluated on 2016 average precision using training loans issued before 2016: seven original-feature variants, two grade-based variants and four reviewed feature extensions. Thresholds were selected on 2017 predictions only. Models were refitted using loans before 2017. No candidate was chosen using its 2018 performance. The initial tuning-only changes did not materially improve 2018 results; all unsuccessful comparisons are retained under outputs/experiments.

The selected extension is expanded_grade_31. It adds independent loan amount, payment burden, revolving balance, account breadth/composition, credit-event history and application state. Grade replaces interest rate; grade, subgrade and interest rate are never combined. The feature audit and pre-fit specification are in predictive_feature_extension_specification.md.

## Same-cohort comparison

| Metric | Original | Expanded model |
| --- | ---: | ---: |
| ROC-AUC | 0.7039 | 0.7195 |
| Average precision | 0.2935 | 0.3162 |
| Brier score | 0.1260 | 0.1230 |
| Log loss | 0.4060 | 0.3977 |
| Descriptive precision at recall >=70% | 24.70% | 25.62% |

Average precision increases by 0.0227 (7.7% relative). The paired 95% percentile bootstrap interval for the difference is [0.0189, 0.0265], using 1,000 loan resamples and seed 5106. The PR-curve precision-at-recall summary is descriptive; it is not a threshold selected for deployment.

## Precision and recall at frozen operating rules

Both models use thresholds chosen on 2017. The full set of observed validation-score thresholds is used, which differs slightly from Ivy's original 159-point threshold grid. The 2018 outcomes do not determine these thresholds.

| Rule | Model | Precision | Recall | F1 | False positives | Defaults found |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| maximum_validation_f1 | original_baseline | 25.41% | 66.04% | 0.3669 | 17,206 | 5,860 |
| maximum_validation_f1 | expanded_grade_31 | 26.60% | 66.19% | 0.3795 | 16,210 | 5,874 |
| validation_cost_5_to_1 | original_baseline | 22.16% | 81.29% | 0.3483 | 25,334 | 7,214 |
| validation_cost_5_to_1 | expanded_grade_31 | 22.62% | 82.60% | 0.3551 | 25,075 | 7,330 |

At the maximum-validation-F1 rule, the extension avoids 996 false positives and finds 14 more defaults. The precision and F1 improvements have positive paired bootstrap intervals, while the recall difference is uncertain. At the illustrative 5:1 cost rule, both precision and recall differences have positive conditional bootstrap intervals. These intervals do not account for model-search or training uncertainty.

## Calibration, subgroups and repeatability

Mean predicted risk is 18.70% versus 15.76% observed default; calibration-in-the-large is -0.231 and slope 0.923. Ideal values are 0 and 1. Aggregate risk remains overpredicted. The original corresponding values were 19.19%, -0.268 and 0.828.

Ivy's original validation utility functions were reused without modification. There are 33 subgroup strata, with 29 meeting the original adequacy rule. Grade G and other small groups do not support performance conclusions. A separate repeated fit reproduced the saved extension metrics, and prediction IDs/outcomes/dates matched the expected test cohort. These checks are specific to the extension; the original Member 6 certificate is not relabelled as its full reproduction.

## Limits and status

The precision improvement is useful but modest: most flagged loans still do not default. Only accepted loans with resolved outcomes are evaluated. Unequal follow-up, selection into resolution, geographic variation, dependence between loans and future economic changes remain limitations. The 2018 set has been reused during this follow-up, so this is not a fresh confirmatory test. No additional temporal robustness fits were performed for the extension; the original model's robustness findings do not automatically apply to it. A new prospective or external cohort would be needed to establish generalization.

This is an experiment note, not the final group report. The original primary results are retained for comparison. Follow-up implementation/execution used Codex; original validation utilities are credited to Ivy. No row-level predictions are published. Local prediction handoffs are ignored under data/cleaned/feature_extension_handoff.

## Reproduction

```bash
python analysis/07_predictive_improvement.py
python analysis/08_grade_prediction_benchmark.py
python analysis/09_frozen_improvement_assessment.py
python analysis/10_predictive_feature_extension.py
python analysis/11_validate_feature_extension.py
```

Aggregate results: outputs/experiments/feature_extension. The earlier experiment directories preserve the unsuccessful tuning-only comparisons.
