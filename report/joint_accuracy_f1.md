# Why accuracy and default F1 are limited

The latest predictive experiment uses the project's frozen chronological 60/20/20 split. The comparison covers 269,620 test loans, including 58,758 defaults. Default is the positive class throughout. Accuracy counts all correct loan classifications. F1 is the harmonic mean of precision and recall for defaults.

## Where the current model makes mistakes

- **False alarms account for most errors.** At its validation-selected cutoff, the retained grade-based XGBoost flagged 119,064 test loans. Of these, 78,959 were fully paid and 40,105 defaulted. It also missed 18,653 defaults. The 33.68% precision limits F1 even though the model finds 68.25% of defaults. Since 78.21% of the loans were fully paid, predicting repayment for everyone would score 78.21% accuracy and zero default F1.
- **The scores only partly separate borrowers within a grade.** Grades C and D together accounted for 58,949 false positives, or 74.66% of all false positives. Within-grade test ROC-AUC was 0.622 for C and 0.615 for D, versus 0.711 overall. About 22% of all test loans had a reference score within 0.05 of the 0.29416 cutoff. Among the 28,607 loans in the first 0.05 above it, 7,070 actually defaulted. Raising the cutoff in this region can remove many false alarms, but also loses real defaults.
- **The population changes over time.** From the 2017 to the 2018 issue-year subset, observed defaults fell from 23.13% to 15.76%. Reference recall stayed near 68%, while precision fell from 35.53% to 25.81%. The study includes only accepted loans with resolved outcomes, and later loans have shorter follow-up. These figures describe a change in the observed cohort. They cannot establish how much of the difference reflects borrower risk or outcome maturity.

The previous experiment reproduced the reference confusion matrices. The current cutoff also attained the highest default F1 among every distinct validation score in this run. Raising accuracy by changing that cutoff alone had a measurable cost. The highest validation F1 available with at least two additional accuracy percentage points was 0.47328 at 67.72% accuracy, versus 0.47398 at 65.42% accuracy under the original rule.

## Change tested

One XGBoost challenger replaced the seven grade categories with the 35 ordered origination subgrades A1 through G5. Trees can split at any subgrade boundary. Three additional quantities came from fields already permitted in the predictive extension.

| Predictor | Calculation |
| --- | --- |
| Revolving balance relative to income | Revolving balance divided by annual income |
| Loan relative to revolving balance | Requested loan amount divided by revolving balance |
| Estimated revolving credit limit | Revolving balance divided by the utilization fraction. Rounding in reported utilization makes this approximate. |

Nonpositive denominators yield missing values. XGBoost handles these natively for the three ratios. The model uses a single lender risk proxy, the existing training-fitted transformations for the other predictors, and the reference depth and positive-class weight. The 30 model inputs exclude IDs, issue date, and post-origination information. The data source, target, and split match the previous predictive experiment.

The latest issue months within training (October through December 2015) selected 351 boosting rounds by early stopping. The model was then refitted on all 808,859 training loans. On validation, the selection rule maximized the smaller of the accuracy and F1 gains over the reference, using both metrics on a 0–1 scale. Each tied score received the same decision. Both gains had to be positive to advance. The selected threshold, 0.3025958240032196, also attained the challenger's maximum validation F1. The threshold and model choice were saved before the challenger was scored on test.

## Same-loan comparison

| Cohort and model | Accuracy | Default F1 | Precision | Recall | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation, reference | 65.42% | 0.47398 | 36.39% | 67.94% | 0.43610 |
| Validation, challenger | 66.38% | 0.47518 | 37.00% | 66.38% | 0.43705 |
| Test, reference | 63.80% | 0.45107 | 33.68% | 68.25% | 0.39087 |
| Test, challenger | 64.72% | 0.45152 | 34.14% | 66.64% | 0.39200 |

| Test errors | Reference | Challenger | Change |
| --- | ---: | ---: | ---: |
| Fully paid loans flagged as defaults | 78,959 | 75,535 | −3,424 |
| Defaults predicted to be fully paid | 18,653 | 19,600 | +947 |
| Total errors | 97,612 | 95,135 | −2,477 |

Accuracy increased by **0.92 percentage points**, with a paired 95% interval of **+0.83 to +1.01 points**. Default F1 increased by **0.00045**, with a paired 95% interval of **−0.00064 to +0.00153**. These percentile intervals use 5,000 paired individual-loan bootstrap samples with seed 5106, conditional on the fitted models and thresholds. Earlier experiments examined this same test set. The evidence supports a modest accuracy improvement on this cohort. A repeatable improvement in default F1 remains uncertain. The feature changes were evaluated together, so their individual contributions cannot be isolated.

## Reproduction

From the repository root, in the environment for `requirements-preprocessing-experiment.txt`, run the following after scripts 13 and 16 have produced the frozen split contract and full-precision reference predictions.

```sh
.venv/bin/python analysis/17_joint_accuracy_f1.py
```

The [validation metrics](../outputs/experiments/joint_accuracy_f1_602020/validation.csv), [frozen decision rule](../outputs/experiments/joint_accuracy_f1_602020/frozen_choice.json), and [test metrics and paired intervals](../outputs/experiments/joint_accuracy_f1_602020/test.csv) are separate from the primary study. The fitted challenger is stored locally at `artifacts/joint_accuracy_f1_602020/selected.joblib`. Its full-precision validation and test predictions are in `data/cleaned/joint_accuracy_f1_602020_handoff/`. The reference grade and issue-year counts come from the [existing subgroup assessment](../outputs/experiments/scoped_tuning_602020/selected_test_diagnostics.csv).
