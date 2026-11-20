# Pavel — Predictive Modelling Report

## Design and equal comparison

All candidates use the same resolved-outcome loans, approved Model C application-time predictors, temporal partitions, random seed, and training-fitted preprocessing. Interest rate is included; grade, issue year, identifiers, raw dates, loan status, and post-outcome fields are excluded. No observations are resampled or removed for preprocessing. Class weighting is evaluated as a prespecified candidate setting.

Training uses issue dates before 2017-01-01; validation uses 2017-01-01 through the day before 2018-01-01; testing uses 2018-01-01 onward. Hyperparameters and the final model minimize validation log loss. Test data do not enter fitting, preprocessing, selection, or tuning. Test metrics are calculated only for the selected candidate from each model family.

| Partition | Rows | Defaults | Default rate |
| --- | ---: | ---: | ---: |
| train | 1,122,460 | 221,317 | 19.72% |
| validation | 169,321 | 39,169 | 23.13% |
| test | 56,318 | 8,874 | 15.76% |

## Model comparison

The selected final model is **hist_gradient_boosting_leaves31_none**, based on validation log loss. PR-AUC should be compared with the partition's default-rate baseline. Lower Brier score and log loss indicate better probabilistic predictions; ROC-AUC and PR-AUC measure ranking rather than calibration.

| Model | Validation ROC-AUC | Validation PR-AUC | Validation log loss | Test ROC-AUC | Test PR-AUC | Test Brier | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_C0.1_none | 0.7016 | 0.3970 | 0.4995 | 0.6939 | 0.2845 | 0.1277 | 0.4114 |
| random_forest_leaf20_none | 0.7057 | 0.4024 | 0.4942 | 0.7028 | 0.2879 | 0.1258 | 0.4062 |
| hist_gradient_boosting_leaves31_none | 0.7081 | 0.4056 | 0.4933 | 0.7039 | 0.2935 | 0.1260 | 0.4060 |

## Logistic interpretation

The selected logistic candidate is interpreted on the original engineered scales by reversing training-set standardization. Odds ratios compare the stated unit or categorical level with its reference while holding other predictors fixed. The strongest terms below are ranked by absolute standardized coefficient, which is a predictive measure and not a causal effect or an inferential significance test.

| Parameter | Odds ratio | Comparison |
| --- | ---: | --- |
| term_months_60 | 1.808 | 60 versus 36 (reference) |
| purpose_grouped_wedding | 0.611 | wedding versus debt_consolidation (reference) |
| purpose_grouped_small_business | 1.534 | small_business versus debt_consolidation (reference) |
| int_rate_per_5 | 1.504 | Per 5 interest-rate percentage points; training-only winsorization where specified |
| home_ownership_grouped_RENT | 1.378 | RENT versus MORTGAGE (reference) |
| purpose_grouped_OTHER_RARE | 1.324 | OTHER_RARE versus debt_consolidation (reference) |
| fico_per_10 | 0.930 | Per 10 FICO points; training-only winsorization where specified |
| purpose_grouped_medical | 1.196 | medical versus debt_consolidation (reference) |

Full coefficients and reference categories are in `outputs/tables/logistic_interpretation.csv`. Continuous predictors marked in the feature contract are capped at training-set 0.5th and 99.5th percentiles.

## Limitations and handoff

The analysis includes only resolved outcomes, so it does not represent current or unresolved applications. The 2018 cohort has shorter outcome follow-up and its observed default rate is not directly comparable with mature vintages. Temporal drift and missingness may reduce transportability. Predictive associations do not establish causation.

`out_of_time_predictions.csv` contains the real 2018 predictions for each selected model family. Confidence intervals, threshold selection, subgroup analysis, and further robustness checks remain Member 5's validation work.
