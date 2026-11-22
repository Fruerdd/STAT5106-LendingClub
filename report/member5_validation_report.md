# Member 5 — Validation, Robustness, and Limitations

## 1. Inputs and verification

- Held-out cohort: issue_date >= 2018-01-01; 56,318 loans, 8,874 defaults. Every model's predictions matched this cohort on ID, outcome, and issue date; no training/validation loan was present.
- Prediction checks (3 models): loan IDs unique within each model and identical to the held-out cohort (56,318 rows per model); all probabilities finite and within [0, 1] (observed range 0.0043 to 0.8745); prediction issue dates 2018-01 to 2018-12.
- Cohort scope: prediction cohorts keep loans with missing values in Member 4's complete-case fields (`revol_util`, `credit_history_months`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`); the prediction pipeline imputes them with training-partition values. Member 4's complete-case rule applies to the inference model only. Primary test cohort: 74 such loans retained (by field: `revol_util` 74, `credit_history_months` 0, `delinq_2yrs` 0, `inq_last_6mths` 0, `open_acc` 0, `pub_rec` 0).
- Selected final model (Pavel, chosen on validation log loss): `hist_gradient_boosting_leaves31_none`. Seed `5106`; validation cutoff 2017-01-01.
- Bootstrap: nonparametric percentile 95% CI, loans resampled with replacement; 1,000 resamples for overall metrics and paired deltas, 500 for thresholds, subgroups, and robustness; seed `5106`. Loan-level resampling assumes independence (ignores issue-month clustering) and does not include model-refitting uncertainty.
- Environment: Python 3.14.6.

## 2. Discrimination

ROC-AUC is the probability that a random defaulter is ranked above a random non-defaulter; it does not depend on prevalence but can look favourable when defaults are rare. PR-AUC is reported as average precision (AP) and must be compared with the default prevalence. Neither metric measures calibration, performance at a specific threshold, or lending costs, and AP is not comparable across cohorts with different default rates.

| Model | Selected | N | Prevalence | ROC-AUC (95% CI) | AP (95% CI) | Brier (95% CI) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| logistic_C0.1_none | No | 56,318 | 15.76% | 0.6939 (0.6881, 0.6995) | 0.2845 (0.2763, 0.2937) | 0.1277 (0.1260, 0.1296) |
| random_forest_leaf20_none | No | 56,318 | 15.76% | 0.7028 (0.6970, 0.7083) | 0.2879 (0.2799, 0.2975) | 0.1258 (0.1241, 0.1276) |
| hist_gradient_boosting_leaves31_none | Yes | 56,318 | 15.76% | 0.7039 (0.6983, 0.7093) | 0.2935 (0.2856, 0.3032) | 0.1260 (0.1243, 0.1278) |

For the selected model, AP 0.2935 is 1.86 times the prevalence baseline 0.1576, which is the expected AP of a random ranking.

Paired bootstrap differences (selected minus comparator, same resampled loans):

- vs `logistic_C0.1_none`: ΔROC-AUC +0.0099 (+0.0081, +0.0117) (CI excludes 0); ΔAP +0.0090 (+0.0064, +0.0117) (CI excludes 0).
- vs `random_forest_leaf20_none`: ΔROC-AUC +0.0011 (-0.0003, +0.0026) (CI includes 0: no clear difference); ΔAP +0.0055 (+0.0030, +0.0081) (CI excludes 0).

## 3. Calibration

Discrimination concerns ranking; calibration concerns whether predicted probabilities match observed default frequencies. Ideal values: intercept 0 (calibration-in-the-large, logistic model with the predicted logit as offset) and slope 1. Intercept < 0 means observed risk is lower than predicted; slope < 1 means predictions are too extreme.

| Model | Mean predicted | Observed | Intercept (95% CI) | Slope (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| logistic_C0.1_none | 19.35% | 15.76% | -0.2800 (-0.3026, -0.2559) | 0.7898 (0.7609, 0.8160) |
| random_forest_leaf20_none | 19.56% | 15.76% | -0.2871 (-0.3100, -0.2635) | 0.9308 (0.8994, 0.9608) |
| hist_gradient_boosting_leaves31_none | 19.19% | 15.76% | -0.2676 (-0.2899, -0.2435) | 0.8282 (0.8012, 0.8542) |

For the selected model, observed default risk is on average **lower** than predicted (intercept -0.2676). Decile bins: `outputs/tables/validation_calibration_bins.csv`.

Brier reference (constant prediction equal to the test prevalence) = p(1 - p) = 0.1327; Brier skill score of the selected model = 1 - Brier / reference = 0.051. The reference uses the observed test prevalence, so it is an optimistic benchmark. The Brier score combines discrimination and calibration.

On the mature-outcome check (train < 2014-01-01, training default rate 15.65%; 2014-01-01 to 2015-01-01 unused; test = 2015-issued 36-month loans, observed default rate 14.89%) the intercept is +0.3298, versus -0.2676 on the primary test period. The sign change cannot be attributed to incomplete follow-up alone: the two analyses also differ in training window, evaluation vintage, and term mix. The alternative-split check is not a truncation-free comparison either, because its evaluation years (2017–2018) are also incompletely followed. Calibration therefore varies by vintage in both directions, and the primary test intercept should not be used to recalibrate the model. Maturity-check term composition: train 36-month 175,426, 60-month 55,280; test 36-month 283,026.

## 4. Thresholds (selected on validation only, then frozen)

- Validation probabilities: Refit of 'hist_gradient_boosting_leaves31_none' on the training partition; reproduces Pavel's test predictions (max |diff| = 9.9e-17); validation/test levels unseen in training: none; fit/transform warnings: none.
- Threshold grid: 159 values from 0.010 to 0.800 (step 0.005); a loan is declined if p >= t. Ties in a selection rule resolve to the lowest threshold.
- Cost units: a false positive (declining a non-defaulter) costs 1 unit and a false negative (approving a defaulter) costs 5 units; cost per loan = (5 x FN + FP) / N. These are relative units, not currency.
- Prevalence: validation 23.13%, test 15.76%.
- Operating goals are **illustrative assumptions set by CLI arguments, pending team confirmation**, not estimates from loss data: cost ratio FN:FP = 5:1, maximum F1, and approval closest to 80%. For a perfectly calibrated model the cost-optimal threshold would be 1/(1+5) = 0.167. Predicted default = declined. Reference rows (approve all / decline all) use no model.

| Goal | Threshold | TP | FP | TN | FN | Precision | Recall (95% CI) | Specificity | F1 (95% CI) | Approval | Default rate among approved (95% CI) | Cost/loan (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cost-optimal (5:1 FN:FP) | 0.140 | 7,195 | 25,195 | 22,249 | 1,679 | 22.21% | 81.08% (80.19%, 81.87%) | 46.90% | 0.3487 (0.3427, 0.3545) | 42.49% | 7.02% (6.70%, 7.38%) | 0.5964 (0.5890, 0.6047) |
| Maximum F1 | 0.195 | 5,870 | 17,262 | 30,182 | 3,004 | 25.38% | 66.15% (65.16%, 67.10%) | 63.62% | 0.3668 (0.3607, 0.3736) | 58.93% | 9.05% (8.77%, 9.36%) | 0.5732 (0.5645, 0.5833) |
| Approval closest to 80% | 0.310 | 3,147 | 6,833 | 40,611 | 5,727 | 31.53% | 35.46% (34.47%, 36.51%) | 85.60% | 0.3338 (0.3253, 0.3435) | 82.28% | 12.36% (12.07%, 12.67%) | 0.6298 (0.6171, 0.6427) |
| Reference: approve all | — | 0 | 0 | 47,444 | 8,874 | N/A | 0.00% (0.00%, 0.00%) | 100.00% | N/A | 100.00% | 15.76% (15.47%, 16.02%) | 0.7878 (0.7735, 0.8009) |
| Reference: decline all | — | 8,874 | 47,444 | 0 | 0 | 15.76% | 100.00% (100.00%, 100.00%) | 0.00% | 0.2722 (0.2680, 0.2768) | 0.00% | N/A | 0.8424 (0.8394, 0.8453) |

Under the assumed cost ratio, the cost-optimal threshold costs 0.5964 per loan versus 0.7878 (approve all) and 0.8424 (decline all): a reduction of 0.1914 per loan relative to the better no-model policy.

## 5. Subgroups (selected model, held-out test)

Adequate = n >= 200 with >= 30 defaults and non-defaults. Metrics and calibration are estimated only for adequate strata; inadequate strata are listed with counts only. Within-grade ROC-AUC is expected to be lower because grade already stratifies risk. Many strata are compared, so isolated differences should not be over-interpreted. None of the subgroup variables (`grade`, `term_months`, `home_ownership`, `purpose`, `verification_status`, `application_type`, `emp_length_years_was_missing`) has missing values in the test cohort, so no `<Missing>` level arises. Predictor missingness is assessed with `emp_length_years_was_missing` (1 = employment length missing and imputed).

| Variable | Level | N | Defaults | Observed | Predicted | ROC-AUC (95% CI) | Cal. intercept | Adequate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| grade | A | 12,820 | 661 | 5.16% | 5.69% | 0.6497 (0.6286, 0.6711) | -0.1062 | Yes |
| grade | B | 15,233 | 1,785 | 11.72% | 13.72% | 0.6050 (0.5914, 0.6193) | -0.1859 | Yes |
| grade | C | 14,755 | 2,751 | 18.64% | 22.24% | 0.6118 (0.5998, 0.6228) | -0.2295 | Yes |
| grade | D | 9,675 | 2,498 | 25.82% | 31.64% | 0.5993 (0.5873, 0.6110) | -0.2993 | Yes |
| grade | E | 3,032 | 883 | 29.12% | 40.74% | 0.6275 (0.6043, 0.6483) | -0.5505 | Yes |
| grade | F | 660 | 240 | 36.36% | 51.69% | 0.5749 (0.5235, 0.6155) | -0.6542 | Yes |
| grade | G | 143 | 56 | 39.16% | 49.76% | N/A | N/A | No |
| term_months | 36 | 41,272 | 5,468 | 13.25% | 14.81% | 0.7055 (0.6986, 0.7130) | -0.1400 | Yes |
| term_months | 60 | 15,046 | 3,406 | 22.64% | 31.21% | 0.6550 (0.6451, 0.6649) | -0.4797 | Yes |
| home_ownership | ANY | 23 | 3 | 13.04% | 27.47% | N/A | N/A | No |
| home_ownership | MORTGAGE | 28,518 | 3,389 | 11.88% | 16.94% | 0.6906 (0.6821, 0.6997) | -0.4563 | Yes |
| home_ownership | OWN | 7,523 | 1,284 | 17.07% | 19.51% | 0.6897 (0.6741, 0.7055) | -0.1813 | Yes |
| home_ownership | RENT | 20,254 | 4,198 | 20.73% | 22.24% | 0.6990 (0.6903, 0.7073) | -0.1017 | Yes |
| purpose | car | 776 | 94 | 12.11% | 14.23% | 0.7699 (0.7261, 0.8162) | -0.2039 | Yes |
| purpose | credit_card | 11,397 | 1,409 | 12.36% | 17.18% | 0.6934 (0.6806, 0.7082) | -0.4293 | Yes |
| purpose | debt_consolidation | 28,734 | 4,643 | 16.16% | 20.78% | 0.6944 (0.6869, 0.7023) | -0.3447 | Yes |
| purpose | home_improvement | 4,522 | 634 | 14.02% | 16.41% | 0.7161 (0.6957, 0.7358) | -0.2064 | Yes |
| purpose | house | 998 | 179 | 17.94% | 17.59% | 0.7160 (0.6754, 0.7530) | +0.0266 | Yes |
| purpose | major_purchase | 1,806 | 349 | 19.32% | 17.43% | 0.7083 (0.6801, 0.7350) | +0.1424 | Yes |
| purpose | medical | 1,031 | 212 | 20.56% | 18.98% | 0.7231 (0.6868, 0.7608) | +0.1120 | Yes |
| purpose | moving | 507 | 105 | 20.71% | 20.83% | 0.7230 (0.6699, 0.7725) | -0.0079 | Yes |
| purpose | other | 5,404 | 970 | 17.95% | 18.30% | 0.7204 (0.7034, 0.7365) | -0.0261 | Yes |
| purpose | renewable_energy | 38 | 6 | 15.79% | 18.02% | N/A | N/A | No |
| purpose | small_business | 569 | 181 | 31.81% | 26.28% | 0.7209 (0.6748, 0.7611) | +0.3061 | Yes |
| purpose | vacation | 534 | 91 | 17.04% | 17.16% | 0.7844 (0.7286, 0.8401) | -0.0090 | Yes |
| purpose | wedding | 2 | 1 | 50.00% | 18.44% | N/A | N/A | No |
| verification_status | Not Verified | 21,012 | 2,462 | 11.72% | 15.36% | 0.7203 (0.7108, 0.7310) | -0.3498 | Yes |
| verification_status | Source Verified | 22,547 | 3,554 | 15.76% | 19.66% | 0.6886 (0.6795, 0.6977) | -0.2953 | Yes |
| verification_status | Verified | 12,759 | 2,858 | 22.40% | 24.68% | 0.6598 (0.6511, 0.6711) | -0.1422 | Yes |
| application_type | Individual | 49,230 | 7,470 | 15.17% | 18.72% | 0.7136 (0.7075, 0.7191) | -0.2817 | Yes |
| application_type | Joint App | 7,088 | 1,404 | 19.81% | 22.49% | 0.6402 (0.6255, 0.6565) | -0.1844 | Yes |
| emp_length_years_was_missing | 0 | 51,384 | 7,690 | 14.97% | 18.61% | 0.7046 (0.6989, 0.7101) | -0.2926 | Yes |
| emp_length_years_was_missing | 1 | 4,934 | 1,184 | 24.00% | 25.26% | 0.6615 (0.6452, 0.6789) | -0.0759 | Yes |

## 6. Robustness (all prespecified checks reported, including failures)

Rows evaluated on different cohorts have different prevalences; compare AP only within a cohort.

| Dimension | Specification | Cohort | Status | N | Default rate | ROC-AUC (95% CI) | AP (95% CI) | Brier | Cal. intercept |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary result (reference) | hist_gradient_boosting_leaves31_none | Test >= 2018-01-01 | reference | 56,318 | 15.76% | 0.7039 (0.6983, 0.7093) | 0.2935 (0.2856, 0.3032) | 0.1260 | -0.2676 |
| Prespecified model family / class weighting | logistic_C0.1_none | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7016 | 0.3970 | 0.1637 | N/A |
| Prespecified model family / class weighting | logistic_C0.1_balanced | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7018 | 0.3967 | 0.2208 | N/A |
| Prespecified model family / class weighting | logistic_C1_none | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7016 | 0.3970 | 0.1637 | N/A |
| Prespecified model family / class weighting | logistic_C1_balanced | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7018 | 0.3967 | 0.2208 | N/A |
| Prespecified model family / class weighting | random_forest_leaf20_none | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7057 | 0.4024 | 0.1620 | N/A |
| Prespecified model family / class weighting | random_forest_leaf20_balanced | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7058 | 0.4023 | 0.2026 | N/A |
| Prespecified model family / class weighting | random_forest_leaf100_none | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7048 | 0.4013 | 0.1625 | N/A |
| Prespecified model family / class weighting | random_forest_leaf100_balanced | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7048 | 0.4006 | 0.2149 | N/A |
| Prespecified model family / class weighting | hist_gradient_boosting_leaves31_none | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7081 | 0.4056 | 0.1616 | N/A |
| Prespecified model family / class weighting | hist_gradient_boosting_leaves31_balanced | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7069 | 0.4045 | 0.2207 | N/A |
| Prespecified model family / class weighting | hist_gradient_boosting_leaves63_none | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7079 | 0.4053 | 0.1616 | N/A |
| Prespecified model family / class weighting | hist_gradient_boosting_leaves63_balanced | Validation 2017-01-01 to 2018-01-01 (not comparable with test rows) | from model_comparison.csv | 169,321 | 23.13% | 0.7076 | 0.4052 | 0.2213 | N/A |
| Alternative temporal split | hist_gradient_boosting_leaves31_none: train < 2016-01-01 | Test >= 2017-01-01 (includes primary validation and test years; 2016-01-01 to 2017-01-01 unused: no tuning in this check) | completed | 225,639 | 21.29% | 0.7043 (0.7019, 0.7070) | 0.3724 (0.3686, 0.3773) | 0.1543 | +0.1834 |
| Feature specification: no lender pricing proxy | hist_gradient_boosting_leaves31_none: int_rate fixed at training median (uninformative) | Test >= 2018-01-01 | completed | 56,318 | 15.76% | 0.6944 (0.6888, 0.7003) | 0.2886 (0.2806, 0.2974) | 0.1256 | -0.2090 |
| Term stratification (36-month loans only) | hist_gradient_boosting_leaves31_none: refit on 36-month training loans (train < 2017-01-01), evaluated on 36-month test loans | Test >= 2018-01-01, 36-month | completed | 41,272 | 13.25% | 0.7074 (0.7007, 0.7142) | 0.2576 (0.2479, 0.2678) | 0.1080 | -0.1431 |
| Outcome maturity (issue_date + term <= 2018-12-31) | hist_gradient_boosting_leaves31_none: mature loans, train < 2014-01-01 | Mature test >= 2015-01-01 | completed | 283,026 | 14.89% | 0.6797 (0.6770, 0.6825) | 0.2548 (0.2512, 0.2583) | 0.1218 | +0.3298 |

Refit diagnostics (unseen categorical levels, fit/transform warnings, cohort composition):

- *Alternative temporal split*: `{"fit_warnings": [], "unseen_level_rows": {}, "unseen_levels": {}}`
- *Feature specification: no lender pricing proxy*: `{"fit_warnings": [], "int_rate_fixed_at": 12.74, "unseen_level_rows": {}, "unseen_levels": {}}`
- *Term stratification (36-month loans only)*: `{"fit_warnings": [], "train_rows": 853383, "unseen_level_rows": {}, "unseen_levels": {}}`
- *Outcome maturity (issue_date + term <= 2018-12-31)*: `{"complete_case_fields": ["revol_util", "credit_history_months", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec"], "complete_case_missing_rows": 418, "complete_case_rows": 675884, "fit_warnings": ["Found unknown categories in columns [4] during transform. These unknown categories will be encoded as all zeros"], "mature_cohort_rows": 676302, "test_issue_years": [2015], "test_term_counts": {"36.0": 283026}, "train_default_rate": 0.15647187329328235, "train_term_counts": {"36.0": 175426, "60.0": 55280}, "unseen_level_rows": {"application_type": 239, "home_ownership_grouped": 1}, "unseen_levels": {"application_type": ["Joint App"], "home_ownership_grouped": ["ANY"]}}`

The 36-month row is a model **refitted on 36-month training loans only**. It differs from the 36-month subgroup in Section 5, which restricts the primary model's predictions to 36-month loans (ROC-AUC 0.7074 vs 0.7055).
Fixing `int_rate` lowers ROC-AUC (0.7039 to 0.6944) but slightly improves the Brier score (0.1260 to 0.1256) and calibration-in-the-large (-0.2676 to -0.2090). One possible explanation is drift in the relationship between lender pricing and default over time; this is an observation, not a tested explanation.
The maturity cohort (676,302 rows) retains 418 loans with a missing value in at least one complete-case field (`revol_util`, `credit_history_months`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`); the prediction pipeline imputes them. Applying Member 4's complete-case rule would leave 675,884 rows.

## 7. Limitations

Observed default rate by issue year (resolved loans only):

| Issue year | Loans | Defaults | Default rate |
| ---: | ---: | ---: | ---: |
| 2007 | 603 | 158 | 26.20% |
| 2008 | 2,393 | 496 | 20.73% |
| 2009 | 5,281 | 723 | 13.69% |
| 2010 | 12,537 | 1,757 | 14.01% |
| 2011 | 21,721 | 3,297 | 15.18% |
| 2012 | 53,367 | 8,644 | 16.20% |
| 2013 | 134,804 | 21,024 | 15.60% |
| 2014 | 223,103 | 41,162 | 18.45% |
| 2015 | 375,546 | 75,804 | 20.19% |
| 2016 | 293,105 | 68,252 | 23.29% |
| 2017 | 169,321 | 39,169 | 23.13% |
| 2018 | 56,318 | 8,874 | 15.76% |

1. **Temporal drift.** Default rates and predictor missingness change across vintages (table above; `validation_missingness_by_partition.csv`). The selected model's test calibration intercept is -0.2676; see Section 3 and the alternative-split and maturity rows in Section 6.
2. **Unresolved-outcome exclusion and unequal follow-up.** The cleaning step excluded 912,602 loans with unresolved or unknown status. Loans issued on or after 2018-01-01 had between 0 and 11 months of follow-up before the dataset end (2018-12-31), so the resolved test loans are a selected subset (resolved early). Test metrics describe that subset, not the full vintage. The share of test-period issued loans that is retained is not reported: the number of loans issued on or after the test cutoff before status exclusion was not supplied (--test-period-issued-loans). The maturity-cohort check in Section 6 reduces, but does not remove, this problem.
3. **Selection bias.** Only accepted and funded loans are observed. Results do not apply to rejected applicants.
4. **Missing data.** Highest test-period missingness among raw predictors: `emp_length_years` train 5.478%, test 8.761%; `dti` train 0.004%, test 0.268%; `revol_util` train 0.060%, test 0.131%. Imputation values were learned on the training partition only. Prediction cohorts keep loans missing any of Member 4's complete-case fields and impute them (primary test: 74; maturity cohort: 418); Member 4's complete-case inference cohort excludes them, so its sample sizes differ from these.
5. **Subgroup limits.** 4 of 33 strata are inadequate and are not interpreted: grade=G (n=143), home_ownership=ANY (n=23), purpose=renewable_energy (n=38), purpose=wedding (n=2).
6. **Non-causal interpretation and generalisability.** Predictors such as interest rate reflect Lending Club's risk-based pricing; associations are not causal. Results apply to U.S. Lending Club accepted loans with resolved outcomes (2007–2018) and may not transfer to other lenders or economic conditions.
7. **Threshold transfer.** The threshold that minimised cost on validation (t = 0.140, prevalence 23.13%) costs 0.5964 (0.5890, 0.6047) per loan on test, more than the 'Maximum F1' threshold (t = 0.195, 0.5732 (0.5645, 0.5833)). The intervals do not overlap (they come from separate bootstraps, so this is an approximate comparison). Test prevalence is 15.76%. Cost-optimal thresholds depend on prevalence and calibration, so they do not transfer unchanged across vintages. Thresholds were not re-tuned on the test set; a deployment threshold should be re-selected on recent, fully observed data.
8. **Method limits.** Thresholds depend on illustrative cost and approval assumptions. The no-pricing-proxy check neutralises `int_rate` at its training median instead of re-specifying the model. Loan-level bootstrap CIs may be too narrow if outcomes cluster in time. In the outcome maturity check, evaluation-period levels not seen in training: `application_type` = Joint App (239 loans); `home_ownership_grouped` = ANY (1 loans). Their encoding depends on `TrainOnlyPreprocessor` (rare-level grouping or all-zero one-hot); preprocessing warnings: `Found unknown categories in columns [4] during transform. These unknown categories will be encoded as all zeros`.
