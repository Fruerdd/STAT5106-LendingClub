# Follow-up experiment on default F1

The equal average of two XGBoost models raised measured test default F1 from **0.45152 to 0.45219**, and accuracy from **64.72% to 65.09%**. The first model is the previous experiment's ordinal-subgrade model. The second gives newer training loans more weight and adds four predictors from the same application records. This average placed first on each of the two forward training comparisons.

| Cohort and model | Accuracy | Default F1 | Precision | Recall | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation, previous model | 66.38% | 0.47518 | 37.00% | 66.38% | 0.43705 |
| Validation, equal average | 66.72% | 0.47586 | 37.24% | 65.89% | 0.43878 |
| Test, previous model | 64.72% | 0.45152 | 34.14% | 66.64% | 0.39200 |
| Test, equal average | 65.09% | 0.45219 | 34.36% | 66.12% | 0.39326 |

The ensemble's maximum-F1 validation cutoff of 0.3097699135541916 already satisfied the accuracy floor. Applying that frozen rule to 269,620 test loans avoided 1,321 false alarms and missed 310 additional defaults. The result is 1,011 fewer classification errors. Probability errors were slightly larger, with test Brier score increasing from 0.16043 to 0.16103.

The paired 95% interval for the F1 change of +0.00068 runs from −0.00008 to +0.00141. The interval for the accuracy gain of 0.375 percentage points runs from +0.317 to +0.431 points. The experiment supports reporting a small observed F1 gain and leaves its repeatability uncertain. [Test comparison and paired intervals](../outputs/experiments/temporal_f1_602020/test.csv)

## Question and comparison

Can the ordinal-subgrade XGBoost separate defaults from fully paid loans better if it learns more from recent loans or gets additional information about debt payments and loan size? The reference is the fitted model from [experiment 17](joint_accuracy_f1.md). The project keeps the same resolved outcomes, origination sources and frozen chronological assignments of 808,859 training loans, 269,620 validation loans and 269,620 test loans.

The comparison covers four alternatives.

| Alternative | Change from the reference |
| --- | --- |
| Recent loans | Weight each training loan by its issue month. Each 12 months of age halves its weight. Normalize weights to mean one and retain every loan. |
| Loan terms | Add four predictors derived from the existing application fields below. Give training loans equal weight across issue months. |
| Combined | Use both changes. |
| Equal average | Average the reference and combined model scores, giving each model half the weight. |

The four additional predictors are estimated monthly debt payments (annual income × reported DTI in percentage points / 1,200), contractual installment divided by those estimated payments, revolving balance per open credit account, and an indicator for requested loan amounts divisible by $1,000. The last feature tests whether the precision of the requested amount contains signal. Nonpositive denominators give missing values. The issue month sets training weights only. Every model scores validation and test loans with application-time predictors and a single global cutoff.

## Selection and assessment

1. Use two forward folds inside the training set. In each fold, select the reference's boosting rounds by early stopping on the latest months of the fitting block, then use that round count for all fold fits. Compare the four alternatives with the same reference depth, class weight and random seed.
2. Choose each cutoff on the next block of training months by maximum default F1, requiring at least the accuracy of the reference's maximum-F1 rule on that block. Score it on a still-later block. Select one challenger by highest mean forward F1 across the two score blocks, breaking ties by mean average precision and then model name.
3. Fit that challenger on the full training set with the reference's 351 rounds, originally selected inside training. On outer validation, report its maximum F1 both with and without the reference accuracy floor. Advance the challenger only if it strictly improves F1 while maintaining at least the reference validation accuracy of 66.3775% (full precision in code).
4. Freeze the selected model and cutoff before evaluating a successful challenger on test. Compute percentile intervals for changes in accuracy and F1 from 5,000 paired individual-loan bootstrap draws, conditional on the fitted models and cutoffs. Earlier experiments examined this test cohort, so the comparison remains exploratory.

To reproduce after experiment 17, run `.venv/bin/python analysis/18_temporal_f1.py` from the project root using the same environment. The [training comparison](../outputs/experiments/temporal_f1_602020/training_ranking.csv), [validation results](../outputs/experiments/temporal_f1_602020/validation.csv), and [selection record](../outputs/experiments/temporal_f1_602020/choice.json) are aggregate files.

The fitted bundle is stored locally at `artifacts/temporal_f1_602020/selected.joblib`. Call `src.temporal_f1.predict(bundle, frame)` with cleaned application records and predict default for scores at or above `bundle["threshold"]`. Full-precision predictions are in `data/cleaned/temporal_f1_602020_handoff/`. Reloading the fitted bundle reproduced validation scores exactly. Independent checks verified loan identities against both frozen partitions, confusion counts, accuracy, F1, precision, recall, and average precision against the exported predictions for both models. All 62 tests passed.
