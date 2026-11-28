# Reducing false alarms in the default model

The precision-focused rule raised test precision from **34.36% to 35.80%** and reduced false alarms by **10,934 (14.7%)**. It missed 3,557 additional defaults. Overall accuracy increased by 2.74 percentage points, and default F1 declined by 0.00357.

| Frozen test set, 269,620 loans | Maximum-F1 rule | Precision-focused rule |
| --- | ---: | ---: |
| Score cutoff | 0.30977 | 0.33609 |
| False positives | 74,214 | 63,280 |
| True positives | 38,848 | 35,291 |
| False negatives | 19,910 | 23,467 |
| Precision | 34.36% | 35.80% |
| Recall | 66.12% | 60.06% |
| Accuracy | 65.09% | 67.83% |
| Default F1 | 0.45219 | 0.44863 |

## How the rule was selected

I used 60% recall on validation as the minimum so the precision-focused policy would still catch most defaults. I selected the highest-precision cutoff that met that floor. The resulting cutoff was frozen before scoring the test predictions. The recall floor is a selection criterion on historical validation data.

I also trained one histogram gradient boosting model on the existing 34 predictors and frozen training loans. It used 351 iterations, 31 maximum leaves, 100 minimum samples per leaf, learning rate 0.05, L2 regularization 5, and double the weight for defaults. Its score was averaged equally with the incumbent score. The blend would advance if it improved both precision at the validation recall floor and average precision. It yielded 38.93% precision and 0.43821 average precision, compared with 38.98% and 0.43878 for the incumbent. The incumbent with the new cutoff advanced to test.

This operating point reduces the review workload and trades away some default detection. Nearly two-thirds of flags are still false alarms. The underlying ranking is the same as the earlier ensemble, with test average precision of 0.39326. The same test loans were examined in earlier experiments.

## Reproduction

After running experiment 18, run `.venv/bin/python analysis/19_precision_focus.py` from the repository root. The [validation comparison](../outputs/experiments/precision_focus_602020/validation.csv), [test comparison](../outputs/experiments/precision_focus_602020/test.csv), and [frozen rule](../outputs/experiments/precision_focus_602020/frozen_rule.json) are saved separately. To use the precision-focused rule, score with the existing experiment 18 bundle and apply the full-precision cutoff in the frozen rule file.
