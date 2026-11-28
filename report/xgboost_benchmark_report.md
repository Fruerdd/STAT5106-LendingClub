# XGBoost benchmark

XGBoost did not establish a clear F1 improvement over the expanded histogram-gradient-boosting model. Its small F1 increase came with higher precision, lower recall, lower average precision and worse probability accuracy. The expanded model remains the preferred reference; no primary results were replaced.

## Controlled development

Six XGBoost settings use the same 27 audited expanded grade predictors: depths 3 and 5, each with positive-class weight 1, weight 2, or the training negative/positive ratio. All transformations and category mappings are fitted on the applicable training partition. Grade replaces interest rate; post-outcome fields are excluded.

Boosting round count is chosen using training before October 2015 and early stopping on 2015 Q4 log loss. Each candidate is then refitted before 2016. Its threshold is chosen on 2016 H1, frozen, and its F1 is evaluated on 2016 H2. Candidates are selected by this late-2016 F1, with average precision and log loss breaking ties. The selected setting is refitted before 2017, its final threshold is chosen on 2017, and 2018 is evaluated once within this benchmark. The 2018 cohort was already examined in earlier project work, so this remains exploratory test reuse.

The selected setting is `xgb_depth5_double`, with 441 rounds and final positive-class weight 2.0. Its development F1 is 0.4751, compared with 0.4733 for the expanded-model reference on the same late-2016 split.

## Same 2018 cohort

| Metric | Expanded model | XGBoost |
| --- | ---: | ---: |
| Precision | 26.60% | 27.35% |
| Recall | 66.19% | 62.71% |
| F1 | 0.3795 | 0.3809 |
| Average precision | 0.3162 | 0.3139 |
| ROC-AUC | 0.7195 | 0.7193 |
| Brier score | 0.1230 | 0.1447 |
| False positives | 16,210 | 14,782 |
| Defaults found | 5,874 | 5,565 |

The F1 difference is +0.0014, with paired 95% percentile bootstrap interval [-0.0012, +0.0040]. It includes zero. Relative to the reference, XGBoost avoids 1,428 false positives and misses 309 additional defaults. The threshold was selected on validation data, not the test outcomes.

Paired uncertainty uses 1,000 individual-loan resamples with seed 5106. Intervals condition on the fixed fitted models and thresholds; they exclude training/search uncertainty, dependence between loans, and effects of reusing the test set.

## Calibration and recommendation

XGBoost predicts 29.50% average risk versus 15.76% observed defaults. Calibration-in-the-large is -0.912 and slope 0.935. Positive-class weighting affects probability estimates. These diagnostic fits do not recalibrate probabilities on the test set.

This benchmark does not justify replacing the expanded model. It supports a precision/recall trade-off, not a substantial improvement in default identification. Additional model searches on this same test set would remain exploratory; independent data are needed for a new confirmatory assessment.

## Reproduction and verification

```bash
python -m pip install -r requirements-xgboost.txt
# macOS also requires: brew install libomp
python analysis/12_xgboost_benchmark.py
```

The run used Python 3.12.6 and XGBoost 3.4.1. The native categorical implementation and early-stopping interface follow the official API: https://xgboost.readthedocs.io/en/stable/python/python_api.html. XGBoost is an optional dependency; the original requirements lock remains unchanged.

All 41 synthetic tests passed, including class-weight calculation, F1 threshold selection, cached-prediction identity/outcome/date checks and native-category/model-reload tests. The real benchmark also verified reference prediction alignment and that saving/loading the fitted model reproduces predictions. Model files and loan-level predictions remain in ignored artifacts/ and data/cleaned/xgboost_handoff directories.

All six development results, selected parameters, source hashes, software versions, paired differences and calibration diagnostics are retained under outputs/experiments/xgboost_benchmark. Original predictive code, Ivy's code/results and the expanded-model results were preserved. This is a benchmark note, not the final group report. Code and execution were assisted by Codex.
