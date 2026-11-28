# Completed preprocessing, ensemble and full refit experiment

The current XGBoost model is retained. The calibrated native blend improved several validation metrics, but did not improve the registered F1 objective. This is an experiment note, not the final group report.

## Code review and fixes

Two reproducible issues were corrected:

1. An entirely missing training numerical column could be dropped by the median imputer, breaking the expected feature layout. The imputer now preserves empty columns. The regression test reproduces the previous failure and checks the repaired layout.
2. Default float32 CSV formatting can move a saved score across an exact stored threshold. The old validation handoff changes one decision when reloaded as float64 and assessed at the original threshold: F1 0.473969 instead of 0.473978. New exports write float64 scores with 17 significant digits and are checked with round-trip parsing. Script 13's future exports were also corrected; historical files were preserved.

Neither issue explains the overall low F1. Both original models were refitted from the frozen cleaned training cohort; all four validation/test prediction comparisons and independent sklearn precision/recall/F1 checks passed. XGBoost prediction differences from the old CSV were at most 2.981e-8, consistent with its previous serialization precision; HGB differences were below 1e-16. The reported reference confusion matrices were reproduced exactly.

The review checked target polarity, confusion-matrix orientation, inclusive/tied thresholds, row/ID/outcome alignment, fitted transformations, category codes, stopping data, class weighting, prediction exports and model reloads. It found no demonstrated F1-calculation or outer-test fitting error in the reviewed model path. This is not a guarantee that the entire repository is bug-free.

An initial attempt at the new calibration split encountered uneven issue-month volumes, leaving an empty earlier block. The check stopped the run before the comparison started. Boundary selection was corrected to choose the nonempty whole-month boundary closest to half the rows, a regression test was added, and the complete successful run restarted from the reference fits.

## Experiment completed

The same frozen assignments remain in force: 808,859 training loans, 269,620 validation loans and 269,620 test loans. No raw cleaning or inference model was rerun or changed beyond the shared imputer robustness fix; the model refits begin from the unchanged cleaned cohort.

Three preparations were compared with fixed XGBoost/HGB settings on two chronological folds inside training: current numerical transformations (27 predictors), native NaNs/uncapped engineered values (27), and native treatment plus 18 fixed missingness indicators (45). Existing category handling, grade proxy and source fields were retained. No external data or new raw predictors were added.

The models met the predefined complementarity gate on both folds. Decision disagreement was approximately 4.2–5.6%; each model uniquely classified some loans correctly. A single equal calibrated blend per preparation was therefore eligible. Calibration used separate out-of-sample blocks entirely inside training, followed by later threshold and score blocks.

All 12 family/preparation fold fits and 18 pipeline/fold comparisons completed. The training-only winner was the native blend, mean forward F1 0.450997 versus 0.449453 for current-preparation XGBoost. Its full-training constituent models used 643 XGBoost rounds and 326 HGB rounds; final calibration maps used pooled out-of-sample training scores. These candidate settings differ from the retained current reference, which reproduced its original 458 rounds and threshold 0.29415860772132874.

## Validation selection

| Metric | Fully refitted current XGBoost | Calibrated native blend |
| --- | ---: | ---: |
| Precision | 36.39% | 36.72% |
| Recall | 67.94% | 66.79% |
| F1 | 0.473978 | 0.473877 |
| Average precision | 0.436095 | 0.436529 |
| ROC-AUC | 0.724901 | 0.725214 |
| Brier score, lower is better | 0.160169 | 0.156796 |
| False positives | 73,406 | 71,157 |
| Missed defaults | 19,822 | 20,532 |

The blend avoided 2,249 validation false alarms but missed 710 additional defaults. Its probability accuracy and ranking metrics improved slightly, while F1 decreased by 0.000100. The models are very close on this measure; the registered empirical F1 rule retained the current XGBoost. No new selection rule was introduced after seeing the result.

## Retained test result

The rejected blend was not assessed on test. The refitted selected reference retains test precision 33.68%, recall 68.25%, F1 0.451069 and average precision 0.390873.

| Actual / Predicted | No default | Default |
| --- | ---: | ---: |
| No default | 131,903 | 78,959 |
| Default | 18,653 | 40,105 |

The paired F1 difference is zero because the selected model is the same reference. Its degenerate interval is not evidence of zero population uncertainty. The experiment supplies no convincing F1 upgrade and does not fix the false-alarm limitation. The blend is a validation-only alternative if priorities later change; no future/test precision claim follows from these validation figures.

## Verification and reproduction

All 56 tests passed, including empty-feature preservation, native missing/extreme values, unseen categories, outcome-independent transformations, tied-threshold metric parity, monotonic calibration, calibration-block separation, uneven-month handling and exact score export. Full reference fits and source/split hashes were verified. The selected fitted bundle was saved/reloaded and reproduced predictions exactly; exported scores also round-tripped exactly. Ivy's original code and all 13 original output hashes remain intact.

Install `requirements-preprocessing-experiment.txt`, then run `python analysis/16_preprocessing_ensemble_rerun.py`. The protocol is in `preprocessing_ensemble_protocol.md`. Aggregate tables and the completed manifest are under `outputs/experiments/preprocessing_ensemble_602020`; fitted bundles and loan-level handoffs remain ignored locally under artifacts/ and data/cleaned/.

The same test has already been examined across earlier experiments, so findings remain exploratory. The accepted/resolved-loan selection, unequal follow-up, vintage drift and conditional uncertainty limits still apply. Human review and an independently repeated complete successful search are not claimed. The dashboard remains unchanged. Functional AI assistance is acknowledged in the AI-use log; no final group report was created.
