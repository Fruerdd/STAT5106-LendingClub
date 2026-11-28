# Scoped predictive improvement protocol

This is an exploratory implementation note, not the final group report. The research question stays the same: which borrower/loan characteristics predict resolved Lending Club default, and how accurately? The inference models, original Member 5 implementation by Ivy, original outputs and raw data are preserved.

The revised final-project guidelines require a dataset introduction, cleaning, visualization, statistical modelling, results, interpretation and limitations, with functional AI use acknowledged. The Week 9 ML notebook and methods supplement support training-fitted preprocessing, a common 60/20/20 comparison, validation-based decisions, and a frozen test assessment. Gradient boosting remains a comparison method for this research project; no lending deployment is proposed.

## Scope and audit

- Keep the exact 1,348,099-loan resolved-outcome cohort and frozen 808,859 / 269,620 / 269,620 assignments. Verify the existing hash contract before fitting.
- Use only sources already in the documented predictive feature extension. Audit all cleaned columns against the primary dictionary and distinguish extension permissions from primary inference permissions.
- Do not import additional raw fields, external data, text, post-outcome payment/recovery fields, updated FICO, or outcome-derived features. The audit does not establish the timing of other raw fields, so they are not added.
- Retain the existing 27-feature representation. Grade and subgrade are alternatives; neither is combined with interest rate. Each encoder, numeric transform and category map is fitted on its own training block.
- Report training/validation source missingness. Later validation/test diagnostics by vintage, grade, income verification and structural credit-history missingness describe errors; they do not select features or subgroup-specific models.

## Registered bounded search

Six fixed XGBoost specifications compare depths 3/5/7, positive weights 1/2, regularization and row/column sampling. One specification replaces grade with already-registered subgrade. The complete settings are in `src/scoped_tuning.py` and the experiment manifest, written before fitting. Learning rate is 0.05, maximum rounds 800, early-stopping patience 40, seed 5106 and CPU threads 2. No automatic search or adaptive expansion is used.

Use two forward folds inside the outer training 60%. Approximate row quantiles set whole-month boundaries:

| Fold | Fit | Threshold block | Score block |
| --- | --- | --- | --- |
| Forward 1 | Earliest 50% of training | Next 10% | Next 15% |
| Forward 2 | Earliest 70% of training | Next 10% | Last 20% |

For each fold, latest whole months covering approximately 10% of its fit block supply early stopping; earlier fit rows train the stopping model. Refit the complete fold fit block at that stopping round count. Maximize F1 on the subsequent threshold block, then compare the frozen rule on the later score block. Whole-month boundaries mean counts are approximate. The outer split remains exactly the locked assignment.

Rank candidates by mean forward F1, then mean AP, then lower mean log loss. Freeze one candidate and its median fold stopping-round count before looking at outer validation scores. Refit only on the full outer training 60%.

## Final selection and assessment

Compare this one candidate with cached current XGBoost on the same outer validation 20%. Each uses a maximum-F1 validation threshold. Choose higher validation F1, then AP, then lower log loss. Keep the cached current model if it wins; the training search is not a promise of an upgrade.

Write the chosen model and threshold to the manifest before scoring test. Assess only the frozen choice against the existing reference on the same test 20%. No rejected search candidate is scored on test. Validation-only precision-target/recall-target alternatives are descriptive and do not change this F1 objective.

Use 1,000 paired individual-loan bootstrap draws for the selected-versus-reference F1 difference. These intervals condition on fitted models and thresholds and omit training/search uncertainty and dependence between loans. Repeated historical test inspection also limits their interpretation.

All results remain exploratory: the test has already been examined, earlier model-family choices used overlapping records, and changing split did not create independent data. Keep resolved-outcome selection and unequal follow-up limitations explicit. Do not change the target, drop difficult test loans, rebalance the test set, or claim causal effects or a production lending tool.

## Run and artifacts

With the optional XGBoost environment installed, run:

```bash
python analysis/15_scoped_predictive_tuning.py
```

Script 13 must previously have generated the frozen contract and cached current-model validation/test predictions. Results are isolated under `outputs/experiments/scoped_tuning_602020`; loan-level handoffs and fitted artifacts remain ignored locally. The dashboard continues to show its earlier published experiments unless separately updated. This work is assisted by Codex; functional use and actual verification are recorded in the AI-use log.
