# Chronological 60/20/20 experiment

The original study and Ivy outputs remain preserved. This is a separate exploratory split comparison, not the final group report.

Earliest 60% of resolved loans train the models; next 20% choose the F1 threshold and model; latest 20% assess performance. Rounded row counts are exhaustive and disjoint. Monthly date ties are ordered by a fixed hash of loan ID and then ID, without using outcomes. Boundary months may span adjacent partitions; this is not a strict whole-month holdout.

partition,rows,fraction,minimum_issue_date,maximum_issue_date,default_count,default_rate
train,808859,0.5999997032858863,2007-06-01,2015-12-01,148778,0.1839356426769066
validation,269620,0.20000014835705687,2015-12-01,2016-10-01,61824,0.22930049699577182
test,269620,0.20000014835705687,2016-10-01,2018-12-01,58758,0.21792893702247607


Both models use the same 27 expanded predictors and training-fitted preprocessing. XGBoost depth 5 and positive weight 2 were fixed before this run; training-only early stopping chose 458 rounds. HGB uses the prior expanded specification and its internal training-only stopping subset. No outer validation/test records fit either encoder or model.

Model chosen by validation F1 (AP then lower log loss break ties): **xgb_depth5_double**. Thresholds maximize validation F1. Test results did not determine this choice.

## Test results

candidate,selected_on_validation,roc_auc,pr_auc,brier_score,log_loss,default_rate,threshold,tn,fp,fn,tp,precision,recall,f1,flagged_fraction
expanded_grade_31,False,0.7100156935849385,0.3898814016868566,0.15595795874032953,0.4805979420297246,0.21792893702247607,0.1766723781729331,133626,77236,19338,39420,0.33791660951858454,0.6708873685285408,0.4494510130320271,0.43266819968845044
xgb_depth5_double,True,0.7111716459424926,0.39087256032794543,0.1605266567998843,0.4907405518888619,0.21792893702247607,0.29415860772132874,131903,78959,18653,40105,0.3368356514143654,0.682545355526056,0.45106904657466457,0.4415992878866553


Do not directly attribute changes from the previous split to a better model: the test period, sample size and default prevalence changed. This experiment repartitions data already used in previous searches, and even the architecture choices were informed by overlapping cohorts. It supplies no fresh confirmatory test. Resolved-loan selection and incomplete follow-up limitations remain.

Reproduce: install requirements-xgboost.txt (macOS needs libomp), then run `python analysis/13_split_602020.py`. Aggregate results are in outputs/experiments/split_602020; loan-level handoffs remain ignored under data/cleaned. Earlier dashboard results retain their original split. Code and execution assisted by Codex.
