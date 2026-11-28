"""Choose one F1 challenger in training, apply a validation accuracy floor, freeze before test."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.joint_accuracy_f1 import OrdinalCreditEncoder, paired_intervals
from src.paths import CLEANED_DATA_DIR, PROJECT_DIR
from src.predictive_extension import READ_COLUMNS
from src.predictive_improvement import threshold_metrics
from src.predictive_modelling import binary_metrics
from src.preprocessing_experiment import write_predictions
from src.proportional_split import frozen_602020
from src.scoped_tuning import training_folds
from src.temporal_f1 import (BLEND, CANDIDATES, REFERENCE, SETTINGS, f1_rule, loan_terms,
                             matrices, member_names, predict, recency_weights)
from src.xgboost_benchmark import check_handoff, feature_types, make_model

EXPERIMENTS = PROJECT_DIR / "outputs/experiments"
PREVIOUS = "joint_accuracy_f1_602020"
OUT = EXPERIMENTS / "temporal_f1_602020"
SPEC = {"max_depth": 5, "weight": "double"}


def metrics(frame, p, threshold, model, **context):
    row = threshold_metrics(frame.default, p, threshold)
    return dict(**context, model=model, accuracy=(row["tn"] + row["tp"]) / len(frame),
                **row, **binary_metrics(frame.default, p))


def inner_rounds(early, stop):
    enc = OrdinalCreditEncoder().fit(early)
    model = make_model(SPEC, 800, feature_types(enc.mask), early.default, early_stopping=True)
    model.fit(enc.transform(early), early.default,
              eval_set=[(enc.transform(stop), stop.default)], verbose=False)
    return model.best_iteration + 1


def fit_member(name, fit, inputs, encoder, rounds):
    extra, recent = SETTINGS[name]
    mask = encoder.mask + [False] * (inputs[1].shape[1] - inputs[0].shape[1]) if extra else encoder.mask
    model = make_model(SPEC, rounds, feature_types(mask), fit.default)
    model.fit(inputs[int(extra)], fit.default, sample_weight=recency_weights(fit.issue_date) if recent else None)
    return model


def forward_comparison(train):
    rows = []
    for fold, fit, cut, later, early, stop in training_folds(train):
        rounds = inner_rounds(early, stop)
        enc = OrdinalCreditEncoder().fit(fit)
        xt, xc, xl = (matrices(enc, frame) for frame in (fit, cut, later))
        cut_scores, later_scores = {}, {}
        for name, (extra, _) in SETTINGS.items():
            model = fit_member(name, fit, xt, enc, rounds)
            cut_scores[name] = model.predict_proba(xc[int(extra)])[:, 1].astype(float)
            later_scores[name] = model.predict_proba(xl[int(extra)])[:, 1].astype(float)
        for scores in (cut_scores, later_scores):
            scores[BLEND] = (scores[REFERENCE] + scores["combined"]) / 2.
        base = f1_rule(cut.default, cut_scores[REFERENCE])
        for name in (REFERENCE, *CANDIDATES):
            rule = f1_rule(cut.default, cut_scores[name], base.accuracy)
            rows.append(metrics(later, later_scores[name], rule.threshold, name, fold=fold,
                                rounds=rounds, fit_n=len(fit), threshold_n=len(cut), score_n=len(later),
                                threshold_block_accuracy_floor=base.accuracy))
        print(f"{fold} completed with {rounds} rounds", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "forward_folds.csv", index=False)
    summary = (folds.groupby("model", as_index=False)
               .agg(mean_f1=("f1", "mean"), mean_ap=("pr_auc", "mean"), mean_accuracy=("accuracy", "mean"))
               .sort_values(["mean_f1", "mean_ap", "model"], ascending=[False, False, True]))
    summary.to_csv(OUT / "training_ranking.csv", index=False)
    return summary.loc[summary.model.isin(CANDIDATES), "model"].iloc[0]


def reference_scores(part, frame):
    path = CLEANED_DATA_DIR / f"{PREVIOUS}_handoff" / f"{part}.csv.gz"
    df = pd.read_csv(path, float_precision="round_trip")
    if not df.model.eq(REFERENCE).all():
        raise ValueError("Reference handoff contains another model")
    return check_handoff(df, frame)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = EXPERIMENTS / "split_602020/frozen_split_contract.json"
    if not contract.is_file():
        raise FileNotFoundError("Existing frozen split contract is required")
    previous = json.loads((EXPERIMENTS / PREVIOUS / "frozen_choice.json").read_text())
    if previous["selected"] != REFERENCE:
        raise ValueError("Expected the ordinal-subgrade model as the incumbent")
    frame = pd.read_csv(CLEANED_DATA_DIR / "clean_v1/lending_club_clean.csv.gz", usecols=READ_COLUMNS,
                        low_memory=False)
    train, valid, test = frozen_602020(frame, contract)
    del frame
    name = forward_comparison(train)
    print(f"Training selected {name}", flush=True)
    path = OUT / "choice.json"
    record = dict(training_winner=name, candidates=CANDIDATES, half_life_months=12,
                  rounds=previous["challenger_rounds"], reference=REFERENCE)
    path.write_text(json.dumps(record, indent=2) + "\n")
    base = joblib.load(PROJECT_DIR / "artifacts" / PREVIOUS / "selected.joblib")
    enc = base["encoder"]
    p0 = reference_scores("validation", valid)
    if (enc.columns != previous["predictor_columns"] or base["threshold"] != previous["threshold"]
            or not np.array_equal(p0, base["model"].predict_proba(enc.transform(valid))[:, 1])):
        raise ValueError("Previous fitted bundle and frozen validation reference disagree")
    fit_inputs = matrices(enc, train)
    models = {n: base["model"] if n == REFERENCE else fit_member(n, train, fit_inputs, enc, record["rounds"])
              for n in member_names(name)}
    bundle = dict(encoder=enc, models=models, candidate=name)
    p = predict(bundle, valid)
    old = metrics(valid, p0, base["threshold"], REFERENCE, rule="frozen")
    peak = f1_rule(valid.default, p)
    rule = f1_rule(valid.default, p, old["accuracy"])
    new = metrics(valid, p, rule.threshold, name, rule="accuracy_floor")
    validation = pd.DataFrame([old, metrics(valid, p, peak.threshold, name, rule="maximum_f1"), new])
    validation.to_csv(OUT / "validation.csv", index=False)
    passed = new["f1"] > old["f1"] and new["accuracy"] >= old["accuracy"]
    record.update(selected=name if passed else REFERENCE, validation_accuracy_floor=old["accuracy"],
                  threshold=float(rule.threshold if passed else base["threshold"]),
                  extra_features=list(loan_terms(train.iloc[:0]).columns) if any(SETTINGS[n][0] for n in models) else [],
                  challenger_test_assessed=bool(passed))
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(validation[["model", "rule", "accuracy", "f1"]].to_string(index=False), flush=True)
    if not passed:
        print("Retaining the incumbent. No challenger test scores calculated.", flush=True)
        return
    bundle["threshold"] = record["threshold"]
    artifact = PROJECT_DIR / "artifacts" / OUT.name
    artifact.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact / "selected.joblib")
    restored = joblib.load(artifact / "selected.joblib")
    if not np.array_equal(p, predict(restored, valid)):
        raise ValueError("Reloaded bundle failed to reproduce validation scores")
    test_scores = predict(restored, test)
    ref_scores = reference_scores("test", test)
    ref = metrics(test, ref_scores, base["threshold"], REFERENCE)
    result = metrics(test, test_scores, record["threshold"], name)
    intervals = paired_intervals(test.default, ref_scores >= base["threshold"], test_scores >= record["threshold"])
    for metric, (lo, hi) in zip(("accuracy", "f1"), intervals):
        result[f"{metric}_delta"] = result[metric] - ref[metric]
        result[f"{metric}_delta_low"], result[f"{metric}_delta_high"] = lo, hi
    handoff = CLEANED_DATA_DIR / f"{OUT.name}_handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    for part, cohort, scores in [("validation", valid, p), ("test", test, test_scores)]:
        dest = handoff / f"{part}.csv.gz"
        write_predictions(cohort, scores, dest, name)
        if not np.array_equal(scores, check_handoff(pd.read_csv(dest, float_precision="round_trip"), cohort)):
            raise ValueError(f"{part} handoff differs from fitted predictions")
    pd.DataFrame([ref, result]).to_csv(OUT / "test.csv", index=False)
    print(f"Frozen test result: F1 {result['f1']:.6f}, accuracy {result['accuracy']:.6f}", flush=True)


if __name__ == "__main__":
    main()
