"""Compare one fixed ordinal-subgrade/credit-ratio challenger on the frozen 60/20/20 split."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.joint_accuracy_f1 import OrdinalCreditEncoder, joint_rule, operating_curve, paired_intervals
from src.paths import CLEANED_DATA_DIR, PROJECT_DIR
from src.predictive_extension import READ_COLUMNS
from src.predictive_improvement import threshold_metrics
from src.predictive_modelling import binary_metrics
from src.preprocessing_experiment import write_predictions
from src.proportional_split import frozen_602020
from src.xgboost_benchmark import check_handoff, feature_types, make_model

EXPERIMENTS = PROJECT_DIR / "outputs/experiments"
REFERENCE = "rerun_current_xgb"
CHALLENGER = "ordinal_subgrade_credit_ratios"
SPEC = {"max_depth": 5, "weight": "double"}


def reference_scores(part, frame):
    path = CLEANED_DATA_DIR / "preprocessing_ensemble_602020_handoff" / f"{part}.csv.gz"
    handoff = pd.read_csv(path, float_precision="round_trip")
    if not handoff.model.eq(REFERENCE).all():
        raise ValueError("Unexpected reference model in saved predictions")
    return check_handoff(handoff, frame)


def evaluate(frame, scores, threshold, model, rule):
    row = threshold_metrics(frame.default, scores, threshold)
    accuracy = (row["tp"] + row["tn"]) / len(frame)
    return dict(model=model, rule=rule, accuracy=accuracy, **row, **binary_metrics(frame.default, scores))


def fit_challenger(train):
    cutoff = train.issue_date.iloc[int(.9 * len(train))]
    early, stop = train.loc[train.issue_date < cutoff], train.loc[train.issue_date >= cutoff]
    encoder = OrdinalCreditEncoder().fit(early)
    model = make_model(SPEC, 800, feature_types(encoder.mask), early.default, early_stopping=True)
    model.fit(encoder.transform(early), early.default,
              eval_set=[(encoder.transform(stop), stop.default)], verbose=False)
    rounds = model.best_iteration + 1
    encoder = OrdinalCreditEncoder().fit(train)
    model = make_model(SPEC, rounds, feature_types(encoder.mask), train.default)
    model.fit(encoder.transform(train), train.default)
    return encoder, model, rounds, cutoff


def main():
    out = EXPERIMENTS / "joint_accuracy_f1_602020"
    out.mkdir(parents=True, exist_ok=True)
    contract = EXPERIMENTS / "split_602020/frozen_split_contract.json"
    if not contract.is_file():
        raise FileNotFoundError("The existing split contract is required")
    frame = pd.read_csv(CLEANED_DATA_DIR / "clean_v1/lending_club_clean.csv.gz",
                        usecols=READ_COLUMNS, low_memory=False)
    train, valid, test = frozen_602020(frame, contract)
    del frame
    ref_manifest = json.loads((EXPERIMENTS / "preprocessing_ensemble_602020/experiment_manifest.json").read_text())
    if ref_manifest["selected_on_validation"] != REFERENCE:
        raise ValueError("Expected the retained XGBoost reference")
    ref_threshold = ref_manifest["threshold"]
    ref_val = reference_scores("validation", valid)
    baseline = evaluate(valid, ref_val, ref_threshold, REFERENCE, "existing_maximum_f1")
    ref_curve = operating_curve(valid.default, ref_val)
    if not np.isclose(ref_curve.f1.max(), baseline["f1"], atol=1e-12, rtol=0):
        raise ValueError("Reference no longer reproduces its validation maximum F1")
    print("Frozen cohort and current validation maximum F1 verified. Fitting challenger.", flush=True)
    encoder, model, rounds, cutoff = fit_challenger(train)
    p = model.predict_proba(encoder.transform(valid))[:, 1].astype(float)
    curve = operating_curve(valid.default, p)
    joint = joint_rule(curve, baseline)
    candidate = evaluate(valid, p, joint.threshold, CHALLENGER, "joint_accuracy_f1")
    peak = curve.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    # Quantify a threshold-only alternative on validation. It cannot exceed baseline F1.
    higher_acc = ref_curve.loc[ref_curve.accuracy >= baseline["accuracy"] + .02]
    tradeoff = higher_acc.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    rows = [baseline, evaluate(valid, ref_val, tradeoff.threshold, REFERENCE, "accuracy_plus_2pp"),
            evaluate(valid, p, peak.threshold, CHALLENGER, "maximum_f1"), candidate]
    pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
    passed = all(candidate[k] > baseline[k] for k in ("accuracy", "f1"))
    choice = dict(selected=CHALLENGER if passed else REFERENCE,
                  threshold=float(joint.threshold if passed else ref_threshold),
                  challenger_rounds=int(rounds), inner_stopping_from=cutoff,
                  predictor_columns=encoder.columns, validation_joint_gain=float(joint.min_gain))
    (out / "frozen_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(pd.DataFrame(rows)[["model", "rule", "accuracy", "f1", "precision", "recall"]], flush=True)

    ref_test = reference_scores("test", test)
    ref_result = evaluate(test, ref_test, ref_threshold, REFERENCE, "existing_maximum_f1")
    test_rows = [ref_result]
    if passed:
        art = PROJECT_DIR / "artifacts/joint_accuracy_f1_602020"
        art.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(encoder=encoder, model=model, threshold=choice["threshold"]), art / "selected.joblib")
        selected = joblib.load(art / "selected.joblib")
        if not np.array_equal(p, selected["model"].predict_proba(selected["encoder"].transform(valid))[:, 1]):
            raise ValueError("Reloaded challenger failed to reproduce validation predictions")
        test_scores = selected["model"].predict_proba(selected["encoder"].transform(test))[:, 1].astype(float)
        handoff = CLEANED_DATA_DIR / "joint_accuracy_f1_602020_handoff"
        handoff.mkdir(parents=True, exist_ok=True)
        for part, cohort, scores in [("validation", valid, p), ("test", test, test_scores)]:
            path = handoff / f"{part}.csv.gz"
            write_predictions(cohort, scores, path, CHALLENGER)
            restored = check_handoff(pd.read_csv(path, float_precision="round_trip"), cohort)
            if not np.array_equal(scores, restored):
                raise ValueError(f"{part} handoff lost score precision")
        result = evaluate(test, test_scores, choice["threshold"], CHALLENGER, "joint_accuracy_f1")
        intervals = paired_intervals(test.default, ref_test >= ref_threshold, test_scores >= choice["threshold"])
        for metric, (lo, hi) in zip(("accuracy", "f1"), intervals):
            result[f"{metric}_delta"] = result[metric] - ref_result[metric]
            result[f"{metric}_delta_low"], result[f"{metric}_delta_high"] = lo, hi
        test_rows.append(result)
    pd.DataFrame(test_rows).to_csv(out / "test.csv", index=False)
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
