"""Compare a precision-focused cutoff and one alternative ranking on the frozen cohort."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, PROJECT_DIR
from src.precision_focus import precision_at_recall_floor
from src.predictive_extension import READ_COLUMNS
from src.predictive_improvement import threshold_metrics
from src.predictive_modelling import binary_metrics
from src.proportional_split import frozen_602020
from src.temporal_f1 import matrices, predict
from src.xgboost_benchmark import check_handoff

EXPERIMENTS = PROJECT_DIR / "outputs/experiments"
BASE = "temporal_f1_602020"
OUT = EXPERIMENTS / "precision_focus_602020"
RECALL_FLOOR = .60


def baseline_scores(part, cohort, model):
    data = pd.read_csv(CLEANED_DATA_DIR / f"{BASE}_handoff" / f"{part}.csv.gz",
                       float_precision="round_trip")
    if not data.model.eq(model).all():
        raise ValueError("Incumbent handoff has an unexpected model")
    return check_handoff(data, cohort)


def metrics(name, cohort, scores, cutoff):
    row = threshold_metrics(cohort.default, scores, cutoff)
    return dict(rule=name, **row, accuracy=(row["tp"] + row["tn"]) / len(cohort),
                **binary_metrics(cohort.default, scores))


def blended_scores(hgb, encoder, frame, incumbent):
    with threadpool_limits(limits=2):
        return (incumbent + hgb.predict_proba(matrices(encoder, frame)[1])[:, 1]) / 2.


def main():
    contract = EXPERIMENTS / "split_602020/frozen_split_contract.json"
    if not contract.is_file():
        raise FileNotFoundError("The existing frozen split contract is required")
    previous = json.loads((EXPERIMENTS / BASE / "choice.json").read_text())
    bundle = joblib.load(PROJECT_DIR / "artifacts" / BASE / "selected.joblib")
    if bundle["candidate"] != previous["selected"] or bundle["threshold"] != previous["threshold"]:
        raise ValueError("Incumbent bundle does not match the frozen choice")
    cleaned = pd.read_csv(CLEANED_DATA_DIR / "clean_v1/lending_club_clean.csv.gz",
                          usecols=READ_COLUMNS, low_memory=False)
    train, valid, test = frozen_602020(cleaned, contract)
    del cleaned
    OUT.mkdir(parents=True, exist_ok=True)

    incumbent = baseline_scores("validation", valid, previous["selected"])
    if not np.array_equal(incumbent, predict(bundle, valid)):
        raise ValueError("Incumbent bundle does not reproduce the validation handoff")
    cutoff = precision_at_recall_floor(valid.default, incumbent, RECALL_FLOOR)
    old = metrics("maximum_f1", valid, incumbent, bundle["threshold"])
    adjusted = metrics("ensemble_precision", valid, incumbent, cutoff)
    print("Frozen cohort verified. Fitting the alternative ranking on training loans.", flush=True)
    encoder = bundle["encoder"]
    x_train = matrices(encoder, train)[1]
    hgb = HistGradientBoostingClassifier(max_iter=351, max_leaf_nodes=31, min_samples_leaf=100,
                                         learning_rate=.05, l2_regularization=5.,
                                         categorical_features=encoder.mask + [False] * (x_train.shape[1] - len(encoder.mask)),
                                         early_stopping=False, random_state=5106)
    with threadpool_limits(limits=2):
        hgb.fit(x_train, train.default, sample_weight=1 + train.default.to_numpy())
    del x_train
    alternative = blended_scores(hgb, encoder, valid, incumbent)
    alt_cutoff = precision_at_recall_floor(valid.default, alternative, RECALL_FLOOR)
    alt = metrics("hgb_blend_precision", valid, alternative, alt_cutoff)
    # Require a better precision operating point and better ranking across cutoffs.
    use_blend = alt["precision"] > adjusted["precision"] and alt["pr_auc"] > adjusted["pr_auc"]
    choice = alt if use_blend else adjusted
    pd.DataFrame([old, adjusted, alt]).to_csv(OUT / "validation.csv", index=False)
    policy = dict(model=previous["selected"], rule=choice["rule"], threshold=choice["threshold"],
                  validation_recall_floor=RECALL_FLOOR)
    (OUT / "frozen_rule.json").write_text(json.dumps(policy, indent=2) + "\n")
    if use_blend:
        artifact = PROJECT_DIR / "artifacts" / OUT.name
        artifact.mkdir(parents=True, exist_ok=True)
        path = artifact / "hgb.joblib"
        joblib.dump(hgb, path)
        hgb = joblib.load(path)
        if not np.array_equal(alternative, blended_scores(hgb, encoder, valid, incumbent)):
            raise ValueError("Reloaded model differs from saved validation scores")

    incumbent = baseline_scores("test", test, previous["selected"])
    test_scores = blended_scores(hgb, encoder, test, incumbent) if use_blend else incumbent
    result = [metrics("maximum_f1", test, incumbent, bundle["threshold"]),
              metrics(choice["rule"], test, test_scores, choice["threshold"])]
    pd.DataFrame(result).to_csv(OUT / "test.csv", index=False)
    print(pd.DataFrame(result)[["rule", "accuracy", "precision", "recall", "f1", "fp", "fn"]], flush=True)


if __name__ == "__main__":
    main()
