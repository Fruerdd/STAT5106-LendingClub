"""Refit both members of the frozen best-F1 ensemble from training data."""
import json
import sys
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.joint_accuracy_f1 import OrdinalCreditEncoder
from src.paths import CLEANED_DATA_DIR, PROJECT_DIR
from src.predictive_extension import READ_COLUMNS
from src.proportional_split import frozen_602020
from src.temporal_f1 import matrices, member_names, predict
from src.xgboost_benchmark import check_handoff


def main():
    experiment = import_module("analysis.18_temporal_f1")
    contract = experiment.EXPERIMENTS / "split_602020/frozen_split_contract.json"
    if not contract.is_file():
        raise FileNotFoundError("The existing frozen split contract is required")
    choice = json.loads((experiment.OUT / "choice.json").read_text())
    policy = json.loads((experiment.EXPERIMENTS / "default_recall_602020/frozen_rules.json").read_text())
    if (policy["selected_rule"] != "maximum_f1" or policy["model"] != choice["selected"]
            or policy["thresholds"]["maximum_f1"] != choice["threshold"]):
        raise ValueError("The active rule does not match the frozen best-F1 ensemble")

    frame = pd.read_csv(CLEANED_DATA_DIR / "clean_v1/lending_club_clean.csv.gz",
                        usecols=READ_COLUMNS, low_memory=False)
    train, valid, test = frozen_602020(frame, contract)
    del frame
    encoder = OrdinalCreditEncoder().fit(train)
    inputs = matrices(encoder, train)
    models = {}
    for name in member_names(choice["selected"]):
        print(f"Refitting {name} on {len(train):,} training loans", flush=True)
        models[name] = experiment.fit_member(name, train, inputs, encoder, choice["rounds"])
    bundle = dict(encoder=encoder, models=models, candidate=choice["selected"], threshold=choice["threshold"])
    del inputs

    out = experiment.EXPERIMENTS / "best_f1_refit_602020"
    artifacts = PROJECT_DIR / "artifacts" / out.name
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / "refitted.joblib"
    joblib.dump(bundle, path)
    reloaded = joblib.load(path)

    rows = []
    for part, cohort in [("validation", valid), ("test", test)]:
        file = CLEANED_DATA_DIR / f"{experiment.OUT.name}_handoff" / f"{part}.csv.gz"
        handoff = pd.read_csv(file, float_precision="round_trip")
        if not handoff.model.eq(choice["selected"]).all():
            raise ValueError("Reference handoff has an unexpected model")
        expected = check_handoff(handoff, cohort)
        scores = predict(reloaded, cohort)
        summary = experiment.metrics(cohort, scores, choice["threshold"], choice["selected"],
                                     partition=part, loans=len(cohort),
                                     exact_score_match=bool(np.array_equal(scores, expected)),
                                     max_score_difference=float(np.max(np.abs(scores - expected))),
                                     changed_decisions=int(np.count_nonzero(
                                         (scores >= choice["threshold"]) != (expected >= choice["threshold"]))))
        rows.append(summary)
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
