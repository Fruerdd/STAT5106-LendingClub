"""Compare the training majority-class baseline with the frozen default model."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, PROJECT_DIR
from src.predictive_improvement import threshold_metrics
from src.proportional_split import frozen_602020
from src.xgboost_benchmark import check_handoff

EXPERIMENTS = PROJECT_DIR / "outputs/experiments"


def metric_row(part, name, cohort, scores, threshold):
    metrics = threshold_metrics(cohort.default, scores, threshold)
    accuracy = (metrics["tn"] + metrics["tp"]) / len(cohort)
    return dict(partition=part, model=name, loans=len(cohort), accuracy=accuracy, **metrics)


def main():
    contract = EXPERIMENTS / "split_602020/frozen_split_contract.json"
    if not contract.is_file():
        raise FileNotFoundError("The existing frozen split contract is required")
    frame = pd.read_csv(CLEANED_DATA_DIR / "clean_v1/lending_club_clean.csv.gz",
                        usecols=["id", "issue_date", "default"])
    train, validation, test = frozen_602020(frame, contract)

    # The constant feature satisfies sklearn's input shape and has no predictive role.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train), 1), dtype=np.int8), train.default)
    majority_label = int(baseline.predict([[0]])[0])
    label_name = {0: "fully_paid", 1: "default"}[majority_label]

    policy = json.loads((EXPERIMENTS / "default_recall_602020/frozen_rules.json").read_text())
    active_rule = policy["selected_rule"]
    rows = []
    for part, cohort in [("validation", validation), ("test", test)]:
        path = CLEANED_DATA_DIR / "temporal_f1_602020_handoff" / f"{part}.csv.gz"
        predictions = pd.read_csv(path, float_precision="round_trip")
        if not predictions.model.eq(policy["model"]).all():
            raise ValueError("Prediction handoff does not match the selected model")
        scores = check_handoff(predictions, cohort)
        baseline_predictions = baseline.predict(np.zeros((len(cohort), 1), dtype=np.int8))
        rows.append(metric_row(part, f"always_{label_name}", cohort, baseline_predictions, .5))
        rows.append(metric_row(part, active_rule, cohort, scores, policy["thresholds"][active_rule]))

    out = EXPERIMENTS / "majority_baseline_602020"
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "comparison.csv", index=False)
    print(f"Training majority: {label_name} ({int(train.default.eq(majority_label).sum()):,}/{len(train):,})")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
