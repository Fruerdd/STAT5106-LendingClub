"""Assess high-recall cutoffs for missed defaults on the frozen loan cohort."""
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, PROJECT_DIR
from src.precision_focus import precision_at_recall_floor
from src.predictive_improvement import threshold_metrics
from src.proportional_split import frozen_602020
from src.xgboost_benchmark import check_handoff

EXPERIMENTS = PROJECT_DIR / "outputs/experiments"
OUT = EXPERIMENTS / "default_recall_602020"
SELECTED_RULE = "maximum_f1"


def load_scores(part, cohort, model):
    path = CLEANED_DATA_DIR / "temporal_f1_602020_handoff" / f"{part}.csv.gz"
    scores = pd.read_csv(path, float_precision="round_trip")
    if not scores.model.eq(model).all():
        raise ValueError("Unexpected model in the prediction handoff")
    return check_handoff(scores, cohort)


def results(cohort, scores, rules):
    rows = []
    for name, threshold in rules.items():
        row = threshold_metrics(cohort.default, scores, threshold)
        rows.append(dict(rule=name, accuracy=(row["tp"] + row["tn"]) / len(cohort), **row))
    return pd.DataFrame(rows)


def main():
    contract = EXPERIMENTS / "split_602020/frozen_split_contract.json"
    if not contract.is_file():
        raise FileNotFoundError("The existing frozen split contract is required")
    current = json.loads((EXPERIMENTS / "precision_focus_602020/frozen_rule.json").read_text())
    f1_choice = json.loads((EXPERIMENTS / "temporal_f1_602020/choice.json").read_text())
    if current["rule"] != "ensemble_precision" or current["model"] != f1_choice["selected"]:
        raise ValueError("Both rules must use the same ensemble scores")
    data = pd.read_csv(CLEANED_DATA_DIR / "clean_v1/lending_club_clean.csv.gz",
                       usecols=["id", "issue_date", "default"])
    _, validation, test = frozen_602020(data, contract)
    validation_scores = load_scores("validation", validation, current["model"])
    targets = {"recall_95": .95, "recall_99": .99, "recall_995_buffered": .995,
               "recall_100_validation": 1.}
    rules = {SELECTED_RULE: f1_choice["threshold"], "previous_rule": current["threshold"]}
    for name, target in targets.items():
        rules[name] = precision_at_recall_floor(validation.default, validation_scores, target)
    rules["flag_all"] = 0.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_rules.json").write_text(json.dumps({
        "model": current["model"], "selected_rule": SELECTED_RULE,
        "validation_recall_targets": targets,
        "thresholds": rules
    }, indent=2) + "\n")

    results(validation, validation_scores, rules).to_csv(OUT / "validation.csv", index=False)
    test_scores = load_scores("test", test, current["model"])
    test_results = results(test, test_scores, rules)
    test_results.to_csv(OUT / "test.csv", index=False)
    print(test_results.to_string(index=False))


if __name__ == "__main__":
    main()
