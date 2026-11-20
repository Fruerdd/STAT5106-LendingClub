"""Pavel: fair out-of-time predictive modelling and logistic interpretation.

Models use identical loans, approved Model C predictors, and training-fitted preprocessing.
The 2017 validation cohort selects hyperparameters and the final model; the 2018 cohort is
transformed and evaluated only after selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import CLEANED_DATA_DIR, PROJECT_DIR, TABLES_DIR
from src.predictive_modelling import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TrainOnlyPreprocessor,
    binary_metrics,
    engineer_features,
    logistic_interpretation,
    prespecified_candidates,
    temporal_partitions,
    validate_input,
)

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
DEFAULT_FEATURE_SPEC = TABLES_DIR / "feature_specification.csv"
DEFAULT_REPORT = PROJECT_DIR / "report" / "pavel_modelling_report.md"
MODEL_FAMILIES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
METRICS = ("roc_auc", "pr_auc", "brier_score", "log_loss", "default_rate")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")


def validate_feature_specification(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Approved feature specification not found: {path}. Complete Member 3's handoff first."
        )
    specification = pd.read_csv(path)
    required_columns = {"engineered_feature", "model_c"}
    if not required_columns.issubset(specification.columns):
        raise ValueError("Feature specification lacks engineered_feature or model_c")
    approved = set(
        specification.loc[specification["model_c"].eq("yes"), "engineered_feature"]
    )
    approved_predictors = approved - {"default", "issue_year"}
    expected = {*NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    if approved_predictors != expected:
        missing = sorted(expected - approved_predictors)
        extra = sorted(approved_predictors - expected)
        raise ValueError(
            f"Predictive Model C contract mismatch; missing={missing}, unexpected={extra}"
        )


def _partition_summary(
    partitions: dict[str, pd.DataFrame],
    validation_cutoff: str,
    test_cutoff: str,
) -> pd.DataFrame:
    definitions = {
        "train": f"issue_date < {validation_cutoff}",
        "validation": f"{validation_cutoff} <= issue_date < {test_cutoff}",
        "test": f"issue_date >= {test_cutoff}",
    }
    rows = []
    for name, frame in partitions.items():
        dates = pd.to_datetime(frame["issue_date"])
        rows.append(
            {
                "partition": name,
                "rule": definitions[name],
                "minimum_issue_date": dates.min().strftime("%Y-%m-%d"),
                "maximum_issue_date": dates.max().strftime("%Y-%m-%d"),
                "rows": len(frame),
                "default_count": int(frame["default"].sum()),
                "default_rate": float(frame["default"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _fixed_settings(family: str) -> dict[str, Any]:
    if family == "logistic_regression":
        return {"solver": "lbfgs", "max_iter": 500, "tolerance": 1e-6}
    if family == "random_forest":
        return {"n_estimators": 200, "max_features": "sqrt", "n_jobs": -1}
    return {
        "learning_rate": 0.08,
        "max_iter": 300,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
    }


def _training_iterations(model, family: str) -> int | None:
    if family == "logistic_regression":
        return int(model.n_iter_[0])
    if family == "hist_gradient_boosting":
        return int(model.n_iter_)
    return None


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    path: Path,
    split: pd.DataFrame,
    comparison: pd.DataFrame,
    interpretation: pd.DataFrame,
    selected_model: str,
    validation_cutoff: str,
    test_cutoff: str,
) -> None:
    selected_rows = comparison.loc[comparison["selected_within_family"]].copy()
    report_lines = [
        "# Pavel — Predictive Modelling Report",
        "",
        "## Design and equal comparison",
        "",
        "All candidates use the same resolved-outcome loans, approved Model C application-time "
        "predictors, temporal partitions, random seed, and training-fitted preprocessing. "
        "Interest rate is included; grade, issue year, identifiers, raw dates, loan status, and "
        "post-outcome fields are excluded. No observations are resampled or removed for "
        "preprocessing. Class weighting is evaluated as a prespecified candidate setting.",
        "",
        f"Training uses issue dates before {validation_cutoff}; validation uses "
        f"{validation_cutoff} through the day before {test_cutoff}; testing uses {test_cutoff} "
        "onward. Hyperparameters and the final model minimize validation log loss. Test data "
        "do not enter fitting, preprocessing, selection, or tuning. Test metrics are calculated "
        "only for the selected candidate from each model family.",
        "",
        "| Partition | Rows | Defaults | Default rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in split.itertuples(index=False):
        report_lines.append(
            f"| {row.partition} | {row.rows:,} | {row.default_count:,} | {row.default_rate:.2%} |"
        )
    report_lines.extend(
        [
            "",
            "## Model comparison",
            "",
            f"The selected final model is **{selected_model}**, based on validation log loss. "
            "PR-AUC should be compared with the partition's default-rate baseline. Lower Brier "
            "score and log loss indicate better probabilistic predictions; ROC-AUC and PR-AUC "
            "measure ranking rather than calibration.",
            "",
            "| Model | Validation ROC-AUC | Validation PR-AUC | Validation log loss | Test ROC-AUC | Test PR-AUC | Test Brier | Test log loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected_rows.itertuples(index=False):
        report_lines.append(
            f"| {row.candidate_id} | {_format_metric(row.validation_roc_auc)} | "
            f"{_format_metric(row.validation_pr_auc)} | {_format_metric(row.validation_log_loss)} | "
            f"{_format_metric(row.test_roc_auc)} | {_format_metric(row.test_pr_auc)} | "
            f"{_format_metric(row.test_brier_score)} | {_format_metric(row.test_log_loss)} |"
        )
    report_lines.extend(
        [
            "",
            "## Logistic interpretation",
            "",
            "The selected logistic candidate is interpreted on the original engineered scales "
            "by reversing training-set standardization. Odds ratios compare the stated unit or "
            "categorical level with its reference while holding other predictors fixed. The "
            "strongest terms below are ranked by absolute standardized coefficient, which is a "
            "predictive measure and not a causal effect or an inferential significance test.",
            "",
            "| Parameter | Odds ratio | Comparison |",
            "| --- | ---: | --- |",
        ]
    )
    non_intercept = interpretation.loc[interpretation["parameter"].ne("Intercept")].copy()
    strongest = non_intercept.reindex(
        non_intercept["training_standardized_coefficient"].abs().sort_values(
            ascending=False
        ).index
    ).head(8)
    for row in strongest.itertuples(index=False):
        report_lines.append(
            f"| {row.parameter} | {row.odds_ratio:.3f} | {row.comparison} |"
        )
    report_lines.extend(
        [
            "",
            "Full coefficients and reference categories are in "
            "`outputs/tables/logistic_interpretation.csv`. Continuous predictors marked in the "
            "feature contract are capped at training-set 0.5th and 99.5th percentiles.",
            "",
            "## Limitations and handoff",
            "",
            "The analysis includes only resolved outcomes, so it does not represent current or "
            "unresolved applications. The 2018 cohort has shorter outcome follow-up and its "
            "observed default rate is not directly comparable with mature vintages. Temporal "
            "drift and missingness may reduce transportability. Predictive associations do not "
            "establish causation.",
            "",
            "`out_of_time_predictions.csv` contains the real 2018 predictions for each selected "
            "model family. Confidence intervals, threshold selection, subgroup analysis, and "
            "further robustness checks remain Member 5's validation work.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--feature-spec", type=Path, default=DEFAULT_FEATURE_SPEC)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-cutoff", default="2017-01-01")
    parser.add_argument("--test-cutoff", default="2018-01-01")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Modelling input not found: {args.input}. Run the cleaning command first."
        )
    validate_feature_specification(args.feature_spec)
    validation_cutoff = pd.Timestamp(args.validation_cutoff).strftime("%Y-%m-%d")
    test_cutoff = pd.Timestamp(args.test_cutoff).strftime("%Y-%m-%d")

    header = pd.read_csv(args.input, nrows=0, compression="infer")
    missing = sorted(set(INPUT_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"Modelling input is missing columns: {', '.join(missing)}")
    frame = pd.read_csv(
        args.input,
        usecols=list(INPUT_COLUMNS),
        compression="infer",
        low_memory=False,
    )
    validate_input(frame)
    train, validation, test = temporal_partitions(
        frame, args.validation_cutoff, args.test_cutoff
    )
    print(
        f"Temporal partitions: train={len(train):,}, validation={len(validation):,}, "
        f"test={len(test):,}"
    )

    preprocessor = TrainOnlyPreprocessor().fit(engineer_features(train))
    x_train = preprocessor.transform(engineer_features(train))
    x_validation = preprocessor.transform(engineer_features(validation))
    y_train = train["default"].to_numpy(dtype="int8")
    y_validation = validation["default"].to_numpy(dtype="int8")
    print(
        f"Training-fitted preprocessing: {len(NUMERIC_FEATURES)} numeric and "
        f"{len(CATEGORICAL_FEATURES)} categorical features -> {x_train.shape[1]} columns"
    )

    specifications = prespecified_candidates()
    comparison_rows: list[dict[str, Any]] = []
    fitted_models = {}
    validation_probabilities = {}
    for specification in specifications:
        started = time.perf_counter()
        model = specification.make_estimator(args.seed)
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        metrics = binary_metrics(y_validation, probability)
        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "candidate_id": specification.candidate_id,
            "model_family": specification.family,
            "training_iterations": _training_iterations(model, specification.family),
            "selected_within_family": False,
            "selected_final_model": False,
        }
        row.update({f"validation_{name}": value for name, value in metrics.items()})
        row.update({f"test_{name}": np.nan for name in METRICS})
        comparison_rows.append(row)
        fitted_models[specification.candidate_id] = model
        validation_probabilities[specification.candidate_id] = probability
        print(
            f"{specification.candidate_id}: validation log_loss={metrics['log_loss']:.6f}, "
            f"ROC-AUC={metrics['roc_auc']:.6f} ({elapsed:.1f}s)"
        )

    comparison = pd.DataFrame(comparison_rows)
    selected_by_family: dict[str, str] = {}
    for family in MODEL_FAMILIES:
        family_indices = comparison.index[comparison["model_family"].eq(family)]
        selected_index = comparison.loc[family_indices, "validation_log_loss"].idxmin()
        comparison.loc[selected_index, "selected_within_family"] = True
        selected_by_family[family] = comparison.loc[selected_index, "candidate_id"]
    selected_test_indices = comparison.index[comparison["selected_within_family"]]
    final_index = comparison.loc[selected_test_indices, "validation_log_loss"].idxmin()
    comparison.loc[final_index, "selected_final_model"] = True
    selected_model = comparison.loc[final_index, "candidate_id"]
    print(f"Selected on validation only: {selected_model}")

    x_test = preprocessor.transform(engineer_features(test))
    y_test = test["default"].to_numpy(dtype="int8")
    prediction_tables = []
    for index in selected_test_indices:
        candidate_id = comparison.loc[index, "candidate_id"]
        model = fitted_models[candidate_id]
        probability = model.predict_proba(x_test)[:, 1]
        metrics = binary_metrics(y_test, probability)
        for name, value in metrics.items():
            comparison.loc[index, f"test_{name}"] = value
        prediction_tables.append(
            pd.DataFrame(
                {
                    "id": test["id"].to_numpy(),
                    "issue_date": pd.to_datetime(test["issue_date"]).dt.strftime("%Y-%m-%d"),
                    "default": y_test,
                    "model": candidate_id,
                    "predicted_probability": probability,
                    "selected_final_model": candidate_id == selected_model,
                }
            )
        )
    predictions = pd.concat(prediction_tables, ignore_index=True)

    logistic_id = selected_by_family["logistic_regression"]
    interpretation = logistic_interpretation(
        fitted_models[logistic_id], preprocessor
    )
    split = _partition_summary(
        {"train": train, "validation": validation, "test": test},
        validation_cutoff,
        test_cutoff,
    )
    specification_table = pd.DataFrame(
        [
            {
                "candidate_id": specification.candidate_id,
                "model_family": specification.family,
                "tuned_parameters": json.dumps(
                    specification.parameters, sort_keys=True, default=_json_default
                ),
                "fixed_settings": json.dumps(
                    _fixed_settings(specification.family),
                    sort_keys=True,
                    default=_json_default,
                ),
                "training_rows": len(train),
                "validation_rows": len(validation),
                "selection_metric": "validation_log_loss",
                "random_seed": args.seed,
            }
            for specification in specifications
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output_dir / "temporal_split.csv", index=False)
    specification_table.to_csv(args.output_dir / "model_specifications.csv", index=False)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    interpretation.to_csv(args.output_dir / "logistic_interpretation.csv", index=False)
    predictions.to_csv(args.output_dir / "out_of_time_predictions.csv", index=False)
    metadata = {
        "input": str(args.input.relative_to(PROJECT_DIR))
        if args.input.is_relative_to(PROJECT_DIR)
        else str(args.input),
        "feature_specification": str(args.feature_spec.relative_to(PROJECT_DIR))
        if args.feature_spec.is_relative_to(PROJECT_DIR)
        else str(args.feature_spec),
        "target": "default",
        "feature_contract": "Model C without issue_year; interest rate included and grade excluded",
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "excluded_fields": [
            "id",
            "issue_date",
            "issue_year",
            "loan_status",
            "grade",
            "sub_grade",
            "all post-outcome and audit fields",
        ],
        "validation_cutoff": validation_cutoff,
        "test_cutoff": test_cutoff,
        "random_seed": args.seed,
        "candidate_count": len(specifications),
        "candidate_families": list(MODEL_FAMILIES),
        "selection_rule": "Minimum validation log loss within each family, then across families",
        "class_imbalance": "No resampling; unweighted and class-weighted candidates compared",
        "selected_by_family": selected_by_family,
        "selected_final_model": selected_model,
        "test_evaluated_candidates": list(selected_by_family.values()),
        "preprocessing_fitted_on": "training partition only",
        "preprocessing": preprocessor.audit(),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    _write_json(args.output_dir / "modelling_metadata.json", metadata)
    write_report(
        args.report_path,
        split,
        comparison,
        interpretation,
        selected_model,
        validation_cutoff,
        test_cutoff,
    )
    print(f"Wrote modelling tables to {args.output_dir}")
    print(f"Wrote modelling report to {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
