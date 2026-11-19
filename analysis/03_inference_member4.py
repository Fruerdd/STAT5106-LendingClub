"""Member 4: prespecified statistical inference and feature engineering.

The script implements the frozen contract in ``report/member4_inference_specification.md``.
It fits three primary binomial GLMs plus prespecified maturity, extreme-value, and
employment-missingness sensitivity analyses. Predictive accuracy is deliberately out of scope.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference_features import (
    RARE_MIN_COUNT,
    Scenario,
    build_model_formula,
    load_inference_input,
    prepare_inference_data,
)
from src.inference_models import fit_inference_model
from src.inference_reporting import (
    create_inference_forest_plot,
    write_inference_report,
)
from src.paths import CLEANED_DATA_DIR, FIGURES_DIR, PROJECT_DIR, TABLES_DIR

DEFAULT_INPUT = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
MIN_BINARY_LEVEL_COUNT = 20
SCENARIO_ORDER: tuple[Scenario, ...] = (
    "primary",
    "mature",
    "unwinsorized",
    "employment_complete_case",
)
MODELS_BY_SCENARIO = {
    "primary": ("A", "B", "C"),
    "mature": ("A",),
    "unwinsorized": ("A",),
    "employment_complete_case": ("A",),
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_outputs(
    output_dir: Path,
    coefficients: list[pd.DataFrame],
    global_tests: list[pd.DataFrame],
    model_audits: list[dict[str, object]],
    preparation_audits: list[dict[str, object]],
    *,
    input_path: Path,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_table = pd.concat(coefficients, ignore_index=True)
    global_test_table = pd.concat(global_tests, ignore_index=True)
    sample_summary_columns = (
        "scenario",
        "model",
        "nobs",
        "default_count",
        "default_rate",
        "parameter_count",
        "converged",
        "iterations",
        "covariance_type",
        "leverage_max",
        "leverage_denominator_clipped_count",
        "standardized_condition_number",
        "maximum_vif",
        "maximum_vif_parameter",
    )
    sample_summary = pd.DataFrame(model_audits).loc[:, sample_summary_columns]

    coefficient_table.to_csv(output_dir / "inference_odds_ratios.csv", index=False)
    global_test_table.to_csv(output_dir / "inference_global_tests.csv", index=False)
    sample_summary.to_csv(output_dir / "inference_sample_summary.csv", index=False)
    audit_payload: dict[str, object] = {
        "input": str(input_path),
        "max_rows": max_rows,
        "preparation_audits": preparation_audits,
        "model_audits": model_audits,
    }
    with (output_dir / "inference_preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump(
            audit_payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")
    return coefficient_table, global_test_table, sample_summary, audit_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_ORDER,
        default=list(SCENARIO_ORDER),
        help="Prespecified scenarios to run; primary fits A/B/C and sensitivities fit A.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Development-only row limit; requires a non-default output directory.",
    )
    parser.add_argument("--rare-min-count", type=int, default=RARE_MIN_COUNT)
    parser.add_argument("--hc3-chunk-size", type=int, default=50_000)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if (
        args.max_rows is not None
        and args.output_dir.resolve() == TABLES_DIR.resolve()
    ):
        raise ValueError(
            "--max-rows is development-only and requires a non-default --output-dir"
        )

    requested_scenarios = [
        scenario for scenario in SCENARIO_ORDER if scenario in args.scenarios
    ]
    full_scenario_run = set(requested_scenarios) == set(SCENARIO_ORDER)
    if (
        not full_scenario_run
        and args.output_dir.resolve() == TABLES_DIR.resolve()
    ):
        raise ValueError(
            "A partial scenario run requires a non-default --output-dir so it "
            "cannot overwrite the complete reviewed inference artefacts"
        )
    coefficient_tables: list[pd.DataFrame] = []
    global_test_tables: list[pd.DataFrame] = []
    model_audits: list[dict[str, object]] = []
    preparation_audits: list[dict[str, object]] = []

    for scenario in requested_scenarios:
        print(f"Preparing scenario: {scenario}")
        raw = load_inference_input(args.input, nrows=args.max_rows)
        prepared = prepare_inference_data(
            raw,
            scenario,
            rare_min_count=args.rare_min_count,
        )
        del raw
        gc.collect()
        preparation_audits.append(prepared.summary)
        print(
            f"  rows={len(prepared.frame):,}; "
            f"default_rate={prepared.frame['default'].mean():.4%}"
        )
        income_missing_counts = (
            prepared.frame["income_or_dti_missing"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )
        include_income_missing = bool(
            income_missing_counts.min() >= MIN_BINARY_LEVEL_COUNT
        )
        prepared.summary["income_or_dti_missing_level_counts"] = {
            str(level): int(count)
            for level, count in income_missing_counts.items()
        }
        prepared.summary["income_or_dti_missing_included"] = include_income_missing
        if not include_income_missing:
            print(
                "  omitting income_or_dti_missing: "
                f"a binary level has fewer than {MIN_BINARY_LEVEL_COUNT} rows"
            )

        for model_name in MODELS_BY_SCENARIO[scenario]:
            formula = build_model_formula(
                model_name,
                include_employment_missing_indicator=(
                    scenario != "employment_complete_case"
                ),
                include_income_or_dti_missing_indicator=include_income_missing,
            )
            print(f"  fitting Model {model_name} with custom GLM HC3 covariance")
            fitted = fit_inference_model(
                prepared.frame,
                formula,
                scenario=scenario,
                model_name=model_name,
                hc3_chunk_size=args.hc3_chunk_size,
                maximum_iterations=args.max_iterations,
            )
            coefficient_tables.append(fitted.coefficients)
            global_test_tables.append(fitted.global_tests)
            model_audits.append(fitted.audit)
            print(
                f"    converged in {fitted.audit['iterations']} iterations; "
                f"parameters={fitted.audit['parameter_count']}"
            )

        del prepared
        gc.collect()

    (
        coefficient_table,
        global_test_table,
        sample_summary,
        audit_payload,
    ) = _write_outputs(
        args.output_dir,
        coefficient_tables,
        global_test_tables,
        model_audits,
        preparation_audits,
        input_path=args.input,
        max_rows=args.max_rows,
    )
    print(f"Inference tables and audits written to {args.output_dir}")

    if full_scenario_run:
        if args.output_dir.resolve() == TABLES_DIR.resolve():
            figure_path = FIGURES_DIR / "inference_forest_plot.png"
            report_path = PROJECT_DIR / "report" / "member4_inference_report.md"
            figure_reference = "../outputs/figures/inference_forest_plot.png"
        else:
            figure_path = args.output_dir / "inference_forest_plot.png"
            report_path = args.output_dir / "member4_inference_report.md"
            figure_reference = "inference_forest_plot.png"
        create_inference_forest_plot(coefficient_table, figure_path)
        write_inference_report(
            coefficient_table,
            global_test_table,
            sample_summary,
            audit_payload,
            figure_reference=figure_reference,
            output_path=report_path,
        )
        print(f"Inference forest plot written to {figure_path}")
        print(f"Inference report written to {report_path}")
    else:
        print("Skipping forest plot and report because not all scenarios were requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
