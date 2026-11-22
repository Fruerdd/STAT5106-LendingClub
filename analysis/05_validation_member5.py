"""Member 5: out-of-time validation, calibration, thresholds, subgroups, robustness, limitations.

Run from the repository root after analysis/04_modelling_pavel.py:

    python analysis/05_validation_member5.py

Validation-period probabilities (threshold selection only) come from --val-predictions when
supplied. Otherwise Pavel's selected candidate is refitted on the training partition with
Pavel's seed, and the refit is accepted only if it reproduces his 2018 test predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lending_club_data_cleaning import LEAKAGE_FIELDS
from src.paths import CLEANED_DATA_DIR, FIGURES_DIR, PROJECT_DIR, TABLES_DIR
from src.predictive_modelling import (
    CATEGORICAL_FEATURES, INPUT_COLUMNS, NUMERIC_FEATURES, RANDOM_STATE,
    TrainOnlyPreprocessor, engineer_features, prespecified_candidates,
    temporal_partitions, validate_input,
)

DEFAULT_CLEANED = CLEANED_DATA_DIR / "clean_v1" / "lending_club_clean.csv.gz"
DEFAULT_CLEANING_REPORT = DEFAULT_CLEANED.with_name("cleaning_report.json")
DEFAULT_PREDICTIONS = TABLES_DIR / "out_of_time_predictions.csv"
DEFAULT_COMPARISON = TABLES_DIR / "model_comparison.csv"
DEFAULT_METADATA = TABLES_DIR / "modelling_metadata.json"
DEFAULT_REPORT = PROJECT_DIR / "report" / "member5_validation_report.md"

REQUIRED_PREDICTION_COLUMNS = {
    "id", "issue_date", "default", "predicted_probability", "model", "selected_final_model",
}
REQUIRED_COMPARISON_COLUMNS = {
    "candidate_id", "validation_default_rate", "validation_roc_auc",
    "validation_pr_auc", "validation_brier_score",
}
SUBGROUP_COLUMNS = ("grade", "term_months", "home_ownership", "purpose", "verification_status",
                    "application_type", "emp_length_years_was_missing")
# Member 4's primary-inference complete-case rule. The prediction pipeline imputes these instead.
COMPLETE_CASE_FIELDS = ("revol_util", "credit_history_months", "delinq_2yrs",
                        "inq_last_6mths", "open_acc", "pub_rec")
# Name fragments of post-origination fields; any model input containing one is rejected.
POST_ORIGINATION_PATTERNS = ("last_fico", "last_pymnt", "last_credit_pull", "next_pymnt",
                             "recoveries", "collection_recovery", "total_pymnt", "total_rec_",
                             "out_prncp", "settlement", "hardship")
NON_PREDICTORS = {"id", "loan_status", "default", "issue_date", "issue_d"}
FALLBACK_SETTINGS = {"validation_cutoff": "2017-01-01", "test_cutoff": "2018-01-01",
                     "random_seed": RANDOM_STATE}
DATASET_END = pd.Timestamp("2018-12-31")
FIGURE_NAMES = ("validation_roc_pr_curves.png", "validation_calibration_plot.png",
                "validation_threshold_tradeoffs.png")
MANIFEST_NAME = "validation_output_manifest.json"
SUMMARY_KEYS = ("n_obs", "prevalence_baseline",
                "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                "pr_auc_ap", "pr_auc_ap_ci_low", "pr_auc_ap_ci_high",
                "brier_score", "brier_score_ci_low", "brier_score_ci_high",
                "calibration_intercept", "calibration_slope")


# --------------------------------------------------------------------------- helpers
def _norm_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and np.all(np.mod(numeric, 1) == 0):
        return numeric.astype("int64").astype(str)
    return series.astype(str).str.strip()


def _binary(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{label} must be binary 0/1 with no missing or non-numeric values.")
    return values.astype("int8")


def _strict_bool(series: pd.Series, label: str) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "false": False, "0": False})
    if mapped.isna().any():
        raise ValueError(f"{label} must be boolean (True/False or 1/0).")
    return mapped.astype(bool)


def _ci(draws: np.ndarray) -> tuple[float, float]:
    valid = draws[np.isfinite(draws)]
    if valid.size == 0:
        return np.nan, np.nan
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def _months_to_end(date: pd.Timestamp) -> int:
    return (DATASET_END.year - date.year) * 12 + DATASET_END.month - date.month


def _diag(row: Any) -> dict[str, Any]:
    raw = row.get("diagnostics")
    return json.loads(raw) if isinstance(raw, str) and raw else {}


def _save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {name}")


# --------------------------------------------------------------------------- inputs
def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Take cutoffs/seed/selected model from Pavel's metadata; reject conflicting CLI values."""
    meta: dict[str, Any] = {}
    if args.metadata.is_file():
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    else:
        warnings.warn(f"{args.metadata} not found; settings cannot be cross-checked with Pavel.")
    settings: dict[str, Any] = {}
    for key in ("validation_cutoff", "test_cutoff", "random_seed"):
        cli, recorded = getattr(args, key), meta.get(key)
        if key.endswith("cutoff"):
            cli = None if cli is None else pd.Timestamp(cli).strftime("%Y-%m-%d")
            recorded = None if recorded is None else pd.Timestamp(recorded).strftime("%Y-%m-%d")
        if cli is not None and recorded is not None and str(cli) != str(recorded):
            raise ValueError(f"{key}={cli} conflicts with Pavel's recorded {key}={recorded}.")
        settings[key] = cli if cli is not None else (recorded or FALLBACK_SETTINGS[key])
    settings["random_seed"] = int(settings["random_seed"])
    settings["metadata_selected_model"] = meta.get("selected_final_model")
    return settings


def check_predictor_contract() -> None:
    predictors = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    leaked_inputs = (set(LEAKAGE_FIELDS) | {"loan_status"}) & set(INPUT_COLUMNS)
    leaked_predictors = (set(LEAKAGE_FIELDS) | NON_PREDICTORS) & predictors
    suspicious = sorted(c for c in INPUT_COLUMNS
                        if any(pat in c.lower() for pat in POST_ORIGINATION_PATTERNS))
    if leaked_inputs or leaked_predictors or suspicious:
        raise AssertionError(
            f"Leakage contract violated: inputs={sorted(leaked_inputs)}, "
            f"predictors={sorted(leaked_predictors)}, post-origination-like names={suspicious}")


def load_cohort(path: Path, subgroup_cols: Sequence[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, compression="infer").columns
    missing = sorted(set(INPUT_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"Cleaned cohort is missing model input columns: {missing}")
    extras = [c for c in dict.fromkeys((*subgroup_cols, "emp_length_years"))
              if c in header and c not in INPUT_COLUMNS]
    cohort = pd.read_csv(path, usecols=[*INPUT_COLUMNS, *extras], compression="infer",
                         low_memory=False)
    validate_input(cohort[list(INPUT_COLUMNS)])
    cohort["default"] = _binary(cohort["default"], "Cleaned cohort 'default'")
    cohort["issue_dt"] = pd.to_datetime(cohort["issue_date"], errors="coerce")
    if cohort["issue_dt"].isna().any():
        raise ValueError("Cleaned cohort contains unparseable issue_date values.")
    cohort["id_key"] = _norm_id(cohort["id"])
    if not cohort["id_key"].is_unique:
        raise ValueError("Cleaned cohort contains duplicate loan IDs.")
    if ("emp_length_years_was_missing" in subgroup_cols
            and "emp_length_years_was_missing" not in cohort.columns
            and "emp_length_years" in cohort.columns):
        cohort["emp_length_years_was_missing"] = cohort["emp_length_years"].isna().astype("int8")
    return cohort


def verify_predictions(cohort: pd.DataFrame, preds: pd.DataFrame, test_cutoff: str,
                       subgroup_cols: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Row identity, uniqueness, outcome/date consistency, probability validity, no training rows."""
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(preds.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    preds = preds.copy()
    preds["selected_final_model"] = _strict_bool(preds["selected_final_model"], "selected_final_model")
    if preds.groupby("model")["selected_final_model"].nunique().gt(1).any():
        raise ValueError("selected_final_model must be constant within each model.")
    n_selected = preds.loc[preds["selected_final_model"], "model"].nunique()
    if n_selected != 1:
        raise ValueError(f"Exactly one model must have selected_final_model=True; found {n_selected}.")
    preds["default"] = _binary(preds["default"], "Predictions 'default'")
    prob = pd.to_numeric(preds["predicted_probability"], errors="coerce")
    if prob.isna().any() or not np.isfinite(prob).all() or not prob.between(0, 1).all():
        raise ValueError("predicted_probability must be finite and within [0, 1].")
    preds["predicted_probability"] = prob.astype("float64")
    preds["id_key"] = _norm_id(preds["id"])
    preds["issue_dt"] = pd.to_datetime(preds["issue_date"], errors="coerce")
    if preds["issue_dt"].isna().any():
        raise ValueError("Predictions contain unparseable issue_date values.")

    cutoff = pd.Timestamp(test_cutoff)
    earlier_ids = set(cohort.loc[cohort["issue_dt"] < cutoff, "id_key"])
    overlap = earlier_ids & set(preds["id_key"])
    if overlap or (preds["issue_dt"] < cutoff).any():
        raise ValueError(f"Leakage: {len(overlap)} training/validation loans found in predictions.")

    test = cohort.loc[cohort["issue_dt"] >= cutoff]
    reference = test.set_index("id_key")
    for name, grp in preds.groupby("model", sort=False):
        if not grp["id_key"].is_unique:
            raise ValueError(f"Duplicate loan IDs in predictions for {name}.")
        if len(grp) != len(reference) or not grp["id_key"].isin(reference.index).all():
            raise ValueError(f"ID identity mismatch with held-out cohort for {name} "
                             f"({len(grp)} rows vs {len(reference)} expected).")
        aligned = reference.loc[grp["id_key"].to_numpy()]
        if (aligned["default"].to_numpy() != grp["default"].to_numpy()).any():
            raise ValueError(f"Outcome mismatch between predictions and cohort for {name}.")
        if (aligned["issue_dt"].to_numpy() != grp["issue_dt"].to_numpy()).any():
            raise ValueError(f"issue_date mismatch between predictions and cohort for {name}.")

    attach = [c for c in dict.fromkeys(subgroup_cols) if c in test.columns and c not in preds.columns]
    enriched = preds.merge(test[["id_key", *attach]], on="id_key", how="left",
                           validate="many_to_one")
    summary = {
        "n_models": int(preds["model"].nunique()),
        "rows_per_model": int(len(reference)),
        "prob_min": float(preds["predicted_probability"].min()),
        "prob_max": float(preds["predicted_probability"].max()),
        "date_min": preds["issue_dt"].min(),
        "date_max": preds["issue_dt"].max(),
    }
    print(f"Verified {len(test):,} held-out loans for {summary['n_models']} model(s).")
    return enriched, summary


# --------------------------------------------------------------------------- metrics
def calibration_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration-in-the-large (offset logistic model) and slope (logistic recalibration)."""
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    logit = np.log(clipped / (1 - clipped))
    family = sm.families.Binomial()
    slope_fit = sm.GLM(y, sm.add_constant(logit, has_constant="add"), family=family).fit()
    citl_fit = sm.GLM(y, np.ones((len(y), 1)), family=family, offset=logit).fit()
    return {
        "mean_predicted_prob": float(np.mean(p)),
        "observed_default_rate": float(np.mean(y)),
        "calibration_intercept": float(citl_fit.params[0]),
        "calibration_slope": float(slope_fit.params[1]),
    }


def bootstrap_metrics(y: np.ndarray, p: np.ndarray, n_resamples: int, seed: int,
                      calibration: bool = False) -> dict[str, Any]:
    """Nonparametric percentile bootstrap, resampling individual loans with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ["roc_auc", "pr_auc_ap", "brier_score"]
    if calibration:
        keys += ["calibration_intercept", "calibration_slope"]
    draws = {k: np.full(n_resamples, np.nan) for k in keys}
    dropped = 0
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        draws["brier_score"][b] = np.mean((pb - yb) ** 2)
        if yb.min() == yb.max():
            dropped += 1
            continue
        draws["roc_auc"][b] = roc_auc_score(yb, pb)
        draws["pr_auc_ap"][b] = average_precision_score(yb, pb)
        if calibration:
            cal = calibration_stats(yb, pb)
            draws["calibration_intercept"][b] = cal["calibration_intercept"]
            draws["calibration_slope"][b] = cal["calibration_slope"]
    out: dict[str, Any] = {
        "n_obs": int(n), "default_count": int(y.sum()), "prevalence_baseline": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_ap": float(average_precision_score(y, p)),
        "brier_score": float(np.mean((p - y) ** 2)),
    }
    if calibration:
        out.update(calibration_stats(y, p))
    for k in keys:
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = _ci(draws[k])
    out.update({"bootstrap_resamples": n_resamples, "dropped_single_class_resamples": dropped,
                "bootstrap_seed": seed, "bootstrap_unit": "individual loan",
                "bootstrap_method": "percentile 95% CI"})
    return out


def paired_model_deltas(preds: pd.DataFrame, model_id: str, n_resamples: int,
                        seed: int) -> pd.DataFrame:
    wide = preds.pivot(index="id_key", columns="model", values="predicted_probability").sort_index()
    y = (preds.loc[preds["model"] == model_id].set_index("id_key")["default"]
         .reindex(wide.index).to_numpy("int8"))
    p_ref = wide[model_id].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for other in [m for m in wide.columns if m != model_id]:
        p_oth = wide[other].to_numpy()
        d_roc, d_pr = np.full(n_resamples, np.nan), np.full(n_resamples, np.nan)
        for b in range(n_resamples):
            idx = rng.integers(0, len(y), size=len(y))
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            d_roc[b] = roc_auc_score(yb, p_ref[idx]) - roc_auc_score(yb, p_oth[idx])
            d_pr[b] = (average_precision_score(yb, p_ref[idx])
                       - average_precision_score(yb, p_oth[idx]))
        row = {"reference_model": model_id, "comparator_model": other,
               "delta_roc_auc": roc_auc_score(y, p_ref) - roc_auc_score(y, p_oth),
               "delta_pr_auc_ap": (average_precision_score(y, p_ref)
                                   - average_precision_score(y, p_oth))}
        row["delta_roc_auc_ci_low"], row["delta_roc_auc_ci_high"] = _ci(d_roc)
        row["delta_pr_auc_ap_ci_low"], row["delta_pr_auc_ap_ci_high"] = _ci(d_pr)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- refits
def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def _fit_predict(model_id: str, train: pd.DataFrame, *eval_frames: pd.DataFrame,
                 seed: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Refit with training-only preprocessing; record unseen levels and all fit/transform warnings."""
    spec = {s.candidate_id: s for s in prespecified_candidates()}[model_id]
    tr_feat = engineer_features(train)
    ev_feats = [engineer_features(f) for f in eval_frames]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prep = TrainOnlyPreprocessor().fit(tr_feat)
        est = spec.make_estimator(seed)
        est.fit(prep.transform(tr_feat), train["default"].to_numpy(dtype="int8"))
        probs = [est.predict_proba(prep.transform(f))[:, 1] for f in ev_feats]
    unseen: dict[str, list[str]] = {}
    unseen_rows: dict[str, int] = {}
    for c in CATEGORICAL_FEATURES:
        if c in tr_feat.columns and all(c in f.columns for f in ev_feats):
            seen = set(tr_feat[c].dropna().astype(str))
            new = set().union(*(set(f[c].dropna().astype(str)) for f in ev_feats)) - seen
            if new:
                unseen[c] = sorted(new)
                unseen_rows[c] = int(sum(f[c].dropna().astype(str).isin(new).sum() for f in ev_feats))
    diag = {"unseen_levels": unseen, "unseen_level_rows": unseen_rows,
            "fit_warnings": sorted({str(w.message) for w in caught})}
    return probs, diag


def obtain_validation_probabilities(cohort: pd.DataFrame, final_test: pd.DataFrame,
                                    val_path: Path | None, model_id: str,
                                    settings: dict[str, Any], tolerance: float
                                    ) -> tuple[np.ndarray, np.ndarray, str]:
    train, val, test = temporal_partitions(cohort[list(INPUT_COLUMNS)],
                                           settings["validation_cutoff"], settings["test_cutoff"])
    y_val = val["default"].to_numpy(dtype="int8")
    val_keys = _norm_id(val["id"]).to_numpy()

    if val_path is not None:
        if not val_path.is_file():
            raise FileNotFoundError(f"Validation predictions not found: {val_path}")
        vdf = pd.read_csv(val_path)
        if not {"id", "predicted_probability"} <= set(vdf.columns):
            raise ValueError("Validation predictions need id and predicted_probability columns.")
        if "model" in vdf.columns:
            vdf = vdf.loc[vdf["model"] == model_id]
        vdf = vdf.assign(id_key=_norm_id(vdf["id"]))
        if not vdf["id_key"].is_unique or set(vdf["id_key"]) != set(val_keys):
            raise ValueError("Validation prediction IDs do not match the validation cohort.")
        p_val = vdf.set_index("id_key").loc[val_keys, "predicted_probability"].to_numpy(float)
        if not (np.isfinite(p_val).all() and ((p_val >= 0) & (p_val <= 1)).all()):
            raise ValueError("Validation probabilities must be finite and within [0, 1].")
        return y_val, p_val, f"Pavel's exported validation predictions ({val_path.name})"

    warnings.warn(f"No validation predictions supplied; refitting '{model_id}' on train "
                  f"(<{settings['validation_cutoff']}) with seed {settings['random_seed']}.")
    (p_val, p_test), diag = _fit_predict(model_id, train, val, test, seed=settings["random_seed"])
    pavel = (final_test.set_index("id_key")["predicted_probability"]
             .reindex(_norm_id(test["id"]).to_numpy()).to_numpy())
    max_diff = float(np.nanmax(np.abs(p_test - pavel)))
    if np.isnan(pavel).any() or max_diff > tolerance:
        raise ValueError(
            f"Refit does not reproduce Pavel's test predictions (max |diff|={max_diff:.2e} > "
            f"{tolerance:.0e}). Ask Pavel to export validation predictions and pass --val-predictions.")
    if diag["unseen_levels"]:
        warnings.warn(f"Primary refit: validation/test levels unseen in training: {diag['unseen_levels']}. "
                      "Check TrainOnlyPreprocessor's OTHER_RARE mapping with Pavel.")
    warn_txt = "; ".join(diag["fit_warnings"]) if diag["fit_warnings"] else "none"
    return y_val, p_val, (f"Refit of '{model_id}' on the training partition; reproduces Pavel's "
                          f"test predictions (max |diff| = {max_diff:.1e}); "
                          f"validation/test levels unseen in training: {diag['unseen_levels'] or 'none'}; "
                          f"fit/transform warnings: {warn_txt}")


# --------------------------------------------------------------------------- thresholds
def _threshold_stats(y: np.ndarray, p: np.ndarray, t: float, cost_ratio: float) -> dict[str, float]:
    """'Positive' = predicted default (declined). Approval = predicted non-default."""
    flag, pos = p >= t, y == 1
    tp, fp = int(np.sum(flag & pos)), int(np.sum(flag & ~pos))
    fn = int(np.sum(~flag & pos))
    tn = len(y) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan)
    approved = tn + fn
    return {"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "specificity": tn / (tn + fp) if tn + fp else np.nan, "f1": f1,
            "approval_rate": approved / len(y),
            "approved_default_rate": fn / approved if approved else np.nan,
            "expected_cost": (cost_ratio * fn + fp) / len(y)}


def evaluate_thresholds(y_val, p_val, y_test, p_test, cost_ratio, target_approval, grid,
                        n_resamples, seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = pd.DataFrame([_threshold_stats(y_val, p_val, t, cost_ratio) for t in grid])
    rules = {
        f"Cost-optimal ({cost_ratio:g}:1 FN:FP)": curve.loc[curve["expected_cost"].idxmin(), "threshold"],
        "Maximum F1": curve.loc[curve["f1"].idxmax(), "threshold"],
        f"Approval closest to {target_approval:.0%}":
            curve.loc[(curve["approval_rate"] - target_approval).abs().idxmin(), "threshold"],
        "Reference: approve all": np.inf,     # nobody flagged as default
        "Reference: decline all": -np.inf,    # everybody flagged as default
    }
    ci_keys = ("precision", "recall", "specificity", "f1", "approval_rate",
               "approved_default_rate", "expected_cost")
    rng = np.random.default_rng(seed)
    rows = []
    for goal, t in rules.items():
        row = {"operating_goal": goal, **_threshold_stats(y_test, p_test, float(t), cost_ratio)}
        draws = {k: np.full(n_resamples, np.nan) for k in ci_keys}
        for b in range(n_resamples):
            idx = rng.integers(0, len(y_test), size=len(y_test))
            stats = _threshold_stats(y_test[idx], p_test[idx], float(t), cost_ratio)
            for k in ci_keys:
                draws[k][b] = stats[k]
        for k in ci_keys:
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = _ci(draws[k])
        rows.append(row)
    return curve, pd.DataFrame(rows)


# --------------------------------------------------------------------------- subgroups
def evaluate_subgroups(test_df: pd.DataFrame, columns: Sequence[str], min_n: int,
                       min_events: int, n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in test_df.columns:
            warnings.warn(f"Subgroup column '{col}' unavailable; skipped.")
            continue
        raw = test_df[col]
        if pd.api.types.is_float_dtype(raw) and (raw.dropna() % 1 == 0).all():
            raw = raw.astype("Int64")          # 36.0 -> 36
        levels = raw.astype("string").fillna("<Missing>")
        for level, grp in test_df.groupby(levels, sort=True):
            y = grp["default"].to_numpy(dtype="int8")
            p = grp["predicted_probability"].to_numpy(dtype="float64")
            events, non_events = int(y.sum()), int(len(y) - y.sum())
            adequate = len(y) >= min_n and events >= min_events and non_events >= min_events
            row: dict[str, Any] = {
                "subgroup_variable": col, "subgroup_level": str(level), "sample_size": len(y),
                "default_count": events, "non_default_count": non_events,
                "observed_default_rate": float(y.mean()), "mean_predicted_prob": float(p.mean()),
                "adequate_sample": adequate,
            }
            if adequate:   # metrics only for adequate strata
                boot = bootstrap_metrics(y, p, n_resamples, seed)
                row.update({k: v for k, v in boot.items()
                            if k.startswith(("roc_auc", "pr_auc_ap", "brier_score"))})
                cal = calibration_stats(y, p)
                row.update(calibration_intercept=cal["calibration_intercept"],
                           calibration_slope=cal["calibration_slope"])
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- robustness
def _summary_row(dimension: str, spec: str, cohort_label: str, status: str,
                 values: dict[str, Any]) -> dict[str, Any]:
    return {"robustness_dimension": dimension, "specification": spec,
            "eval_cohort": cohort_label, "status": status,
            **{k: values.get(k, np.nan) for k in SUMMARY_KEYS}, "diagnostics": ""}


def run_robustness_checks(cohort: pd.DataFrame, comparison_path: Path, model_id: str,
                          primary: dict[str, Any], settings: dict[str, Any],
                          args: argparse.Namespace) -> pd.DataFrame:
    seed, n_boot = settings["random_seed"], args.secondary_bootstrap_resamples
    vc, tc = settings["validation_cutoff"], settings["test_cutoff"]
    frame = cohort[list(INPUT_COLUMNS)]
    records = [_summary_row("Primary result (reference)", model_id, f"Test >= {tc}",
                            "reference", primary)]

    n_val = int(((cohort["issue_dt"] >= pd.Timestamp(vc)) & (cohort["issue_dt"] < pd.Timestamp(tc))).sum())
    if comparison_path.is_file():
        comp = pd.read_csv(comparison_path)
        missing = REQUIRED_COMPARISON_COLUMNS - set(comp.columns)
        if missing:
            raise ValueError(f"Model comparison table missing columns: {sorted(missing)}")
        for r in comp.itertuples(index=False):
            records.append(_summary_row(
                "Prespecified model family / class weighting", r.candidate_id,
                f"Validation {vc} to {tc} (not comparable with test rows)", "from model_comparison.csv",
                {"n_obs": n_val, "prevalence_baseline": r.validation_default_rate,
                 "roc_auc": r.validation_roc_auc, "pr_auc_ap": r.validation_pr_auc,
                 "brier_score": r.validation_brier_score}))
    else:
        warnings.warn(f"{comparison_path} not found; candidate comparison rows omitted.")

    def alt_split():
        tr, _, te = temporal_partitions(frame, args.alt_validation_cutoff, args.alt_test_cutoff)
        (p,), diag = _fit_predict(model_id, tr, te, seed=seed)
        return te, p, diag

    def no_int_rate():
        tr, _, te = temporal_partitions(frame, vc, tc)
        median = pd.to_numeric(tr["int_rate"], errors="coerce").median()
        tr, te = tr.assign(int_rate=median), te.assign(int_rate=median)
        (p,), diag = _fit_predict(model_id, tr, te, seed=seed)
        diag["int_rate_fixed_at"] = float(median)
        return te, p, diag

    def term_36():
        sub = frame.loc[pd.to_numeric(cohort["term_months"], errors="coerce").eq(36).to_numpy()]
        tr, _, te = temporal_partitions(sub, vc, tc)
        (p,), diag = _fit_predict(model_id, tr, te, seed=seed)
        diag["train_rows"] = int(len(tr))
        return te, p, diag

    def mature_cohort():
        month_index = (cohort["issue_dt"].dt.year * 12 + cohort["issue_dt"].dt.month - 1
                       + pd.to_numeric(cohort["term_months"], errors="coerce"))
        end_index = DATASET_END.year * 12 + DATASET_END.month - 1
        sub = frame.loc[(month_index <= end_index).to_numpy()]
        tr, _, te = temporal_partitions(sub, args.mature_validation_cutoff, args.mature_test_cutoff)
        (p,), diag = _fit_predict(model_id, tr, te, seed=seed)
        cc_cols = [c for c in COMPLETE_CASE_FIELDS if c in sub.columns]
        cc_missing = int(sub[cc_cols].isna().any(axis=1).sum())
        diag.update(
            mature_cohort_rows=int(len(sub)),
            complete_case_fields=cc_cols,
            complete_case_missing_rows=cc_missing,          # retained here, imputed by the pipeline
            complete_case_rows=int(len(sub) - cc_missing),  # should equal Member 4's mature cohort
            train_default_rate=float(tr["default"].mean()),
            test_issue_years=sorted(pd.to_datetime(te["issue_date"]).dt.year.unique().tolist()),
            train_term_counts=_counts(tr["term_months"]),
            test_term_counts=_counts(te["term_months"]))
        return te, p, diag

    checks = [
        ("Alternative temporal split", f"train < {args.alt_validation_cutoff}",
         f"Test >= {args.alt_test_cutoff} (includes primary validation and test years; "
         f"{args.alt_validation_cutoff} to {args.alt_test_cutoff} unused: no tuning in this check)",
         alt_split),
        ("Feature specification: no lender pricing proxy",
         "int_rate fixed at training median (uninformative)", f"Test >= {tc}", no_int_rate),
        ("Term stratification (36-month loans only)",
         f"refit on 36-month training loans (train < {vc}), evaluated on 36-month test loans",
         f"Test >= {tc}, 36-month", term_36),
        ("Outcome maturity (issue_date + term <= 2018-12-31)",
         f"mature loans, train < {args.mature_validation_cutoff}",
         f"Mature test >= {args.mature_test_cutoff}", mature_cohort),
    ]
    for dimension, spec, cohort_label, fn in checks:
        try:
            te, p, diag = fn()
            y = te["default"].to_numpy(dtype="int8")
            values = bootstrap_metrics(y, p, n_boot, seed)
            values.update(calibration_stats(y, p))
            row = _summary_row(dimension, f"{model_id}: {spec}", cohort_label, "completed", values)
            row["diagnostics"] = json.dumps(diag, sort_keys=True)
        except Exception as exc:
            warnings.warn(f"Robustness check '{dimension}' not completed: {exc}")
            row = _summary_row(dimension, f"{model_id}: {spec}", cohort_label,
                               f"NOT COMPLETED: {type(exc).__name__}: {exc}", {})
        records.append(row)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- drift tables
def drift_tables(cohort: pd.DataFrame, vc: str, tc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    vintage = (cohort.groupby(cohort["issue_dt"].dt.year)["default"]
               .agg(loans="size", defaults="sum", default_rate="mean")
               .rename_axis("issue_year").reset_index())
    partition = np.select([cohort["issue_dt"] < pd.Timestamp(vc), cohort["issue_dt"] < pd.Timestamp(tc)],
                          ["train", "validation"], "test")
    raw_predictors = [c for c in INPUT_COLUMNS if c not in NON_PREDICTORS]
    missing = (cohort[raw_predictors].isna().groupby(partition).mean().T
               .reindex(columns=["train", "validation", "test"])
               .rename_axis("column").reset_index())
    return vintage, missing


# --------------------------------------------------------------------------- figures
def generate_figures(preds: pd.DataFrame, model_id: str, val_curve: pd.DataFrame,
                     thresholds: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    final = preds.loc[preds["model"] == model_id]
    y_f = final["default"].to_numpy()
    p_f = final["predicted_probability"].to_numpy()
    prevalence = float(y_f.mean())

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, grp in preds.groupby("model", sort=False):
        y, p = grp["default"].to_numpy(), grp["predicted_probability"].to_numpy()
        tag, lw = (" [selected]", 2.4) if name == model_id else ("", 1.2)
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax_roc.plot(fpr, tpr, lw=lw, label=f"{name}{tag} (AUC={roc_auc_score(y, p):.3f})")
        ax_pr.plot(rec, prec, lw=lw, label=f"{name}{tag} (AP={average_precision_score(y, p):.3f})")
    ax_roc.plot((0, 1), (0, 1), "k--", lw=1, label="Random ranking (0.500)")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
               title=f"Out-of-time ROC (n={len(y_f):,})")
    ax_pr.axhline(prevalence, color="k", ls="--", lw=1, label=f"Prevalence baseline ({prevalence:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title=f"Out-of-time precision-recall (n={len(y_f):,})")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_pr.legend(loc="upper right", fontsize=8)
    _save_figure(fig, FIGURE_NAMES[0], figures_dir)

    bins = (pd.DataFrame({"bin": pd.qcut(p_f, q=10, duplicates="drop"), "y": y_f, "p": p_f})
            .groupby("bin", observed=True)
            .agg(n_loans=("y", "size"), observed_default_rate=("y", "mean"),
                 mean_predicted_prob=("p", "mean")).reset_index())
    bins["bin"] = bins["bin"].astype(str)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                   gridspec_kw={"height_ratios": (3, 1)})
    upper = min(1.0, max(bins["mean_predicted_prob"].max(), bins["observed_default_rate"].max()) + 0.05)
    ax1.plot((0, 1), (0, 1), "k--", lw=1, label="Perfect calibration")
    ax1.plot(bins["mean_predicted_prob"], bins["observed_default_rate"], "s-", lw=2,
             label=f"{model_id} (prediction deciles)")
    ax1.set(ylabel="Observed default rate", xlim=(0, upper), ylim=(0, upper),
            title=f"Calibration — {model_id}, out-of-time test (n={len(y_f):,})")
    ax1.legend(loc="upper left")
    ax2.hist(p_f, bins=40, alpha=0.8)
    ax2.set(xlabel="Predicted default probability", ylabel="Loans")
    _save_figure(fig, FIGURE_NAMES[1], figures_dir)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for col, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                       ("approval_rate", "Approval rate")):
        ax.plot(val_curve["threshold"], val_curve[col], label=f"{label} (validation)")
    ax_cost = ax.twinx()
    ax_cost.plot(val_curve["threshold"], val_curve["expected_cost"], color="crimson", ls="-.",
                 label="Expected cost per loan (validation)")
    ax_cost.set_ylabel("Expected cost per loan", color="crimson")
    finite = thresholds.loc[np.isfinite(thresholds["threshold"])]
    for colour, r in zip(("dimgray", "navy", "darkgreen"), finite.itertuples(index=False)):
        ax.axvline(r.threshold, color=colour, ls=":", lw=1.6,
                   label=f"{r.operating_goal} (t={r.threshold:.3f})")
    ax.set(xlabel="Probability threshold (declined if p >= t)", ylabel="Rate / score",
           title="Threshold selection on the validation cohort only")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_cost.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.5)
    _save_figure(fig, FIGURE_NAMES[2], figures_dir)
    return bins


# --------------------------------------------------------------------------- report
def _n(x, d=4, sign=False):
    return "N/A" if pd.isna(x) else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def _p(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.2%}"


def _p3(x):
    return "N/A" if pd.isna(x) else f"{x:.3%}"


def _thr(x):
    if pd.isna(x):
        return "N/A"
    return "—" if not np.isfinite(x) else f"{x:.3f}"


def _ci_verdict(row, key):
    lo, hi = row.get(f"{key}_ci_low"), row.get(f"{key}_ci_high")
    if pd.isna(lo) or pd.isna(hi):
        return "CI unavailable"
    return "CI includes 0: no clear difference" if lo <= 0 <= hi else "CI excludes 0"


def _rob_row(rob: pd.DataFrame, prefix: str) -> dict[str, Any] | None:
    hit = rob.loc[rob["robustness_dimension"].str.startswith(prefix) & rob["status"].eq("completed")]
    return None if hit.empty else hit.iloc[0].to_dict()

def _ci_txt(row, key, fmt=_n):
    lo, hi = row.get(key + "_ci_low"), row.get(key + "_ci_high")
    if pd.isna(lo) and pd.isna(hi):
        return fmt(row.get(key))
    return f"{fmt(row.get(key))} ({fmt(lo)}, {fmt(hi)})"


def _terms_txt(d: dict[str, int] | None) -> str:
    if not d:
        return "N/A"
    return ", ".join(f"{float(k):g}-month {v:,}" for k, v in sorted(d.items(), key=lambda kv: float(kv[0])))


def _levels_txt(levels: dict[str, list[str]], rows: dict[str, int]) -> str:
    parts = []
    for k, v in levels.items():
        n = rows.get(k)
        parts.append(f"`{k}` = {', '.join(v)} ({f'{n:,} loans' if n is not None else 'count N/A'})")
    return "; ".join(parts)

def write_report(path: Path, ctx: dict[str, Any]) -> None:
    a, s = ctx["args"], ctx["settings"]
    overall, sel = ctx["overall"], ctx["overall"].loc[ctx["overall"]["selected_final_model"]].iloc[0]
    sub, rob, vint, miss = ctx["subgroups"], ctx["robustness"], ctx["vintage"], ctx["missing"]
    th, ver = ctx["thresholds"], ctx["verification"]
    direction = "lower" if sel["calibration_intercept"] < 0 else "higher"
    mat_row = _rob_row(rob, "Outcome maturity")
    mat_diag = _diag(mat_row) if mat_row else {}
    cc_fields_txt = ", ".join(f"`{c}`" for c in ctx["cc_fields"])

    # ---- 1-2. inputs and discrimination
    L = [
        "# Member 5 — Validation, Robustness, and Limitations", "",
        "## 1. Inputs and verification", "",
        f"- Held-out cohort: issue_date >= {s['test_cutoff']}; {int(sel['n_obs']):,} loans, "
        f"{int(sel['default_count']):,} defaults. Every model's predictions matched this cohort on ID, "
        "outcome, and issue date; no training/validation loan was present.",
        f"- Prediction checks ({ver['n_models']} models): loan IDs unique within each model and identical "
        f"to the held-out cohort ({ver['rows_per_model']:,} rows per model); all probabilities finite and "
        f"within [0, 1] (observed range {ver['prob_min']:.4f} to {ver['prob_max']:.4f}); prediction issue "
        f"dates {ver['date_min']:%Y-%m} to {ver['date_max']:%Y-%m}.",
        f"- Cohort scope: prediction cohorts keep loans with missing values in Member 4's complete-case "
        f"fields ({cc_fields_txt}); the prediction pipeline imputes them with training-partition values. "
        f"Member 4's complete-case rule applies to the inference model only. Primary test cohort: "
        f"{ctx['test_cc_missing']:,} such loans retained (by field: "
        + ", ".join(f"`{k}` {v:,}" for k, v in ctx["test_cc_by_field"].items()) + ").",
        f"- Selected final model (Pavel, chosen on validation log loss): `{ctx['model_id']}`. "
        f"Seed `{s['random_seed']}`; validation cutoff {s['validation_cutoff']}.",
        f"- Bootstrap: nonparametric percentile 95% CI, loans resampled with replacement; "
        f"{a.bootstrap_resamples:,} resamples for overall metrics and paired deltas, "
        f"{a.secondary_bootstrap_resamples:,} for thresholds, subgroups, and robustness; seed "
        f"`{s['random_seed']}`. Loan-level resampling assumes independence (ignores issue-month "
        "clustering) and does not include model-refitting uncertainty.",
        f"- Environment: Python {sys.version.split()[0]}.", "",
        "## 2. Discrimination", "",
        "ROC-AUC is the probability that a random defaulter is ranked above a random non-defaulter; "
        "it does not depend on prevalence but can look favourable when defaults are rare. PR-AUC is "
        "reported as average precision (AP) and must be compared with the default prevalence. "
        "Neither metric measures calibration, performance at a specific threshold, or lending costs, "
        "and AP is not comparable across cohorts with different default rates.", "",
        "| Model | Selected | N | Prevalence | ROC-AUC (95% CI) | AP (95% CI) | Brier (95% CI) |",
        "| --- | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in overall.to_dict("records"):
        L.append(f"| {r['model']} | {'Yes' if r['selected_final_model'] else 'No'} | {r['n_obs']:,} | "
                 f"{_p(r['prevalence_baseline'])} | {_ci_txt(r, 'roc_auc')} | {_ci_txt(r, 'pr_auc_ap')} | "
                 f"{_ci_txt(r, 'brier_score')} |")
    prev = float(sel["prevalence_baseline"])
    L += ["", f"For the selected model, AP {sel['pr_auc_ap']:.4f} is {sel['pr_auc_ap'] / prev:.2f} times the "
          f"prevalence baseline {prev:.4f}, which is the expected AP of a random ranking."]
    if not ctx["paired"].empty:
        L += ["", "Paired bootstrap differences (selected minus comparator, same resampled loans):", ""]
        signed = lambda x: _n(x, sign=True)  # noqa: E731
        for r in ctx["paired"].to_dict("records"):
            L.append(f"- vs `{r['comparator_model']}`: ΔROC-AUC {_ci_txt(r, 'delta_roc_auc', signed)} "
                     f"({_ci_verdict(r, 'delta_roc_auc')}); ΔAP {_ci_txt(r, 'delta_pr_auc_ap', signed)} "
                     f"({_ci_verdict(r, 'delta_pr_auc_ap')}).")

    # ---- 3. calibration
    L += ["", "## 3. Calibration", "",
          "Discrimination concerns ranking; calibration concerns whether predicted probabilities "
          "match observed default frequencies. Ideal values: intercept 0 (calibration-in-the-large, "
          "logistic model with the predicted logit as offset) and slope 1. Intercept < 0 means "
          "observed risk is lower than predicted; slope < 1 means predictions are too extreme.", "",
          "| Model | Mean predicted | Observed | Intercept (95% CI) | Slope (95% CI) |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for r in overall.to_dict("records"):
        L.append(f"| {r['model']} | {_p(r['mean_predicted_prob'])} | {_p(r['observed_default_rate'])} | "
                 f"{_ci_txt(r, 'calibration_intercept', lambda x: _n(x, sign=True))} | "
                 f"{_ci_txt(r, 'calibration_slope')} |")
    brier_ref = prev * (1 - prev)
    L += ["", f"For the selected model, observed default risk is on average **{direction}** than predicted "
          f"(intercept {_n(sel['calibration_intercept'], sign=True)}). Decile bins: "
          "`outputs/tables/validation_calibration_bins.csv`.",
          "", f"Brier reference (constant prediction equal to the test prevalence) = p(1 - p) = "
          f"{brier_ref:.4f}; Brier skill score of the selected model = 1 - Brier / reference = "
          f"{1 - sel['brier_score'] / brier_ref:.3f}. The reference uses the observed test prevalence, so "
          "it is an optimistic benchmark. The Brier score combines discrimination and calibration."]
    if mat_row:
        m = float(mat_row["calibration_intercept"])
        years = mat_diag.get("test_issue_years") or []
        yr_txt = "N/A" if not years else (str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}")
        term_keys = sorted(mat_diag.get("test_term_counts", {}), key=float)
        term_txt = "/".join(f"{float(t):g}" for t in term_keys) or "N/A"
        terms = (f" Maturity-check term composition: train {_terms_txt(mat_diag.get('train_term_counts'))}; "
                 f"test {_terms_txt(mat_diag.get('test_term_counts'))}.")
        alt_years = f"{pd.Timestamp(a.alt_test_cutoff).year}–{DATASET_END.year}"
        if np.sign(m) != np.sign(sel["calibration_intercept"]):
            L += ["", f"On the mature-outcome check (train < {a.mature_validation_cutoff}, training default "
                      f"rate {_p(mat_diag.get('train_default_rate'))}; {a.mature_validation_cutoff} to "
                      f"{a.mature_test_cutoff} unused; test = {yr_txt}-issued {term_txt}-month loans, observed "
                      f"default rate {_p(mat_row['prevalence_baseline'])}) the intercept is {m:+.4f}, versus "
                      f"{sel['calibration_intercept']:+.4f} on the primary test period. The sign change cannot be "
                      "attributed to incomplete follow-up alone: the two analyses also differ in training window, "
                      "evaluation vintage, and term mix. The alternative-split check is not a truncation-free comparison "
                      f"either, because its evaluation years ({alt_years}) are also incompletely followed. "
                      "Calibration therefore varies by vintage in both directions, and the primary test "
                      "intercept should not be used to recalibrate the model." + terms]
        else:
            L += ["", f"On the mature-outcome check the intercept is {m:+.4f} (same sign as the test "
                  "period), consistent with miscalibration persisting in fully observed loans." + terms]

    # ---- 4. thresholds
    lo, hi, steps = ctx["grid"]
    step = (hi - lo) / (steps - 1) if steps > 1 else 0.0
    r_ = a.fn_fp_cost_ratio
    L += ["", "## 4. Thresholds (selected on validation only, then frozen)", "",
          f"- Validation probabilities: {ctx['val_note']}.",
          f"- Threshold grid: {steps} values from {lo:.3f} to {hi:.3f} (step {step:.3f}); a loan is declined "
          "if p >= t. Ties in a selection rule resolve to the lowest threshold.",
          f"- Cost units: a false positive (declining a non-defaulter) costs 1 unit and a false negative "
          f"(approving a defaulter) costs {r_:g} units; cost per loan = ({r_:g} x FN + FP) / N. These are "
          "relative units, not currency.",
          f"- Prevalence: validation {_p(ctx['val_prevalence'])}, test {_p(prev)}.",
          f"- Operating goals are **illustrative assumptions set by CLI arguments, pending team "
          f"confirmation**, not estimates from loss data: cost ratio FN:FP = {r_:g}:1, "
          f"maximum F1, and approval closest to {a.target_approval_rate:.0%}. For a perfectly calibrated "
          f"model the cost-optimal threshold would be 1/(1+{r_:g}) = "
          f"{1 / (1 + r_):.3f}. Predicted default = declined. "
          "Reference rows (approve all / decline all) use no model.", "",
          "| Goal | Threshold | TP | FP | TN | FN | Precision | Recall (95% CI) | Specificity | F1 (95% CI) | Approval | Default rate among approved (95% CI) | Cost/loan (95% CI) |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in th.to_dict("records"):
        L.append(f"| {r['operating_goal']} | {_thr(r['threshold'])} | {r['tp']:,} | {r['fp']:,} | "
                 f"{r['tn']:,} | {r['fn']:,} | {_p(r['precision'])} | {_ci_txt(r, 'recall', _p)} | "
                 f"{_p(r['specificity'])} | {_ci_txt(r, 'f1')} | {_p(r['approval_rate'])} | "
                 f"{_ci_txt(r, 'approved_default_rate', _p)} | {_ci_txt(r, 'expected_cost')} |")
    costs = th.set_index("operating_goal")["expected_cost"]
    cost_goal = next((g for g in costs.index if g.startswith("Cost-optimal")), None)
    if cost_goal and {"Reference: approve all", "Reference: decline all"} <= set(costs.index):
        best_ref = min(costs["Reference: approve all"], costs["Reference: decline all"])
        L += ["", f"Under the assumed cost ratio, the cost-optimal threshold costs {costs[cost_goal]:.4f} per "
              f"loan versus {costs['Reference: approve all']:.4f} (approve all) and "
              f"{costs['Reference: decline all']:.4f} (decline all): a reduction of "
              f"{best_ref - costs[cost_goal]:.4f} per loan relative to the better no-model policy."]

    # ---- 5. subgroups
    inadequate = sub.loc[~sub["adequate_sample"]]
    sg_miss = ctx["subgroup_missing"]
    if any(sg_miss.values()):
        miss_sg_txt = ("Missing values in subgroup variables are reported as a `<Missing>` level: "
                       + ", ".join(f"`{k}` {v:,}" for k, v in sg_miss.items()) + ".")
    else:
        miss_sg_txt = ("None of the subgroup variables (" + ", ".join(f"`{k}`" for k in sg_miss)
                       + ") has missing values in the test cohort, so no `<Missing>` level arises.")
    if "emp_length_years_was_missing" in sg_miss:
        miss_sg_txt += (" Predictor missingness is assessed with `emp_length_years_was_missing` "
                        "(1 = employment length missing and imputed).")
    L += ["", "## 5. Subgroups (selected model, held-out test)", "",
          f"Adequate = n >= {a.min_subgroup_n} with >= {a.min_subgroup_events} defaults and non-defaults. "
          "Metrics and calibration are estimated only for adequate strata; inadequate strata are listed "
          "with counts only. Within-grade ROC-AUC is expected to be lower because grade already "
          "stratifies risk. Many strata are compared, so isolated differences should not be over-interpreted. "
          + miss_sg_txt, "",
          "| Variable | Level | N | Defaults | Observed | Predicted | ROC-AUC (95% CI) | Cal. intercept | Adequate |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for r in sub.to_dict("records"):
        auc = _ci_txt(r, "roc_auc") if pd.notna(r.get("roc_auc")) else "N/A"
        L.append(f"| {r['subgroup_variable']} | {r['subgroup_level']} | {r['sample_size']:,} | "
                 f"{r['default_count']:,} | {_p(r['observed_default_rate'])} | {_p(r['mean_predicted_prob'])} | "
                 f"{auc} | {_n(r.get('calibration_intercept'), sign=True)} | "
                 f"{'Yes' if r['adequate_sample'] else 'No'} |")

    # ---- 6. robustness
    L += ["", "## 6. Robustness (all prespecified checks reported, including failures)", "",
          "Rows evaluated on different cohorts have different prevalences; compare AP only within a cohort.", "",
          "| Dimension | Specification | Cohort | Status | N | Default rate | ROC-AUC (95% CI) | AP (95% CI) | Brier | Cal. intercept |",
          "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rob.to_dict("records"):
        n_txt = "N/A" if pd.isna(r["n_obs"]) else f"{int(r['n_obs']):,}"
        L.append(f"| {r['robustness_dimension']} | {r['specification']} | {r['eval_cohort']} | {r['status']} | "
                 f"{n_txt} | {_p(r['prevalence_baseline'])} | {_ci_txt(r, 'roc_auc')} | {_ci_txt(r, 'pr_auc_ap')} | "
                 f"{_n(r['brier_score'])} | {_n(r['calibration_intercept'], sign=True)} |")
    diag_lines = [f"- *{r['robustness_dimension']}*: `{r['diagnostics']}`"
                  for r in rob.to_dict("records")
                  if isinstance(r.get("diagnostics"), str) and r["diagnostics"]]
    if diag_lines:
        L += ["", "Refit diagnostics (unseen categorical levels, fit/transform warnings, cohort composition):", "",
              *diag_lines]

    notes = []
    term_row = _rob_row(rob, "Term stratification")
    sub36 = sub.loc[sub["subgroup_variable"].eq("term_months") & sub["subgroup_level"].eq("36")]
    if term_row and not sub36.empty:
        notes.append(f"The 36-month row is a model **refitted on 36-month training loans only**. It differs "
                     f"from the 36-month subgroup in Section 5, which restricts the primary model's predictions "
                     f"to 36-month loans (ROC-AUC {term_row['roc_auc']:.4f} vs {sub36['roc_auc'].iat[0]:.4f}).")
    nir = _rob_row(rob, "Feature specification")
    prim = rob.loc[rob["status"].eq("reference")].iloc[0]
    if nir and nir["brier_score"] < prim["brier_score"] and \
            abs(nir["calibration_intercept"]) < abs(prim["calibration_intercept"]):
        notes.append(f"Fixing `int_rate` lowers ROC-AUC ({prim['roc_auc']:.4f} to {nir['roc_auc']:.4f}) but "
                     f"slightly improves the Brier score ({prim['brier_score']:.4f} to {nir['brier_score']:.4f}) "
                     f"and calibration-in-the-large ({prim['calibration_intercept']:+.4f} to "
                     f"{nir['calibration_intercept']:+.4f}). One possible explanation is drift in the relationship "
                     "between lender pricing and default over time; this is an observation, not a tested explanation.")
    if mat_diag.get("complete_case_rows") is not None:
        txt = (f"The maturity cohort ({mat_diag['mature_cohort_rows']:,} rows) retains "
               f"{mat_diag['complete_case_missing_rows']:,} loans with a missing value in at least one "
               f"complete-case field ({cc_fields_txt}); the prediction pipeline imputes them. Applying Member 4's "
               f"complete-case rule would leave {mat_diag['complete_case_rows']:,} rows.")
        m4 = a.member4_mature_complete_case_rows
        if m4 is not None:
            txt += (f" This matches Member 4's reported count ({m4:,})." if m4 == mat_diag["complete_case_rows"]
                    else f" **This does not match Member 4's reported count ({m4:,}); reconcile before submission.**")
        notes.append(txt)
    if notes:
        L += ["", *notes]

    # ---- 7. limitations
    excluded = ctx["cleaning_report"].get("status_exclusions", {}).get("total")
    top_missing = miss.sort_values("test", ascending=False).head(3)
    miss_txt = "; ".join(f"`{r.column}` train {_p3(r.train)}, test {_p3(r.test)}"
                         for r in top_missing.itertuples(index=False))
    n_test = int(sel["n_obs"])
    if a.test_period_issued_loans:
        share_txt = (f" Of {a.test_period_issued_loans:,} loans issued on or after {s['test_cutoff']}, "
                     f"{n_test:,} ({n_test / a.test_period_issued_loans:.1%}) had resolved outcomes and are evaluated.")
    else:
        share_txt = (" The share of test-period issued loans that is retained is not reported: the number of "
                     "loans issued on or after the test cutoff before status exclusion was not supplied "
                     "(--test-period-issued-loans).")

    model_th = th.loc[np.isfinite(th["threshold"])]
    cost_rows = model_th.loc[model_th["operating_goal"].str.startswith("Cost-optimal")]
    if not cost_rows.empty and not model_th.empty:
        c = cost_rows.iloc[0]
        b = model_th.loc[model_th["expected_cost"].idxmin()]
        if b["operating_goal"] != c["operating_goal"]:
            overlap = not (c["expected_cost_ci_low"] > b["expected_cost_ci_high"]
                           or b["expected_cost_ci_low"] > c["expected_cost_ci_high"])
            thr_txt = (f"The threshold that minimised cost on validation (t = {c['threshold']:.3f}, prevalence "
                       f"{_p(ctx['val_prevalence'])}) costs {_ci_txt(c, 'expected_cost')} per loan on test, more "
                       f"than the '{b['operating_goal']}' threshold (t = {b['threshold']:.3f}, "
                       f"{_ci_txt(b, 'expected_cost')}). The intervals {'overlap' if overlap else 'do not overlap'} "
                       "(they come from separate bootstraps, so this is an approximate comparison). Test "
                       f"prevalence is {_p(prev)}. Cost-optimal thresholds depend on prevalence and calibration, "
                       "so they do not transfer unchanged across vintages. Thresholds were not re-tuned on the "
                       "test set; a deployment threshold should be re-selected on recent, fully observed data.")
        else:
            thr_txt = (f"On test, the validation cost-optimal threshold (t = {c['threshold']:.3f}) remains the "
                       "lowest-cost model rule, although validation and test prevalences differ "
                       f"({_p(ctx['val_prevalence'])} vs {_p(prev)}).")
    else:
        thr_txt = "Threshold transfer could not be assessed."

    unseen_parts = []
    for r in rob.to_dict("records"):
        d = _diag(r)
        if not d.get("unseen_levels"):
            continue
        name = r["robustness_dimension"].split(" (")[0].lower()
        warns = "; ".join(f"`{w}`" for w in d.get("fit_warnings", [])) or "none"
        unseen_parts.append(
            f" In the {name} check, evaluation-period levels not seen in training: "
            f"{_levels_txt(d['unseen_levels'], d.get('unseen_level_rows', {}))}. Their encoding depends on "
            f"`TrainOnlyPreprocessor` (rare-level grouping or all-zero one-hot); preprocessing warnings: {warns}.")
    unseen_txt = "".join(unseen_parts)

    L += ["", "## 7. Limitations", "",
          "Observed default rate by issue year (resolved loans only):", "",
          "| Issue year | Loans | Defaults | Default rate |", "| ---: | ---: | ---: | ---: |"]
    L += [f"| {int(r.issue_year)} | {r.loans:,} | {r.defaults:,} | {_p(r.default_rate)} |"
          for r in vint.itertuples(index=False)]
    L += ["",
          f"1. **Temporal drift.** Default rates and predictor missingness change across vintages "
          f"(table above; `validation_missingness_by_partition.csv`). The selected model's test "
          f"calibration intercept is {_n(sel['calibration_intercept'], sign=True)}; see Section 3 and the "
          "alternative-split and maturity rows in Section 6.",
          f"2. **Unresolved-outcome exclusion and unequal follow-up.** "
          f"{'The cleaning step excluded ' + format(excluded, ',') + ' loans with unresolved or unknown status. ' if excluded is not None else ''}"
          f"Loans issued on or after {s['test_cutoff']} had between {ctx['min_followup_months']} and "
          f"{ctx['max_followup_months']} months of follow-up before the dataset end "
          f"({DATASET_END:%Y-%m-%d}), so the resolved test loans are a selected subset (resolved early). "
          "Test metrics describe that subset, not the full vintage." + share_txt + " The maturity-cohort "
          "check in Section 6 reduces, but does not remove, this problem.",
          "3. **Selection bias.** Only accepted and funded loans are observed. Results do not apply to rejected applicants.",
          f"4. **Missing data.** Highest test-period missingness among raw predictors: {miss_txt}. "
          "Imputation values were learned on the training partition only. Prediction cohorts keep loans "
          f"missing any of Member 4's complete-case fields and impute them (primary test: "
          f"{ctx['test_cc_missing']:,}; maturity cohort: {mat_diag.get('complete_case_missing_rows', 'N/A')}); "
          "Member 4's complete-case inference cohort excludes them, so its sample sizes differ from these.",
          f"5. **Subgroup limits.** {len(inadequate)} of {len(sub)} strata are inadequate and are not interpreted"
          + (": " + ", ".join(f"{r.subgroup_variable}={r.subgroup_level} (n={r.sample_size:,})"
                              for r in inadequate.itertuples(index=False)) if len(inadequate) else "") + ".",
          "6. **Non-causal interpretation and generalisability.** Predictors such as interest rate reflect "
          "Lending Club's risk-based pricing; associations are not causal. Results apply to U.S. Lending Club "
          "accepted loans with resolved outcomes (2007–2018) and may not transfer to other lenders or economic conditions.",
          f"7. **Threshold transfer.** {thr_txt}",
          "8. **Method limits.** Thresholds depend on illustrative cost and approval assumptions. The "
          "no-pricing-proxy check neutralises `int_rate` at its training median instead of re-specifying the "
          "model. Loan-level bootstrap CIs may be too narrow if outcomes cluster in time." + unseen_txt, ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote validation report to {path}")


# --------------------------------------------------------------------------- manifest
def write_manifest(tables_dir: Path, figures_dir: Path, report_path: Path,
                   table_names: Sequence[str]) -> None:
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_DIR.resolve()))
        except ValueError:
            return str(p)

    paths = [*(tables_dir / n for n in table_names), *(figures_dir / f for f in FIGURE_NAMES), report_path]
    current = set(table_names) | {MANIFEST_NAME}
    stale = sorted(p.name for p in tables_dir.glob("validation_*") if p.name not in current)
    if stale:
        warnings.warn(f"Files in {tables_dir} not written by this run (likely stale; delete them): {stale}")
    manifest = {
        "files": {_rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
        "stale_validation_files_not_written_by_this_run": stale,
        "python_version": sys.version.split()[0],
    }
    (tables_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    print(f"Wrote output manifest to {tables_dir / MANIFEST_NAME}")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    p.add_argument("--cleaning-report", type=Path, default=DEFAULT_CLEANING_REPORT)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--val-predictions", type=Path, default=None,
                   help="Pavel's validation-period predictions; if omitted, refit and verify.")
    p.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--validation-cutoff", default=None, help="Default: Pavel's metadata.")
    p.add_argument("--test-cutoff", default=None, help="Default: Pavel's metadata.")
    p.add_argument("--seed", dest="random_seed", type=int, default=None, help="Default: Pavel's metadata.")
    p.add_argument("--alt-validation-cutoff", default="2016-01-01")
    p.add_argument("--alt-test-cutoff", default="2017-01-01")
    p.add_argument("--mature-validation-cutoff", default="2014-01-01")
    p.add_argument("--mature-test-cutoff", default="2015-01-01")
    p.add_argument("--fn-fp-cost-ratio", type=float, default=5.0)
    p.add_argument("--target-approval-rate", type=float, default=0.80)
    p.add_argument("--threshold-grid", nargs=3, type=float, default=(0.01, 0.80, 159),
                   metavar=("MIN", "MAX", "STEPS"))
    p.add_argument("--subgroup-cols", nargs="+", default=list(SUBGROUP_COLUMNS))
    p.add_argument("--min-subgroup-n", type=int, default=200)
    p.add_argument("--min-subgroup-events", type=int, default=30)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--secondary-bootstrap-resamples", type=int, default=500)
    p.add_argument("--refit-tolerance", type=float, default=1e-6)
    p.add_argument("--test-period-issued-loans", type=int, default=None,
                   help="All loans issued on/after the test cutoff before status exclusion "
                        "(from Member 1); used to report the share of test-period loans retained.")
    p.add_argument("--member4-mature-complete-case-rows", type=int, default=None,
                   help="Member 4's complete-case maturity-cohort size, for cross-checking.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for label, path in (("Cleaned cohort", args.cleaned), ("Out-of-time predictions", args.predictions)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}. Run cleaning and 04_modelling_pavel.py first.")
    check_predictor_contract()
    settings = resolve_settings(args)
    seed = settings["random_seed"]

    cohort = load_cohort(args.cleaned, args.subgroup_cols)
    preds, verification = verify_predictions(cohort, pd.read_csv(args.predictions),
                                             settings["test_cutoff"], args.subgroup_cols)
    model_id = str(preds.loc[preds["selected_final_model"], "model"].iat[0])
    if settings["metadata_selected_model"] not in (None, model_id):
        raise ValueError(f"Predictions select '{model_id}' but metadata records "
                         f"'{settings['metadata_selected_model']}'.")
    candidate_ids = {spec.candidate_id for spec in prespecified_candidates()}
    if model_id not in candidate_ids:
        raise ValueError(f"Selected model '{model_id}' is not a prespecified candidate: {sorted(candidate_ids)}")

    overall_rows = []
    for name, grp in preds.groupby("model", sort=False):
        row = bootstrap_metrics(grp["default"].to_numpy("int8"), grp["predicted_probability"].to_numpy(),
                                args.bootstrap_resamples, seed, calibration=True)
        overall_rows.append({"model": name, "selected_final_model": name == model_id, **row})
    overall = pd.DataFrame(overall_rows)
    paired = paired_model_deltas(preds, model_id, args.bootstrap_resamples, seed)

    final_test = preds.loc[preds["model"] == model_id].copy()
    y_val, p_val, val_note = obtain_validation_probabilities(
        cohort, final_test, args.val_predictions, model_id, settings, args.refit_tolerance)
    lo, hi, steps = args.threshold_grid
    val_curve, thresholds = evaluate_thresholds(
        y_val, p_val, final_test["default"].to_numpy("int8"),
        final_test["predicted_probability"].to_numpy(), args.fn_fp_cost_ratio,
        args.target_approval_rate, np.round(np.linspace(lo, hi, int(steps)), 6),
        args.secondary_bootstrap_resamples, seed)

    subgroups = evaluate_subgroups(final_test, args.subgroup_cols, args.min_subgroup_n,
                                   args.min_subgroup_events, args.secondary_bootstrap_resamples, seed)
    subgroup_missing = {c: int(final_test[c].isna().sum())
                        for c in args.subgroup_cols if c in final_test.columns}
    primary = overall.loc[overall["selected_final_model"]].iloc[0].to_dict()
    robustness = run_robustness_checks(cohort, args.comparison, model_id, primary, settings, args)
    vintage, missing = drift_tables(cohort, settings["validation_cutoff"], settings["test_cutoff"])
    bins = generate_figures(preds, model_id, val_curve, thresholds, args.figures_dir)

    test_mask = cohort["issue_dt"] >= pd.Timestamp(settings["test_cutoff"])
    test_dates = cohort.loc[test_mask, "issue_dt"]
    max_fu, min_fu = _months_to_end(test_dates.min()), _months_to_end(test_dates.max())
    cc_fields = [c for c in COMPLETE_CASE_FIELDS if c in cohort.columns]
    test_cc_missing = int(cohort.loc[test_mask, cc_fields].isna().any(axis=1).sum())
    test_cc_by_field = {c: int(cohort.loc[test_mask, c].isna().sum()) for c in cc_fields}

    mat_row = _rob_row(robustness, "Outcome maturity")
    mat_cc = _diag(mat_row).get("complete_case_rows") if mat_row else None
    if (args.member4_mature_complete_case_rows is not None and mat_cc is not None
            and mat_cc != args.member4_mature_complete_case_rows):
        warnings.warn(f"Maturity complete-case rows {mat_cc:,} != Member 4's "
                      f"{args.member4_mature_complete_case_rows:,}; reconcile with Member 4.")
    if args.test_period_issued_loans is None:
        warnings.warn("--test-period-issued-loans not supplied; share of test-period loans retained "
                      "is omitted from the limitations section.")

    tables = {
        "validation_overall_metrics.csv": overall,
        "validation_paired_model_deltas.csv": paired,
        "validation_threshold_curve_validation.csv": val_curve,
        "validation_threshold_performance.csv": thresholds,
        "validation_calibration_bins.csv": bins,
        "validation_subgroup_performance.csv": subgroups,
        "validation_robustness_checks.csv": robustness,
        "validation_vintage_default_rates.csv": vintage,
        "validation_missingness_by_partition.csv": missing,
    }
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.tables_dir / name, index=False)
    print(f"Saved {len(tables)} validation tables to {args.tables_dir}")

    cleaning_report = {}
    if args.cleaning_report.is_file():
        cleaning_report = json.loads(args.cleaning_report.read_text(encoding="utf-8"))
    else:
        warnings.warn(f"{args.cleaning_report} not found; exclusion counts omitted from limitations.")
    write_report(args.report_path, {
        "args": args, "settings": settings, "model_id": model_id, "overall": overall,
        "paired": paired, "val_note": val_note, "thresholds": thresholds, "subgroups": subgroups,
        "robustness": robustness, "vintage": vintage, "missing": missing,
        "cleaning_report": cleaning_report, "verification": verification,
        "val_prevalence": float(y_val.mean()), "grid": (float(lo), float(hi), int(steps)),
        "subgroup_missing": subgroup_missing, "cc_fields": cc_fields,
        "test_cc_missing": test_cc_missing, "test_cc_by_field": test_cc_by_field,
        "max_followup_months": max_fu, "min_followup_months": min_fu,
    })
    write_manifest(args.tables_dir, args.figures_dir, args.report_path, list(tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())