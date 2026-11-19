"""Binomial GLM fitting and inference extraction for Member 4.

Statsmodels 0.14.x accepts ``cov_type="HC3"`` for GLM but computes a white
HC0 covariance internally. This module therefore computes the prespecified HC3
score-sandwich covariance explicitly and in chunks.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, ValueWarning

HC3_MIN_DENOMINATOR = 1e-8


@dataclass
class InferenceFit:
    """Small extracted artefacts from one fitted model."""

    coefficients: pd.DataFrame
    global_tests: pd.DataFrame
    audit: dict[str, object]


def _fdr_qvalues(p_values: pd.Series) -> np.ndarray:
    q_values = np.full(len(p_values), np.nan, dtype="float64")
    finite = np.isfinite(p_values.to_numpy(dtype="float64"))
    if finite.any():
        q_values[finite] = multipletests(
            p_values.to_numpy(dtype="float64")[finite],
            alpha=0.05,
            method="fdr_bh",
        )[1]
    return q_values


def glm_score_sandwich_covariance(
    result,
    *,
    adjustment: str = "HC3",
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Compute HC0 or leverage-adjusted HC3 covariance without an n-by-n matrix.

    For a canonical binomial-logit GLM, each observation's score contribution is
    ``x_i * (y_i - mu_i)``. HC3 divides the scalar score residual by
    ``1 - h_i`` before forming the sandwich meat, where ``h_i`` is the diagonal
    of the weighted GLM hat matrix.
    """
    if adjustment not in {"HC0", "HC3"}:
        raise ValueError("adjustment must be HC0 or HC3")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if float(result.scale) != 1.0:
        raise ValueError("The custom covariance requires binomial scale=1")

    model = result.model
    exog = np.asarray(model.exog, dtype="float64")
    params = np.asarray(result.params, dtype="float64")
    bread = np.asarray(result.normalized_cov_params, dtype="float64")
    if bread.shape != (exog.shape[1], exog.shape[1]):
        raise ValueError("Unexpected normalized covariance shape")

    score_factor = np.asarray(model.score_factor(params, scale=1.0), dtype="float64")
    hessian_factor = np.asarray(
        model.hessian_factor(params, scale=1.0, observed=False),
        dtype="float64",
    )
    if score_factor.shape != (len(exog),) or hessian_factor.shape != (len(exog),):
        raise ValueError("Unexpected GLM score or Hessian factor shape")

    meat = np.zeros_like(bread)
    leverage_min = np.inf
    leverage_max = -np.inf
    clipped_count = 0

    for start in range(0, len(exog), chunk_size):
        stop = min(start + chunk_size, len(exog))
        x_chunk = exog[start:stop]
        x_bread = x_chunk @ bread
        leverage = hessian_factor[start:stop] * np.einsum(
            "ij,ij->i",
            x_bread,
            x_chunk,
        )
        leverage_min = min(leverage_min, float(np.min(leverage)))
        leverage_max = max(leverage_max, float(np.max(leverage)))

        if adjustment == "HC3":
            denominator = 1.0 - leverage
            clipped_count += int((denominator < HC3_MIN_DENOMINATOR).sum())
            denominator = np.maximum(denominator, HC3_MIN_DENOMINATOR)
        else:
            denominator = np.ones_like(leverage)

        adjusted_score = score_factor[start:stop] / denominator
        score_rows = x_chunk * adjusted_score[:, None]
        meat += score_rows.T @ score_rows

    covariance = bread @ meat @ bread.T
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise FloatingPointError(f"{adjustment} covariance contains nonfinite values")

    audit: dict[str, float | int | str] = {
        "covariance_type": adjustment,
        "hc3_chunk_size": int(chunk_size),
        "leverage_min": float(leverage_min),
        "leverage_max": float(leverage_max),
        "leverage_denominator_clipped_count": int(clipped_count),
    }
    return covariance, audit


def _parameter_terms(model) -> list[str]:
    term_by_column = [""] * model.exog.shape[1]
    design_info = _formula_design_info(model)
    for term, column_slice in design_info.term_name_slices.items():
        for index in range(column_slice.start, column_slice.stop):
            term_by_column[index] = str(term)
    if any(not term for term in term_by_column):
        raise ValueError("Could not map every parameter to a formula term")
    return term_by_column


def _formula_design_info(model):
    """Return Patsy term metadata across statsmodels 0.14 and 0.15."""
    design_info = getattr(model.data, "model_spec", None)
    if design_info is None:
        design_info = getattr(model.data, "design_info", None)
    if design_info is None or not hasattr(design_info, "term_name_slices"):
        raise AttributeError("Formula design metadata is unavailable")
    return design_info


def _coefficient_table(
    result,
    covariance: np.ndarray,
    *,
    scenario: str,
    model_name: str,
) -> pd.DataFrame:
    params = np.asarray(result.params, dtype="float64")
    variance = np.diag(covariance)
    if (variance <= 0).any() or not np.isfinite(variance).all():
        raise FloatingPointError("Robust covariance has invalid diagonal entries")
    robust_se = np.sqrt(variance)
    z_statistic = params / robust_se
    p_values = 2 * stats.norm.sf(np.abs(z_statistic))
    critical = stats.norm.ppf(0.975)
    ci_low = params - critical * robust_se
    ci_high = params + critical * robust_se

    table = pd.DataFrame(
        {
            "scenario": scenario,
            "model": model_name,
            "term": _parameter_terms(result.model),
            "parameter": list(result.model.exog_names),
            "coefficient": params,
            "robust_se": robust_se,
            "z_statistic": z_statistic,
            "p_value": p_values,
            "ci_log_odds_low": ci_low,
            "ci_log_odds_high": ci_high,
        }
    )
    with np.errstate(over="ignore"):
        table["odds_ratio"] = np.exp(table["coefficient"])
        table["or_ci_low"] = np.exp(table["ci_log_odds_low"])
        table["or_ci_high"] = np.exp(table["ci_log_odds_high"])

    table["q_value"] = np.nan
    tested = table["parameter"].ne("Intercept")
    table.loc[tested, "q_value"] = _fdr_qvalues(table.loc[tested, "p_value"])
    return table


def _global_test_table(
    result,
    covariance: np.ndarray,
    *,
    scenario: str,
    model_name: str,
) -> pd.DataFrame:
    params = np.asarray(result.params, dtype="float64")
    records: list[dict[str, object]] = []
    design_info = _formula_design_info(result.model)
    for term, column_slice in design_info.term_name_slices.items():
        indices = np.arange(column_slice.start, column_slice.stop)
        if not len(indices):
            continue
        term_covariance = covariance[np.ix_(indices, indices)]
        rank = int(np.linalg.matrix_rank(term_covariance))
        if rank != len(indices):
            raise ValueError(
                f"Robust covariance for term {term!s} has rank {rank}, expected {len(indices)}"
            )
        term_params = params[indices]
        statistic = float(
            term_params.T
            @ np.linalg.pinv(term_covariance, hermitian=True)
            @ term_params
        )
        records.append(
            {
                "scenario": scenario,
                "model": model_name,
                "term": str(term),
                "df": rank,
                "wald_chi2": statistic,
                "p_value": float(stats.chi2.sf(statistic, rank)),
            }
        )

    table = pd.DataFrame.from_records(records)
    table["q_value"] = np.nan
    tested = table["term"].ne("Intercept")
    table.loc[tested, "q_value"] = _fdr_qvalues(table.loc[tested, "p_value"])
    return table


def _design_diagnostics(
    exog: np.ndarray,
    names: list[str],
    *,
    maximum_rows: int = 50_000,
) -> dict[str, object]:
    if len(exog) > maximum_rows:
        indices = np.linspace(0, len(exog) - 1, maximum_rows, dtype="int64")
        sample = np.asarray(exog[indices], dtype="float64")
    else:
        sample = np.asarray(exog, dtype="float64")

    keep = [index for index, name in enumerate(names) if name != "Intercept"]
    sample = sample[:, keep]
    kept_names = [names[index] for index in keep]
    means = sample.mean(axis=0)
    standard_deviations = sample.std(axis=0, ddof=1)
    nonconstant = standard_deviations > 0
    if not nonconstant.all():
        constants = [
            kept_names[index]
            for index, is_nonconstant in enumerate(nonconstant)
            if not is_nonconstant
        ]
        raise ValueError(f"Constant non-intercept design columns: {constants}")

    standardized = (sample - means) / standard_deviations
    correlation = standardized.T @ standardized / (len(sample) - 1)
    eigenvalues = np.linalg.eigvalsh(correlation)
    positive = eigenvalues[eigenvalues > np.finfo(float).eps]
    if len(positive) != len(eigenvalues):
        raise ValueError("Sampled standardized design matrix is rank deficient")
    condition_number = float(np.sqrt(positive.max() / positive.min()))
    inverse_correlation = np.linalg.pinv(correlation, hermitian=True)
    vif = np.diag(inverse_correlation)
    maximum_vif_index = int(np.argmax(vif))
    return {
        "diagnostic_sample_rows": int(len(sample)),
        "standardized_condition_number": condition_number,
        "maximum_vif": float(vif[maximum_vif_index]),
        "maximum_vif_parameter": kept_names[maximum_vif_index],
        "parameters_with_vif_over_10": [
            kept_names[index] for index, value in enumerate(vif) if value > 10
        ],
    }


def fit_inference_model(
    frame: pd.DataFrame,
    formula: str,
    *,
    scenario: str,
    model_name: str,
    hc3_chunk_size: int = 50_000,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> InferenceFit:
    """Fit one prespecified model and retain only small extracted artefacts."""
    model = None
    result = None
    try:
        model = smf.glm(
            formula=formula,
            data=frame,
            family=sm.families.Binomial(),
            missing="raise",
        )
        caught_warning_messages: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("error", category=PerfectSeparationWarning)
            warnings.filterwarnings("error", category=ValueWarning)
            result = model.fit(
                method="IRLS",
                maxiter=maximum_iterations,
                tol=tolerance,
                wls_method="lstsq",
                cov_type="nonrobust",
                use_t=False,
                disp=False,
            )
            caught_warning_messages = [str(item.message) for item in caught]

        if not result.converged:
            raise RuntimeError(f"{scenario}/{model_name} did not converge")
        if int(result.df_model) + 1 != model.exog.shape[1]:
            raise ValueError(
                f"{scenario}/{model_name} design matrix is rank deficient: "
                f"rank={int(result.df_model) + 1}, columns={model.exog.shape[1]}"
            )

        covariance, covariance_audit = glm_score_sandwich_covariance(
            result,
            adjustment="HC3",
            chunk_size=hc3_chunk_size,
        )
        coefficients = _coefficient_table(
            result,
            covariance,
            scenario=scenario,
            model_name=model_name,
        )
        global_tests = _global_test_table(
            result,
            covariance,
            scenario=scenario,
            model_name=model_name,
        )
        diagnostics = _design_diagnostics(model.exog, list(model.exog_names))
        iterations = result.fit_history.get("iteration")
        audit: dict[str, object] = {
            "scenario": scenario,
            "model": model_name,
            "formula": formula,
            "nobs": int(result.nobs),
            "default_count": int(frame["default"].sum()),
            "default_rate": float(frame["default"].mean()),
            "parameter_count": int(len(result.params)),
            "df_model": int(result.df_model),
            "converged": bool(result.converged),
            "iterations": int(iterations) if iterations is not None else None,
            "deviance": float(result.deviance),
            "log_likelihood": float(result.llf),
            "aic": float(result.aic),
            "maximum_absolute_coefficient": float(np.max(np.abs(result.params))),
            "statsmodels_version": statsmodels.__version__,
            "fit_warnings": caught_warning_messages,
            **covariance_audit,
            **diagnostics,
        }
        return InferenceFit(
            coefficients=coefficients,
            global_tests=global_tests,
            audit=audit,
        )
    finally:
        result = None
        model = None
        gc.collect()
