"""Shared predictive-modelling pipeline for Pavel's out-of-time analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 5106
RARE_LEVEL = "OTHER_RARE"
RARE_MIN_COUNT = 1_000
MISSING_CATEGORY = "MISSING"
WINSOR_LOWER = 0.005
WINSOR_UPPER = 0.995

INPUT_COLUMNS = (
    "id",
    "issue_date",
    "default",
    "loan_amnt",
    "annual_inc",
    "fico_score",
    "dti",
    "revol_util",
    "credit_history_months",
    "emp_length_years",
    "emp_length_years_was_missing",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "term_months",
    "purpose",
    "home_ownership",
    "verification_status",
    "application_type",
    "int_rate",
)
NUMERIC_FEATURES = (
    "fico_per_10",
    "log2_annual_inc",
    "log2_loan_to_annual_income",
    "income_or_dti_missing",
    "dti_per_10",
    "revol_util_per_10",
    "credit_history_years",
    "emp_length_years",
    "emp_length_years_was_missing",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc_per_5",
    "pub_rec",
    "int_rate_per_5",
)
CATEGORICAL_FEATURES = (
    "term_months",
    "purpose_grouped",
    "home_ownership_grouped",
    "verification_status",
    "application_type",
)
RARE_GROUP_FEATURES = ("purpose_grouped", "home_ownership_grouped")
WINSOR_FEATURES = (
    "log2_annual_inc",
    "log2_loan_to_annual_income",
    "dti_per_10",
    "revol_util_per_10",
    "credit_history_years",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc_per_5",
    "pub_rec",
)
REFERENCE_LEVELS = {
    "term_months": "36",
    "purpose_grouped": "debt_consolidation",
    "home_ownership_grouped": "MORTGAGE",
    "verification_status": "Not Verified",
    "application_type": "Individual",
}
FEATURE_UNITS = {
    "fico_per_10": "10 FICO points",
    "log2_annual_inc": "doubling of annual income",
    "log2_loan_to_annual_income": "doubling of 1 + loan amount / annual income",
    "income_or_dti_missing": "1 versus 0",
    "dti_per_10": "10 DTI percentage points",
    "revol_util_per_10": "10 utilization percentage points",
    "credit_history_years": "1 year of credit history",
    "emp_length_years": "1 year of employment",
    "emp_length_years_was_missing": "1 versus 0",
    "delinq_2yrs": "1 delinquency in the previous two years",
    "inq_last_6mths": "1 inquiry in the previous six months",
    "open_acc_per_5": "5 open accounts",
    "pub_rec": "1 derogatory public record",
    "int_rate_per_5": "5 interest-rate percentage points",
}


def validate_input(frame: pd.DataFrame) -> None:
    missing = sorted(set(INPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Modelling input is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Modelling input is empty")
    outcome = pd.to_numeric(frame["default"], errors="coerce")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError("default must contain only nonmissing 0 and 1 values")
    dates = pd.to_datetime(frame["issue_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Temporal modelling requires a valid issue_date for every row")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("id must be nonmissing and unique in the cleaned cohort")


def temporal_partitions(
    frame: pd.DataFrame,
    validation_cutoff: str | pd.Timestamp = "2017-01-01",
    test_cutoff: str | pd.Timestamp = "2018-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_date = pd.Timestamp(validation_cutoff)
    test_date = pd.Timestamp(test_cutoff)
    if validation_date >= test_date:
        raise ValueError("validation_cutoff must be earlier than test_cutoff")
    dates = pd.to_datetime(frame["issue_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Temporal splitting requires valid issue dates")
    train = frame.loc[dates < validation_date].copy()
    validation = frame.loc[(dates >= validation_date) & (dates < test_date)].copy()
    test = frame.loc[dates >= test_date].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Each temporal partition must contain at least one row")
    for name, partition in (("training", train), ("validation", validation)):
        if partition["default"].nunique() != 2:
            raise ValueError(f"The {name} partition must contain both outcome classes")
    return train, validation, test


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_source = {
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in (
            "loan_amnt",
            "annual_inc",
            "fico_score",
            "dti",
            "revol_util",
            "credit_history_months",
            "emp_length_years",
            "emp_length_years_was_missing",
            "delinq_2yrs",
            "inq_last_6mths",
            "open_acc",
            "pub_rec",
            "term_months",
            "int_rate",
        )
    }
    positive_income = numeric_source["annual_inc"].gt(0) & numeric_source[
        "annual_inc"
    ].notna()
    ratio = numeric_source["loan_amnt"] / numeric_source["annual_inc"].where(
        positive_income
    )
    features = pd.DataFrame(index=frame.index)
    features["fico_per_10"] = numeric_source["fico_score"] / 10
    features["log2_annual_inc"] = np.log2(
        numeric_source["annual_inc"].where(positive_income)
    )
    features["log2_loan_to_annual_income"] = np.log2(1 + ratio)
    features["income_or_dti_missing"] = (
        (~positive_income) | numeric_source["dti"].isna()
    ).astype("int8")
    features["dti_per_10"] = numeric_source["dti"] / 10
    features["revol_util_per_10"] = numeric_source["revol_util"] / 10
    features["credit_history_years"] = numeric_source["credit_history_months"] / 12
    features["emp_length_years"] = numeric_source["emp_length_years"]
    features["emp_length_years_was_missing"] = numeric_source[
        "emp_length_years_was_missing"
    ]
    features["delinq_2yrs"] = numeric_source["delinq_2yrs"]
    features["inq_last_6mths"] = numeric_source["inq_last_6mths"]
    features["open_acc_per_5"] = numeric_source["open_acc"] / 5
    features["pub_rec"] = numeric_source["pub_rec"]
    features["int_rate_per_5"] = numeric_source["int_rate"] / 5
    features = features.astype("float32")
    features["term_months"] = numeric_source["term_months"].astype("Int64").astype(
        "string"
    )
    features["purpose_grouped"] = frame["purpose"].astype("string")
    features["home_ownership_grouped"] = frame["home_ownership"].astype("string")
    features["verification_status"] = frame["verification_status"].astype("string")
    features["application_type"] = frame["application_type"].astype("string")
    return features.loc[:, [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: tuple[str, ...] = WINSOR_FEATURES,
        feature_names: tuple[str, ...] = NUMERIC_FEATURES,
        lower_quantile: float = WINSOR_LOWER,
        upper_quantile: float = WINSOR_UPPER,
    ) -> None:
        self.columns = columns
        self.feature_names = feature_names
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        values = np.asarray(X)
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_bounds_ = np.full(values.shape[1], -np.inf)
        self.upper_bounds_ = np.full(values.shape[1], np.inf)
        for column in self.columns:
            index = self.feature_names.index(column)
            lower, upper = np.quantile(
                values[:, index], [self.lower_quantile, self.upper_quantile]
            )
            self.lower_bounds_[index] = lower
            self.upper_bounds_[index] = upper
        return self

    def transform(self, X):
        values = np.asarray(X).copy()
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names, dtype=object)


class CategoricalPreparer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: tuple[str, ...] = CATEGORICAL_FEATURES,
        rare_columns: tuple[str, ...] = RARE_GROUP_FEATURES,
        rare_min_count: int = RARE_MIN_COUNT,
    ) -> None:
        self.columns = columns
        self.rare_columns = rare_columns
        self.rare_min_count = rare_min_count

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, list(self.columns)]
        return pd.DataFrame(X, columns=list(self.columns))

    def fit(self, X, y=None):
        if self.rare_min_count < 1:
            raise ValueError("rare_min_count must be positive")
        frame = self._as_frame(X)
        self.fill_values_ = {}
        self.allowed_levels_ = {}
        self.rare_levels_ = {}
        self.category_counts_ = {}
        for column in self.columns:
            values = frame[column].astype("string")
            nonmissing = values.dropna()
            if nonmissing.empty:
                raise ValueError(f"No observed categories available for {column}")
            fill_value = str(nonmissing.mode().iloc[0])
            self.fill_values_[column] = fill_value
            values = values.fillna(fill_value)
            counts = values.value_counts(dropna=False)
            self.category_counts_[column] = {
                str(level): int(count) for level, count in counts.sort_index().items()
            }
            if column in self.rare_columns:
                rare = {str(level) for level, count in counts.items() if count < self.rare_min_count}
                self.rare_levels_[column] = sorted(rare)
                self.allowed_levels_[column] = {
                    str(level) for level in counts.index if str(level) not in rare
                }
        return self

    def transform(self, X):
        frame = self._as_frame(X)
        output: dict[str, pd.Series] = {}
        for column in self.columns:
            values = frame[column].astype("string").fillna(self.fill_values_[column])
            if column in self.rare_columns:
                values = values.mask(
                    ~values.isin(self.allowed_levels_[column]), RARE_LEVEL
                )
            output[column] = values.astype(object)
        return pd.DataFrame(output, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


class TrainOnlyPreprocessor:
    def __init__(self, rare_min_count: int = RARE_MIN_COUNT) -> None:
        self.rare_min_count = rare_min_count
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("winsorize", Winsorizer()),
                ("scale", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            [
                (
                    "prepare",
                    CategoricalPreparer(rare_min_count=self.rare_min_count),
                ),
                (
                    "encode",
                    OneHotEncoder(
                        drop=[REFERENCE_LEVELS[column] for column in CATEGORICAL_FEATURES],
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype="float32",
                    ),
                ),
            ]
        )
        self.transformer = ColumnTransformer(
            [
                ("numeric", numeric, list(NUMERIC_FEATURES)),
                ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ],
            sparse_threshold=0,
            verbose_feature_names_out=False,
        )

    def fit(self, features: pd.DataFrame) -> "TrainOnlyPreprocessor":
        self.transformer.fit(features)
        self.feature_names_ = list(self.transformer.get_feature_names_out())
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.transformer.transform(features)
        array = np.asarray(transformed, dtype="float32")
        if not np.isfinite(array).all():
            raise ValueError("Preprocessing produced nonfinite predictor values")
        return array

    def audit(self) -> dict[str, Any]:
        numeric = self.transformer.named_transformers_["numeric"]
        imputer: SimpleImputer = numeric.named_steps["impute"]
        winsorizer: Winsorizer = numeric.named_steps["winsorize"]
        scaler: StandardScaler = numeric.named_steps["scale"]
        categorical: CategoricalPreparer = self.transformer.named_transformers_[
            "categorical"
        ].named_steps["prepare"]
        return {
            "numeric_medians": {
                column: float(value)
                for column, value in zip(NUMERIC_FEATURES, imputer.statistics_)
            },
            "winsor_quantiles": [WINSOR_LOWER, WINSOR_UPPER],
            "winsor_bounds": {
                column: {
                    "lower": float(winsorizer.lower_bounds_[index]),
                    "upper": float(winsorizer.upper_bounds_[index]),
                }
                for index, column in enumerate(NUMERIC_FEATURES)
                if column in WINSOR_FEATURES
            },
            "training_means": {
                column: float(value)
                for column, value in zip(NUMERIC_FEATURES, scaler.mean_)
            },
            "training_standard_deviations": {
                column: float(value)
                for column, value in zip(NUMERIC_FEATURES, scaler.scale_)
            },
            "categorical_fill_values": categorical.fill_values_,
            "rare_min_count": self.rare_min_count,
            "rare_levels": categorical.rare_levels_,
            "training_category_counts": categorical.category_counts_,
            "reference_levels": REFERENCE_LEVELS,
            "processed_feature_names": self.feature_names_,
        }


@dataclass(frozen=True)
class ModelSpecification:
    candidate_id: str
    family: str
    parameters: dict[str, Any]

    def make_estimator(self, seed: int = RANDOM_STATE):
        if self.family == "logistic_regression":
            return LogisticRegression(
                **self.parameters,
                solver="lbfgs",
                max_iter=500,
                tol=1e-6,
                random_state=seed,
            )
        if self.family == "random_forest":
            return RandomForestClassifier(
                **self.parameters,
                n_estimators=200,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            )
        if self.family == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                **self.parameters,
                learning_rate=0.08,
                max_iter=300,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
            )
        raise ValueError(f"Unknown model family: {self.family}")


def prespecified_candidates() -> list[ModelSpecification]:
    candidates: list[ModelSpecification] = []
    for c_value in (0.1, 1.0):
        for weight in (None, "balanced"):
            label = "none" if weight is None else weight
            candidates.append(
                ModelSpecification(
                    candidate_id=f"logistic_C{c_value:g}_{label}",
                    family="logistic_regression",
                    parameters={"C": c_value, "class_weight": weight},
                )
            )
    for leaf_size in (20, 100):
        for weight in (None, "balanced_subsample"):
            label = "none" if weight is None else "balanced"
            candidates.append(
                ModelSpecification(
                    candidate_id=f"random_forest_leaf{leaf_size}_{label}",
                    family="random_forest",
                    parameters={
                        "min_samples_leaf": leaf_size,
                        "class_weight": weight,
                    },
                )
            )
    for leaves in (31, 63):
        for weight in (None, "balanced"):
            label = "none" if weight is None else weight
            candidates.append(
                ModelSpecification(
                    candidate_id=f"hist_gradient_boosting_leaves{leaves}_{label}",
                    family="hist_gradient_boosting",
                    parameters={"max_leaf_nodes": leaves, "class_weight": weight},
                )
            )
    return candidates


def binary_metrics(outcome: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(outcome, dtype="int8")
    p = np.asarray(probability, dtype="float64")
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Outcomes and probabilities must have the same nonzero length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between zero and one")
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "default_rate": float(y.mean()),
    }


def logistic_interpretation(
    model: LogisticRegression,
    preprocessor: TrainOnlyPreprocessor,
) -> pd.DataFrame:
    if model.coef_.shape[1] != len(preprocessor.feature_names_):
        raise ValueError("Logistic coefficients do not match processed features")
    numeric_pipeline = preprocessor.transformer.named_transformers_["numeric"]
    scaler: StandardScaler = numeric_pipeline.named_steps["scale"]
    scaled = model.coef_[0].astype("float64")
    means = scaler.mean_.astype("float64")
    scales = scaler.scale_.astype("float64")
    original = scaled.copy()
    original[: len(NUMERIC_FEATURES)] = scaled[: len(NUMERIC_FEATURES)] / scales
    intercept = float(model.intercept_[0] - np.dot(scaled[: len(scales)], means / scales))
    rows: list[dict[str, Any]] = [
        {
            "parameter": "Intercept",
            "feature": "Intercept",
            "coefficient": intercept,
            "odds_ratio": float(np.exp(intercept)),
            "training_standardized_coefficient": np.nan,
            "comparison": "Log-odds at zero numeric values and all categorical references",
        }
    ]
    for index, name in enumerate(preprocessor.feature_names_):
        if index < len(NUMERIC_FEATURES):
            feature = NUMERIC_FEATURES[index]
            comparison = f"Per {FEATURE_UNITS[feature]}; training-only winsorization where specified"
        else:
            feature = next(
                column
                for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True)
                if name.startswith(f"{column}_")
            )
            level = name[len(feature) + 1 :]
            comparison = f"{level} versus {REFERENCE_LEVELS[feature]} (reference)"
        rows.append(
            {
                "parameter": name,
                "feature": feature,
                "coefficient": float(original[index]),
                "odds_ratio": float(np.exp(original[index])),
                "training_standardized_coefficient": float(scaled[index]),
                "comparison": comparison,
            }
        )
    return pd.DataFrame(rows)
