from __future__ import annotations

import numpy as np
import pandas as pd

from src.predictive_modelling import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    TrainOnlyPreprocessor,
    binary_metrics,
    engineer_features,
    logistic_interpretation,
    prespecified_candidates,
    temporal_partitions,
)


def synthetic_modelling_frame(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(5106)
    years = np.resize([2015, 2016, 2017, 2018], n)
    income = rng.uniform(30_000, 100_000, n)
    dti = rng.uniform(2, 35, n)
    dti[0] = np.nan
    employment = rng.uniform(0, 10, n)
    employment[1] = np.nan
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "issue_date": pd.to_datetime(pd.Series(years).astype(str) + "-06-01"),
            "default": ((np.arange(n) // 4) % 4 == 0).astype("int8"),
            "loan_amnt": rng.uniform(2_000, 30_000, n),
            "annual_inc": income,
            "fico_score": rng.uniform(650, 800, n),
            "dti": dti,
            "revol_util": rng.uniform(5, 95, n),
            "credit_history_months": rng.uniform(24, 360, n),
            "emp_length_years": employment,
            "emp_length_years_was_missing": pd.Series(employment).isna().astype("int8"),
            "delinq_2yrs": rng.integers(0, 3, n),
            "inq_last_6mths": rng.integers(0, 4, n),
            "open_acc": rng.integers(2, 20, n),
            "pub_rec": rng.integers(0, 2, n),
            "term_months": np.where(np.arange(n) % 3 == 0, 60, 36),
            "purpose": np.where(
                np.arange(n) % 11 == 0, "credit_card", "debt_consolidation"
            ),
            "home_ownership": np.where(np.arange(n) % 2 == 0, "MORTGAGE", "RENT"),
            "verification_status": np.where(
                np.arange(n) % 2 == 0, "Not Verified", "Verified"
            ),
            "application_type": "Individual",
            "int_rate": rng.uniform(5, 25, n),
        }
    )
    frame.loc[0, "home_ownership"] = "NONE"
    return frame.loc[:, list(INPUT_COLUMNS)]


def test_temporal_partitions_use_fixed_date_boundaries() -> None:
    frame = synthetic_modelling_frame()
    train, validation, test = temporal_partitions(frame)
    assert set(train["issue_date"].dt.year) == {2015, 2016}
    assert set(validation["issue_date"].dt.year) == {2017}
    assert set(test["issue_date"].dt.year) == {2018}
    assert len(train) + len(validation) + len(test) == len(frame)


def test_preprocessor_uses_training_values_and_handles_unseen_categories() -> None:
    frame = synthetic_modelling_frame()
    train, validation, _ = temporal_partitions(frame)
    validation = validation.copy()
    validation.iloc[0, validation.columns.get_loc("purpose")] = "unseen_future_category"
    validation.iloc[0, validation.columns.get_loc("home_ownership")] = "NONE"
    preprocessor = TrainOnlyPreprocessor(rare_min_count=20).fit(engineer_features(train))
    train_matrix = preprocessor.transform(engineer_features(train))
    validation_matrix = preprocessor.transform(engineer_features(validation))

    expected_median = float((train["fico_score"] / 10).median())
    audit = preprocessor.audit()
    assert np.isclose(audit["numeric_medians"]["fico_per_10"], expected_median)
    assert audit["rare_levels"]["purpose_grouped"] == ["credit_card"]
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(validation_matrix).all()
    rare_index = preprocessor.feature_names_.index("purpose_grouped_OTHER_RARE")
    assert validation_matrix[0, rare_index] == 1
    assert train_matrix.shape[1] == validation_matrix.shape[1]


def test_candidates_have_equal_tuning_budgets_and_shared_model_families() -> None:
    candidates = prespecified_candidates()
    assert len(candidates) == 12
    for family in {candidate.family for candidate in candidates}:
        family_candidates = [candidate for candidate in candidates if candidate.family == family]
        assert len(family_candidates) == 4
        assert any(candidate.parameters["class_weight"] is None for candidate in family_candidates)
        assert any(candidate.parameters["class_weight"] is not None for candidate in family_candidates)


def test_logistic_coefficients_are_returned_to_engineered_units() -> None:
    frame = synthetic_modelling_frame()
    train, validation, _ = temporal_partitions(frame)
    preprocessor = TrainOnlyPreprocessor(rare_min_count=20).fit(engineer_features(train))
    x_train = preprocessor.transform(engineer_features(train))
    x_validation = preprocessor.transform(engineer_features(validation))
    specification = prespecified_candidates()[0]
    model = specification.make_estimator()
    model.fit(x_train, train["default"])
    metrics = binary_metrics(
        validation["default"], model.predict_proba(x_validation)[:, 1]
    )
    interpretation = logistic_interpretation(model, preprocessor)

    assert 0 <= metrics["roc_auc"] <= 1
    assert metrics["brier_score"] >= 0
    assert len(interpretation) == len(preprocessor.feature_names_) + 1
    assert np.isfinite(interpretation["coefficient"]).all()
    fico = interpretation.loc[interpretation["feature"].eq("fico_per_10")].iloc[0]
    processed_index = preprocessor.feature_names_.index("fico_per_10")
    scale = preprocessor.transformer.named_transformers_["numeric"].named_steps[
        "scale"
    ].scale_[NUMERIC_FEATURES.index("fico_per_10")]
    assert np.isclose(fico["coefficient"], model.coef_[0, processed_index] / scale)
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
