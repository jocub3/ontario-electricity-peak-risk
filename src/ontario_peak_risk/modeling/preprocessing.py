"""Shared preprocessing policies for model-specific pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


@dataclass(frozen=True)
class PreprocessingProfile:
    """Declarative preprocessing profile for one model family."""

    numeric_scaler: str
    categorical_encoding: str
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"


def get_preprocessing_profile(model_family: str) -> PreprocessingProfile:
    """Return the default preprocessing policy for a model family."""
    family = model_family.lower().strip()

    if family in {"linear", "linear_regression", "logistic", "logistic_regression", "neural", "neural_network"}:
        return PreprocessingProfile(
            numeric_scaler="standard",
            categorical_encoding="one_hot",
        )

    if family == "robust_linear":
        return PreprocessingProfile(
            numeric_scaler="robust",
            categorical_encoding="one_hot",
        )

    if family in {
        "tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "hist_gradient_boosting",
        "histgradientboostingclassifier",
    }:
        return PreprocessingProfile(
            numeric_scaler="none",
            categorical_encoding="model_specific",
        )

    if family in {"statistical", "sarimax"}:
        return PreprocessingProfile(
            numeric_scaler="model_specific",
            categorical_encoding="model_specific",
        )

    raise ValueError(f"Unknown model family: {model_family}")


def build_sklearn_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    profile: PreprocessingProfile,
) -> ColumnTransformer:
    """
    Build a generic sklearn preprocessing object.

    IMPORTANT:
    The returned object must be fitted on training data only.
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=profile.numeric_imputation))
    ]

    if profile.numeric_scaler == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif profile.numeric_scaler == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    elif profile.numeric_scaler == "none":
        pass
    else:
        raise ValueError(
            "This helper supports standard, robust, or no numeric scaling. "
            "Model-specific scaling belongs in the individual model branch."
        )

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=profile.categorical_imputation))
    ]

    if profile.categorical_encoding == "one_hot":
        categorical_steps.append(
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        )
    elif profile.categorical_encoding == "model_specific":
        raise ValueError(
            "Model-specific categorical encoding must be implemented "
            "inside the corresponding model branch."
        )
    else:
        raise ValueError(
            f"Unsupported categorical encoding: {profile.categorical_encoding}"
        )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessor_training_only(preprocessor, train_features: pd.DataFrame):
    """Fit a preprocessing object using training data only."""
    return preprocessor.fit(train_features)
