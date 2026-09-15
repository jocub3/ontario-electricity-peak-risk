"""Memory-conscious SHAP calculations for preprocessed tree pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import feature_family
from .importance import transformed_to_public_feature


def _require_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for EXPL_02 onward. Install the `shap` "
            "package in the project environment before running these notebooks."
        ) from exc
    return shap


def transformed_matrix(pipeline, X: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    names = list(preprocessor.get_feature_names_out())
    return transformed, names


def _normalize_shap_array(values, task: str) -> np.ndarray:
    """
    Normalize SHAP outputs to rows x transformed_features.

    Binary tree classifiers differ across SHAP versions:
    - rows x features
    - list[class] of rows x features
    - rows x features x classes
    """
    if isinstance(values, list):
        array = np.asarray(values[-1] if task == "xgb" else values[0])
    else:
        array = np.asarray(values)

    if array.ndim == 3:
        # For binary classification use positive class; for regression this
        # branch is not expected.
        array = array[:, :, -1]

    if array.ndim != 2:
        raise ValueError(
            f"Unsupported SHAP output shape for {task}: {array.shape}"
        )
    return array


def compute_tree_shap(
    pipeline,
    X: pd.DataFrame,
    metadata: dict,
    config: dict,
    *,
    task: str,
    horizon: int,
) -> dict[str, object]:
    shap = _require_shap()

    transformed, transformed_names = transformed_matrix(pipeline, X)
    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(transformed)
    values = _normalize_shap_array(raw_values, task)

    if values.shape[1] != len(transformed_names):
        raise ValueError(
            "SHAP transformed feature count does not match preprocessor output."
        )

    mapping = [
        transformed_to_public_feature(name, metadata)
        for name in transformed_names
    ]

    # Aggregate one-hot contributions by original public feature for each row.
    public_names = list(dict.fromkeys(mapping))
    public_values = np.zeros((len(X), len(public_names)), dtype=float)

    for source_index, public_name in enumerate(mapping):
        target_index = public_names.index(public_name)
        public_values[:, target_index] += values[:, source_index]

    global_rows = []
    for index, public_name in enumerate(public_names):
        global_rows.append(
            {
                "task": task,
                "horizon": int(horizon),
                "feature": public_name,
                "feature_family": feature_family(public_name, config),
                "mean_abs_shap": float(
                    np.mean(np.abs(public_values[:, index]))
                ),
                "mean_shap": float(np.mean(public_values[:, index])),
            }
        )

    global_table = (
        pd.DataFrame(global_rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    public_shap = pd.DataFrame(public_values, columns=public_names)
    public_shap.insert(0, "row_id", range(len(public_shap)))

    return {
        "explainer": explainer,
        "transformed_values": transformed,
        "transformed_names": transformed_names,
        "shap_values_transformed": values,
        "public_feature_names": public_names,
        "shap_values_public": public_values,
        "public_shap_frame": public_shap,
        "global_table": global_table,
        "base_value": getattr(explainer, "expected_value", None),
    }


def feature_value_and_shap_long(
    X: pd.DataFrame,
    shap_result: dict,
    *,
    task: str,
    horizon: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    names = shap_result["public_feature_names"]
    values = shap_result["shap_values_public"]
    selected = features or names

    parts = []
    for feature in selected:
        if feature not in names or feature not in X.columns:
            continue
        index = names.index(feature)
        parts.append(
            pd.DataFrame(
                {
                    "task": task,
                    "horizon": int(horizon),
                    "row_id": np.arange(len(X)),
                    "feature": feature,
                    "feature_value": X[feature].to_numpy(),
                    "shap_value": values[:, index],
                }
            )
        )

    return (
        pd.concat(parts, ignore_index=True)
        if parts else
        pd.DataFrame(
            columns=[
                "task", "horizon", "row_id",
                "feature", "feature_value", "shap_value",
            ]
        )
    )
