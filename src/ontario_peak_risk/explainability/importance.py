"""Built-in tree feature importance with one-hot aggregation."""

from __future__ import annotations

import pandas as pd

from .common import feature_family, public_features
from .model_access import load_horizon_pipeline


def transformed_to_public_feature(
    transformed_name: str,
    metadata: dict,
) -> str:
    name = str(transformed_name)

    if name in metadata["numeric_features"]:
        return name

    for categorical in metadata["categorical_features"]:
        if name == categorical or name.startswith(f"{categorical}_"):
            return categorical

    # Defensive fallback for ColumnTransformer prefixes if a future sklearn
    # version emits them despite verbose_feature_names_out=False.
    if "__" in name:
        short = name.split("__", 1)[1]
        if short in metadata["numeric_features"]:
            return short
        for categorical in metadata["categorical_features"]:
            if short == categorical or short.startswith(f"{categorical}_"):
                return categorical

    return name


def horizon_builtin_importance(
    pipeline,
    metadata: dict,
    *,
    task: str,
    horizon: int,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    names = list(preprocessor.get_feature_names_out())
    values = model.feature_importances_

    transformed = pd.DataFrame(
        {
            "task": task,
            "horizon": int(horizon),
            "transformed_feature": names,
            "importance": values,
        }
    )
    transformed["feature"] = transformed["transformed_feature"].map(
        lambda name: transformed_to_public_feature(name, metadata)
    )
    transformed["feature_family"] = transformed["feature"].map(
        lambda name: feature_family(name, config)
    )

    public = (
        transformed.groupby(
            ["task", "horizon", "feature", "feature_family"],
            observed=True,
        )["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    total = public["importance"].sum()
    if total > 0:
        public["importance_pct"] = public["importance"] / total * 100
    else:
        public["importance_pct"] = 0.0

    return public, transformed


def run_builtin_importance(
    task: str,
    horizons: list[int],
    metadata: dict,
    config: dict,
    project_root,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    public_parts = []
    transformed_parts = []

    for horizon in horizons:
        pipeline = load_horizon_pipeline(
            task, horizon, config, project_root
        )
        public, transformed = horizon_builtin_importance(
            pipeline,
            metadata,
            task=task,
            horizon=horizon,
            config=config,
        )
        public_parts.append(public)
        transformed_parts.append(transformed)

        # Load only one large horizon-specific artifact at a time.
        del pipeline

    return (
        pd.concat(public_parts, ignore_index=True),
        pd.concat(transformed_parts, ignore_index=True),
    )


def summarize_global_importance(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    return (
        importance.groupby(
            ["task", "feature", "feature_family"],
            observed=True,
        )
        .agg(
            mean_importance_pct=("importance_pct", "mean"),
            median_importance_pct=("importance_pct", "median"),
            min_importance_pct=("importance_pct", "min"),
            max_importance_pct=("importance_pct", "max"),
        )
        .reset_index()
        .sort_values(
            ["task", "mean_importance_pct"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )
