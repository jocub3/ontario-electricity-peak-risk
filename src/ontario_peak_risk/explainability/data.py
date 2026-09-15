"""Build reproducible explanation samples using the original model feature builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_modeling_config,
)
from src.ontario_peak_risk.models.forecasting.random_forest import (
    features as rf_features_module,
)
from src.ontario_peak_risk.models.peak_risk.xgboost import (
    features as xgb_features_module,
)

from .common import public_features, resolve_path


def load_historical_feature_dataset(
    config: dict,
    project_root: Path,
) -> pd.DataFrame:
    modeling_path = resolve_path(
        project_root,
        config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(modeling_path)
    return load_feature_dataset(modeling_config, project_root)


def candidate_origins(
    feature_dataset: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    start = pd.Timestamp(config["analysis"]["sample_period_start"])
    end = pd.Timestamp(config["analysis"]["sample_period_end"])

    frame = feature_dataset[["fsa", "timestamp"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.loc[frame["timestamp"].between(start, end)].copy()
    frame = frame.rename(columns={"timestamp": "forecast_origin"})
    return frame.drop_duplicates(["fsa", "forecast_origin"])


def sample_origins(
    feature_dataset: pd.DataFrame,
    config: dict,
    *,
    n_per_fsa: int | None = None,
) -> pd.DataFrame:
    n_per_fsa = int(
        n_per_fsa
        or config["analysis"]["origins_per_fsa_per_horizon"]
    )
    random_seed = int(config["analysis"]["random_seed"])
    candidates = candidate_origins(feature_dataset, config)

    parts = []
    for index, (fsa, group) in enumerate(
        candidates.groupby("fsa", observed=True)
    ):
        n = min(n_per_fsa, len(group))
        parts.append(
            group.sample(
                n=n,
                random_state=random_seed + index,
                replace=False,
            )
        )

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["fsa", "forecast_origin"])
        .reset_index(drop=True)
    )


def build_model_features(
    task: str,
    origins: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    horizon: int,
    metadata: dict,
) -> pd.DataFrame:
    if task == "rf":
        module = rf_features_module
    elif task == "xgb":
        module = xgb_features_module
    else:
        raise ValueError("task must be 'rf' or 'xgb'.")

    frame = module.build_horizon_features(
        origins,
        feature_dataset,
        horizon=int(horizon),
    )

    expected = public_features(metadata)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{task} h+{horizon} explanation frame missing features: {missing}"
        )

    # Explanations should use actual model-compatible rows, not imputed rows
    # created only because a sampling timestamp lacks required history.
    complete = frame[expected].notna().all(axis=1)
    return frame.loc[complete].reset_index(drop=True)


def cap_sample(
    frame: pd.DataFrame,
    max_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    if len(frame) <= int(max_rows):
        return frame.reset_index(drop=True)
    return (
        frame.sample(
            n=int(max_rows),
            random_state=int(random_seed),
            replace=False,
        )
        .sort_index()
        .reset_index(drop=True)
    )
