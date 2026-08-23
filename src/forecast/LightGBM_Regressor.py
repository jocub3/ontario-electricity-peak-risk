"""Pooled multi-output LightGBM regressor: 24h-ahead consumption forecast, all 6 FSAs.

One `MultiOutputRegressor(LGBMRegressor(...))` trained per fold on `fsa_hourly_features.parquet`
(`src/common/feature_engineering.py`), predicting all 24 horizons (`target_h1` .. `target_h24`)
in a single `.fit()`/`.predict()` call. `fsa` and `season` are the categorical features.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.common.folds import Fold, split_fold

TARGET_COLUMNS = [f"target_h{horizon}" for horizon in range(1, 25)]
CATEGORICAL_COLUMNS = ["fsa", "season"]
ORIGIN_TIMESTAMP_COLUMN = "timestamp_local"

# Text columns dropped as redundant with numeric/cyclical features already in the dataset:
# `station` duplicates `fsa` one-to-one; `month_name`/`weekday_name`/`hour_group` duplicate
# `month`/`weekday`/`hour` (plus their sin/cos encodings); `holiday_name` duplicates the numeric
# `is_public_holiday`/`is_day_before_holiday`/`is_day_after_holiday` flags.
DROP_COLUMNS = [
    "station",
    "timestamp_utc",
    "timestamp_toronto",
    "date",
    "month_name",
    "weekday_name",
    "hour_group",
    "holiday_name",
]

MODEL_NAME = "LightGBM_Regressor"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the origin timestamp, the dropped/redundant columns, and the targets."""
    excluded = set(DROP_COLUMNS) | set(TARGET_COLUMNS) | {ORIGIN_TIMESTAMP_COLUMN}
    return [c for c in df.columns if c not in excluded]


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast `fsa`/`season` to pandas ``category`` dtype so LightGBM treats them natively."""
    df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("category")
    return df


def build_model(**lgbm_params) -> MultiOutputRegressor:
    params = {"random_state": 42, "verbosity": -1}
    params.update(lgbm_params)
    return MultiOutputRegressor(LGBMRegressor(**params))


def train_fold(
    df: pd.DataFrame, fold: Fold, model_params: dict | None = None
) -> tuple[MultiOutputRegressor, list[str], pd.DataFrame]:
    """Train on one fold's training split. Returns the fitted model, the feature columns used,
    and the fold's test split (still needed downstream to build predictions). ``model_params``
    overrides the default LightGBM hyperparameters, e.g. to compare against a tuned config."""
    train, test = split_fold(df, fold, timestamp_column=ORIGIN_TIMESTAMP_COLUMN)
    train = prepare_categoricals(train).dropna(subset=TARGET_COLUMNS)
    test = prepare_categoricals(test)

    feature_columns = get_feature_columns(df)
    model = build_model(**(model_params or {}))
    model.fit(train[feature_columns], train[TARGET_COLUMNS])
    return model, feature_columns, test


def predict_long_format(
    model: MultiOutputRegressor,
    feature_columns: list[str],
    test: pd.DataFrame,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Predict all 24 horizons and reshape to the team's forecast output contract:
    fsa, origin_timestamp, forecast_timestamp, horizon, actual, prediction, model_name."""
    predictions = model.predict(test[feature_columns])

    horizon_frames = [
        pd.DataFrame(
            {
                "fsa": test["fsa"].to_numpy(),
                "origin_timestamp": test[ORIGIN_TIMESTAMP_COLUMN].to_numpy(),
                "forecast_timestamp": test[ORIGIN_TIMESTAMP_COLUMN].to_numpy()
                + pd.Timedelta(hours=horizon),
                "horizon": horizon,
                "actual": test[f"target_h{horizon}"].to_numpy(),
                "prediction": predictions[:, horizon - 1],
                "model_name": model_name,
            }
        )
        for horizon in range(1, 25)
    ]
    return (
        pd.concat(horizon_frames, ignore_index=True)
        .sort_values(["fsa", "origin_timestamp", "horizon"])
        .reset_index(drop=True)
    )
