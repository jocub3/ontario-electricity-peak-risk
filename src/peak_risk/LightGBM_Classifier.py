"""Peak-risk classifier: predicts whether a forecasted hour will exceed the peak threshold.

Standard `LGBMClassifier`, row-wise over (fsa, forecast_timestamp, horizon), not multi-output
like the regressor. Uses the regressor's forecasted consumption plus calendar features of the
forecast timestamp (known in advance) and weather/lag/episode features observed at the origin
timestamp only, never weather from the future hour itself, since a real deployment would only
have a weather forecast for that hour, not the true observed value (see `CLAUDE.md`).
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier

from src.common.feature_engineering import add_temperature_episode_features
from src.common.folds import Fold, split_fold
from src.common.peak_risk import compute_peak_thresholds, label_peaks
from src.forecast.LightGBM_Regressor import (
    build_model as build_regressor_model,
    get_feature_columns as get_regressor_feature_columns,
    prepare_categoricals as prepare_regressor_categoricals,
    predict_long_format as predict_regressor_long_format,
    TARGET_COLUMNS,
)

ORIGIN_TIMESTAMP_COLUMN = "origin_timestamp"
FORECAST_TIMESTAMP_COLUMN = "forecast_timestamp"

# Calendar columns of the forecast timestamp: deterministic, known ahead of time, so safe to use
# as features even though they describe the future hour being classified.
CALENDAR_FEATURES = [
    "season",
    "year",
    "quarter",
    "month",
    "week_of_year",
    "day_of_year",
    "day_of_month",
    "hour",
    "weekday",
    "is_weekend",
    "is_workday",
    "is_monday",
    "is_friday",
    "is_business_hour",
    "is_peak_hour_window",
    "is_month_start",
    "is_month_end",
    "is_year_start",
    "is_year_end",
    "is_public_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "days_to_holiday",
    "days_after_holiday",
    "is_long_weekend",
    "is_daylight_saving_time",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

# Weather, consumption, and lag columns of the origin timestamp: the last real-world information
# available at the moment a forecast is issued.
ORIGIN_FEATURES = [
    "electricity_consumption",
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
    "visibility_km",
    "consumption_lag_1h",
    "consumption_lag_2h",
    "consumption_lag_3h",
    "consumption_lag_24h",
    "consumption_lag_48h",
    "consumption_lag_168h",
    "consumption_roll_mean_24h",
    "consumption_roll_std_24h",
    "consumption_roll_mean_168h",
    "consumption_roll_std_168h",
    "heat_episode_hours",
    "cold_episode_hours",
]

CATEGORICAL_COLUMNS = ["fsa", "season"]
MODEL_NAME = "LightGBM_Classifier"


def build_regressor_forecasts(dataset: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Forecasted consumption for every row in a fold, train and test alike.

    `lightgbm_regressor_predictions.parquet` only covers each fold's test split, but the
    classifier also needs a forecast for its own training rows, so this trains the fold's
    regressor once and predicts on both splits.
    """
    train, test = split_fold(dataset, fold, timestamp_column="timestamp_local")
    train_clean = prepare_regressor_categoricals(train).dropna(subset=TARGET_COLUMNS)
    train_ready = prepare_regressor_categoricals(train)
    test_ready = prepare_regressor_categoricals(test)

    feature_columns = get_regressor_feature_columns(dataset)
    model = build_regressor_model()
    model.fit(train_clean[feature_columns], train_clean[TARGET_COLUMNS])

    forecasts = pd.concat(
        [
            predict_regressor_long_format(model, feature_columns, train_ready),
            predict_regressor_long_format(model, feature_columns, test_ready),
        ],
        ignore_index=True,
    )
    return forecasts.rename(columns={"prediction": "forecast_consumption"})


def build_feature_table(forecasts: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """Join calendar features (forecast timestamp) and weather/lag features (origin timestamp)
    onto the regressor's forecasts, one row per (fsa, origin_timestamp, horizon)."""
    dataset = add_temperature_episode_features(dataset)

    calendar_at_forecast = dataset[["fsa", "timestamp_local", *CALENDAR_FEATURES]].rename(
        columns={"timestamp_local": FORECAST_TIMESTAMP_COLUMN}
    )
    origin_features = dataset[["fsa", "timestamp_local", *ORIGIN_FEATURES]].rename(
        columns={"timestamp_local": ORIGIN_TIMESTAMP_COLUMN}
    )

    table = forecasts.merge(calendar_at_forecast, on=["fsa", FORECAST_TIMESTAMP_COLUMN], how="left")
    table = table.merge(origin_features, on=["fsa", ORIGIN_TIMESTAMP_COLUMN], how="left")
    return table


def get_feature_columns() -> list[str]:
    return ["forecast_consumption", "horizon", "fsa", *CALENDAR_FEATURES, *ORIGIN_FEATURES]


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast `fsa`/`season` to pandas ``category`` dtype so LightGBM treats them natively."""
    df = df.copy()
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype("category")
    return df


def build_model(**lgbm_params) -> LGBMClassifier:
    params = {"random_state": 42, "verbosity": -1, "class_weight": "balanced"}
    params.update(lgbm_params)
    return LGBMClassifier(**params)


def train_fold(
    dataset: pd.DataFrame, fold: Fold, model_params: dict | None = None
) -> tuple[LGBMClassifier, list[str], pd.DataFrame]:
    """Train on one fold's training split. Returns the fitted model, the feature columns used,
    and the fold's labeled test split (still needed downstream to build predictions)."""
    forecasts = build_regressor_forecasts(dataset, fold)
    feature_table = build_feature_table(forecasts, dataset)

    train_dataset, _ = split_fold(dataset, fold, timestamp_column="timestamp_local")
    thresholds = compute_peak_thresholds(train_dataset)
    feature_table = feature_table.dropna(subset=["actual"])
    feature_table["electricity_consumption"] = feature_table["actual"]
    labeled = label_peaks(feature_table, thresholds)
    labeled = prepare_categoricals(labeled)

    train, test = split_fold(labeled, fold, timestamp_column=ORIGIN_TIMESTAMP_COLUMN)

    feature_columns = get_feature_columns()
    model = build_model(**(model_params or {}))
    model.fit(train[feature_columns], train["actual_peak"])
    return model, feature_columns, test


def predict_long_format(
    model: LGBMClassifier,
    feature_columns: list[str],
    test: pd.DataFrame,
    model_name: str = MODEL_NAME,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict peak probability and reshape to the team's peak-risk output contract:
    fsa, forecast_timestamp, actual_peak, peak_probability, predicted_peak, model_name."""
    peak_probability = model.predict_proba(test[feature_columns])[:, 1]
    return pd.DataFrame(
        {
            "fsa": test["fsa"].to_numpy(),
            "forecast_timestamp": test[FORECAST_TIMESTAMP_COLUMN].to_numpy(),
            "actual_peak": test["actual_peak"].to_numpy(),
            "peak_probability": peak_probability,
            "predicted_peak": (peak_probability >= threshold).astype(int),
            "model_name": model_name,
            # extra diagnostic columns, not part of the contract, already on hand from the feature table
            "origin_timestamp": test[ORIGIN_TIMESTAMP_COLUMN].to_numpy(),
            "horizon": test["horizon"].to_numpy(),
        }
    )
