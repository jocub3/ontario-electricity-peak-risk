"""XGBoost multi-output regressor for 24h-ahead consumption forecasting - Model 3 (pooled across FSAs)."""

import numpy as np
import pandas as pd
import xgboost as xgb

from common.feature_selection import SELECTED_FEATURES
from common.targets import HORIZONS, build_wide_targets
from common.transforms import add_log_target, set_categorical_dtypes

MODEL_NAME = "xgboost_regressor"

LOG_TARGET_COLUMNS = [f"actual_h{h}" for h in HORIZONS]
REAL_TARGET_COLUMNS = [f"real_h{h}" for h in HORIZONS]


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """SELECTED_FEATURES (at origin_timestamp) are merged with the 24-horizon wide targets.

    Two versions of the targets are kept: the log-scale columns (actual_h1..actual_h24, what
    the model is fit on) and the real-kWh columns (real_h1..real_h24, used only for reporting
    and metrics - the required output schema and forecast_metrics are in real kWh, not log).
    """
    df = add_log_target(df)
    df = set_categorical_dtypes(df)

    log_wide = build_wide_targets(df, value_col="log_consumption")
    real_wide = build_wide_targets(df, value_col="TOTAL_CONSUMPTION")
    real_wide = real_wide.rename(columns=dict(zip(LOG_TARGET_COLUMNS, REAL_TARGET_COLUMNS)))

    feature_cols = [c for c in SELECTED_FEATURES if c != "FSA"]
    features = df[["FSA", "timestamp_local"] + feature_cols].rename(
        columns={"timestamp_local": "origin_timestamp"}
    )

    table = log_wide.merge(real_wide, on=["FSA", "origin_timestamp"])
    table = table.merge(features, on=["FSA", "origin_timestamp"])
    table["year"] = table["origin_timestamp"].dt.year
    return table


def make_model() -> xgb.XGBRegressor:
    """First-pass hyperparameters, not tuned - `multi_strategy="one_output_per_tree"` is what
    gives XGBoost native multi-output support (one column per horizon) instead of needing a
    wrapper like sklearn's MultiOutputRegressor. `enable_categorical=True` lets `fsa` be used
    natively as a category column instead of one-hot encoded."""
    return xgb.XGBRegressor(
        tree_method="hist",
        multi_strategy="one_output_per_tree",
        enable_categorical=True,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """The model is fit on train_df, predicted on test_df, and long-form results are returned in the required output schema."""
    model = make_model()
    model.fit(train_df[SELECTED_FEATURES], train_df[LOG_TARGET_COLUMNS])

    log_pred = model.predict(test_df[SELECTED_FEATURES])
    real_pred = np.expm1(log_pred)

    frames = []
    for i, h in enumerate(HORIZONS):
        frames.append(
            pd.DataFrame(
                {
                    "FSA": test_df["FSA"],
                    "origin_timestamp": test_df["origin_timestamp"],
                    "forecast_timestamp": test_df["origin_timestamp"] + pd.Timedelta(hours=h),
                    "horizon": h,
                    "actual": test_df[f"real_h{h}"],
                    "prediction": real_pred[:, i],
                    "model_name": MODEL_NAME,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
