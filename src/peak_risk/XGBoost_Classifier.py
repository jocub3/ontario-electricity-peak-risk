"""XGBoost row-wise classifier for peak-demand risk - Model 3 (pooled across FSAs)."""

import pandas as pd
import xgboost as xgb

from common.feature_selection import SELECTED_FEATURES
from common.labeling import label_peaks
from common.targets import build_long_targets
from common.transforms import set_categorical_dtypes

MODEL_NAME = "xgboost_classifier"

CLASSIFIER_FEATURES = SELECTED_FEATURES + ["horizon"]


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """The joined hourly dataset is reshaped into one row per (FSA, origin_timestamp, horizon).

    Each row asks: from origin_timestamp, using only what is known then (weather/calendar
    persisted from origin, same assumption as the regressor), is forecast_timestamp
    (horizon hours later) a peak? actual_peak itself is not added here, the peak threshold
    can only be computed from a fold's own training data, so labeling happens per fold in
    train_and_predict(), not in this fold-independent table.

    Two season columns are kept on purpose: "season" is the origin's season (a model
    feature, part of SELECTED_FEATURES), "forecast_season" is the season of the hour being
    judged as a peak or not, they matter for different things and are usually but not
    always the same value.
    """
    df = set_categorical_dtypes(df)
    long_targets = build_long_targets(df, value_col="TOTAL_CONSUMPTION")

    season_at_forecast = df[["FSA", "timestamp_local", "season"]].rename(
        columns={"timestamp_local": "forecast_timestamp", "season": "forecast_season"}
    )
    long_targets = long_targets.merge(
        season_at_forecast, on=["FSA", "forecast_timestamp"], how="left"
    )

    feature_cols = [c for c in SELECTED_FEATURES if c != "FSA"]
    features = df[["FSA", "timestamp_local"] + feature_cols].rename(
        columns={"timestamp_local": "origin_timestamp"}
    )

    table = long_targets.merge(features, on=["FSA", "origin_timestamp"])
    table["year"] = table["origin_timestamp"].dt.year
    return table


def make_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    """First-pass hyperparameters, not tuned - same tree settings as the regressor.
    `scale_pos_weight` (n_negatives/n_positives, computed per fold by the caller) rebalances
    the loss toward the minority peak class; `eval_metric="aucpr"` matches PR-AUC, the team's
    primary metric for this model."""
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )


def train_and_predict(
    train_table: pd.DataFrame, test_table: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    """Both tables are labeled with the fold's thresholds, the model is fit on train and predicts on test.

    thresholds must come from compute_peak_thresholds() run on that fold's own raw training
    hours (not on train_table) - the pivoted table repeats most real hours once per horizon
    that reaches them, so the fold's raw hourly data is the correct, unambiguous source for
    the percentile.
    """
    train_labeled = label_peaks(
        train_table, thresholds, value_col="actual", season_col="forecast_season"
    )
    test_labeled = label_peaks(
        test_table, thresholds, value_col="actual", season_col="forecast_season"
    )

    # the merge above can downcast FSA/season from category to object if thresholds' own
    # FSA/season columns are not category (e.g. computed from a raw, not-yet-cast dataframe)
    train_labeled = set_categorical_dtypes(train_labeled)
    test_labeled = set_categorical_dtypes(test_labeled)

    n_pos = (train_labeled["actual_peak"] == 1).sum()
    n_neg = (train_labeled["actual_peak"] == 0).sum()
    model = make_model(scale_pos_weight=n_neg / n_pos)
    model.fit(train_labeled[CLASSIFIER_FEATURES], train_labeled["actual_peak"])

    peak_probability = model.predict_proba(test_labeled[CLASSIFIER_FEATURES])[:, 1]
    predicted_peak = (peak_probability >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "FSA": test_labeled["FSA"],
            "forecast_timestamp": test_labeled["forecast_timestamp"],
            "actual_peak": test_labeled["actual_peak"],
            "peak_probability": peak_probability,
            "predicted_peak": predicted_peak,
            "model_name": MODEL_NAME,
            # extra column beyond the required schema: a given forecast_timestamp gets one row
            # per horizon that reaches it (same origin-flexible design as the regressor), so
            # horizon is kept to tell those rows apart and to check accuracy by horizon
            "horizon": test_labeled["horizon"],
        }
    )
