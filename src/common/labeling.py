"""Peak-risk labeling: percentile_97.5(FSA, season) threshold, computed only on a fold's train data.

Locked by the team's methodology - the threshold must come only from that fold's training rows,
never from validation/test or the full history, and gets recomputed separately for each fold.
"""

import pandas as pd

PEAK_PERCENTILE = 0.975


def compute_peak_thresholds(
    train_df: pd.DataFrame, value_col: str = "TOTAL_CONSUMPTION"
) -> pd.DataFrame:
    """One threshold per (FSA, season), computed only from train_df."""
    return (
        train_df.groupby(["FSA", "season"], observed=True)[value_col]
        .quantile(PEAK_PERCENTILE)
        .rename("peak_threshold")
        .reset_index()
    )


def label_peaks(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    value_col: str = "TOTAL_CONSUMPTION",
    season_col: str = "season",
) -> pd.DataFrame:
    """Add actual_peak (1/0): 1 where value_col exceeds its (FSA, season) threshold.

    season_col lets a caller point at a season column that isn't literally named "season" -
    the classifier's training table keeps two: the origin's season (a model feature) and the
    forecast_timestamp's season (what the threshold must match, since that's the hour being
    judged as a peak or not). They're usually the same value but can differ across a
    season boundary within the horizon window.
    """
    thresholds = thresholds.rename(columns={"season": season_col})
    labeled = df.merge(thresholds, on=["FSA", season_col], how="left")
    labeled["actual_peak"] = (labeled[value_col] > labeled["peak_threshold"]).astype(int)
    return labeled
