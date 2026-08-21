"""Peak-risk threshold and labeling utilities.

Peak = 1 if consumption exceeds the 97.5th percentile for that (FSA, season),
computed only on a fold's training split so no future information leaks into
the classifier's ground truth.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_PERCENTILE = 97.5
GROUP_COLUMNS = ("fsa", "season")


def compute_peak_thresholds(
    train: pd.DataFrame,
    percentile: float = DEFAULT_PERCENTILE,
    consumption_column: str = "electricity_consumption",
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Compute the peak-consumption threshold per group, from training data only."""
    thresholds = (
        train.groupby(list(group_columns))[consumption_column]
        .quantile(percentile / 100)
        .rename("peak_threshold")
        .reset_index()
    )
    return thresholds


def label_peaks(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    consumption_column: str = "electricity_consumption",
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Add an ``actual_peak`` column to ``df`` using precomputed thresholds.

    ``thresholds`` should come from :func:`compute_peak_thresholds` on the
    same fold's training split, so train and test get labeled consistently.
    """
    labeled = df.merge(thresholds, on=list(group_columns), how="left", validate="many_to_one")
    labeled["actual_peak"] = (labeled[consumption_column] > labeled["peak_threshold"]).astype(int)
    return labeled
