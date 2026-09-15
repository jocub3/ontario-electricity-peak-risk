"""Historical electricity-demand lag features."""

from __future__ import annotations

import pandas as pd

from .common import make_feature_log


def add_target_lags(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add target lags independently within each FSA.

    Lag t-k at timestamp t contains only an observation available k hours earlier.
    """
    output = frame.copy()
    target = config["feature_engineering"]["target_column"]
    lags = config["feature_engineering"]["lags"]["target_lags_hours"]

    rows = []

    grouped_target = output.groupby("fsa", observed=True)[target]

    for lag in lags:
        name = f"{target}_lag_{lag}h"
        output[name] = grouped_target.shift(lag)

        rows.append(
            make_feature_log(
                name,
                "target_lag",
                [target],
                f"Electricity consumption observed {lag} hour(s) before the current timestamp within the same FSA.",
                True,
                True,
                "historical_only",
                "low_if_computed_after_temporal_ordering",
            )
        )

    return output, pd.DataFrame(rows)
