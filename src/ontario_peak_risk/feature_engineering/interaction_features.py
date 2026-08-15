"""Small set of EDA-justified interaction features."""

from __future__ import annotations

import pandas as pd

from .common import make_feature_log


def add_interaction_features(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create only explicitly configured interactions."""
    output = frame.copy()
    settings = config["feature_engineering"]["interactions"]
    rows = []

    temperature = config["feature_engineering"]["weather"]["temperature_column"]

    if (
        settings.get("create_temperature_hour")
        and temperature in output.columns
        and "hour" in output.columns
    ):
        output["temperature_x_hour"] = output[temperature] * output["hour"]
        rows.append(
            make_feature_log(
                "temperature_x_hour",
                "interaction",
                [temperature, "hour"],
                "Interaction between current temperature and hour of day.",
                True,
                True,
                "current_exogenous",
            )
        )

    if (
        settings.get("create_temperature_weekend")
        and temperature in output.columns
        and "is_weekend" in output.columns
    ):
        output["temperature_x_weekend"] = (
            output[temperature] * output["is_weekend"]
        )
        rows.append(
            make_feature_log(
                "temperature_x_weekend",
                "interaction",
                [temperature, "is_weekend"],
                "Interaction between current temperature and weekend status.",
                True,
                True,
                "current_exogenous",
            )
        )

    return output, pd.DataFrame(rows)
