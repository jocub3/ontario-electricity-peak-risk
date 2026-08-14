"""Create cyclical temporal representations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import make_feature_log


def add_temporal_features(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add cyclical encodings for configured temporal variables."""
    output = frame.copy()
    rows = []

    temporal_config = config["feature_engineering"]["temporal"]

    if not temporal_config.get("create_cyclical_features", True):
        return output, pd.DataFrame(rows)

    for column, specification in temporal_config["cyclical_features"].items():
        if column not in output.columns:
            continue

        period = float(specification["period"])
        numeric = pd.to_numeric(output[column], errors="coerce")

        sin_name = f"{column}_sin"
        cos_name = f"{column}_cos"

        output[sin_name] = np.sin(2 * np.pi * numeric / period)
        output[cos_name] = np.cos(2 * np.pi * numeric / period)

        for feature_name, function_name in [
            (sin_name, "sine"),
            (cos_name, "cosine"),
        ]:
            rows.append(
                make_feature_log(
                    feature_name,
                    "temporal_cyclical",
                    [column],
                    f"{function_name.title()} cyclical encoding of {column} with period {period:g}.",
                    True,
                    True,
                    "static_dataset",
                )
            )

    return output, pd.DataFrame(rows)
