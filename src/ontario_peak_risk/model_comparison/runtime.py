"""Runtime and complexity evidence.

Runtime is included only when it can be derived from saved outputs or explicit
manual overrides. Missing evidence stays missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import ModelReports


_TIME_COLUMNS = [
    "task_seconds",
    "fit_seconds",
    "training_seconds",
    "runtime_seconds",
    "elapsed_seconds",
    "total_seconds",
]


def _seconds_from_frame(frame: pd.DataFrame | None) -> tuple[float, str]:
    if frame is None or frame.empty:
        return np.nan, "unavailable"

    for column in _TIME_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).dropna()

            if not values.empty:
                return float(values.sum()), f"sum({column})"

    return np.nan, "unavailable"


def runtime_table(
    reports: dict[str, ModelReports],
    manual_overrides: dict | None = None,
) -> pd.DataFrame:
    manual_overrides = manual_overrides or {}
    rows = []

    for key, model in reports.items():
        seconds = np.nan
        source = "unavailable"

        if key in manual_overrides:
            value = manual_overrides[key]
            if isinstance(value, dict):
                seconds = float(value["seconds"])
                source = value.get("source", "manual_override")
            else:
                seconds = float(value)
                source = "manual_override"
        else:
            for frame_name, frame in [
                ("runtime_diagnostics", model.runtime_diagnostics),
                ("tuning_details", model.tuning_details),
            ]:
                candidate, candidate_source = _seconds_from_frame(
                    frame
                )
                if np.isfinite(candidate):
                    seconds = candidate
                    source = f"{frame_name}:{candidate_source}"
                    break

        rows.append(
            {
                "model": model.display_name,
                "family": model.family,
                "runtime_seconds_evidence": seconds,
                "runtime_minutes_evidence": (
                    seconds / 60 if np.isfinite(seconds) else np.nan
                ),
                "runtime_source": source,
            }
        )

    return pd.DataFrame(rows)
