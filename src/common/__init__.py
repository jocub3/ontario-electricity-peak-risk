"""Shared, reusable utilities for the Ontario Electricity Demand Forecasting
and Peak-Risk project.

Every model branch (baseline, SARIMAX, XGBoost, LightGBM) is expected to
import from here instead of re-implementing data loading, cleaning, or
evaluation logic, so results stay comparable across the team.
"""

from .data_loading import (
    build_all_fsa_dataset,
    build_fsa_hourly_dataset,
    load_calendar,
    load_consumption,
    load_weather,
    save_interim_dataset,
)

__all__ = [
    "build_all_fsa_dataset",
    "build_fsa_hourly_dataset",
    "load_calendar",
    "load_consumption",
    "load_weather",
    "save_interim_dataset",
]
