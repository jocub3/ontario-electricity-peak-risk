"""Validation contracts for lightweight application artifacts."""

from __future__ import annotations

import pandas as pd


PREDICTION_COLUMNS = [
    "fsa",
    "forecast_origin",
    "target_timestamp",
    "horizon",
    "forecast_consumption_kwh",
    "peak_risk_score",
    "peak_alert",
]

SCENARIO_COLUMNS = [
    "fsa",
    "forecast_origin",
    "target_timestamp",
    "horizon",
    "scenario",
    "temperature_delta_c",
    "forecast_consumption_kwh",
    "baseline_forecast_kwh",
    "forecast_delta_kwh",
    "peak_risk_score",
    "baseline_peak_risk_score",
    "peak_risk_delta",
    "peak_alert",
    "baseline_peak_alert",
    "alert_changed",
]


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def validate_predictions(
    frame: pd.DataFrame,
    fsas: list[str],
    horizons: int,
    threshold: float,
) -> pd.DataFrame:
    """Validate the public app's core prediction contract."""
    _require_columns(frame, PREDICTION_COLUMNS, "Prediction dataset")

    data = frame.copy()
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"])
    data["target_timestamp"] = pd.to_datetime(data["target_timestamp"])

    rows = []
    for origin, group in data.groupby("forecast_origin"):
        expected = len(fsas) * int(horizons)
        unique_keys = group[["fsa", "horizon"]].drop_duplicates().shape[0]

        checks = {
            "row_count": len(group) == expected,
            "unique_fsa_horizon": unique_keys == expected,
            "all_fsas": set(group["fsa"]) == set(fsas),
            "all_horizons": set(group["horizon"]) == set(range(1, horizons + 1)),
            "forecast_non_null": group["forecast_consumption_kwh"].notna().all(),
            "forecast_positive": group["forecast_consumption_kwh"].gt(0).all(),
            "risk_non_null": group["peak_risk_score"].notna().all(),
            "risk_in_0_1": group["peak_risk_score"].between(0, 1).all(),
            "alert_consistent": (
                group["peak_alert"].astype(bool)
                == group["peak_risk_score"].ge(float(threshold))
            ).all(),
        }

        for check, passed in checks.items():
            rows.append(
                {
                    "forecast_origin": origin,
                    "check": check,
                    "status": "PASS" if passed else "FAIL",
                }
            )

    return pd.DataFrame(rows)


def validate_scenarios(
    frame: pd.DataFrame,
    fsas: list[str],
    horizons: int,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    _require_columns(frame, SCENARIO_COLUMNS, "Scenario dataset")
    data = frame.copy()
    data["forecast_origin"] = pd.to_datetime(data["forecast_origin"])

    rows = []
    for origin, group in data.groupby("forecast_origin"):
        found_scenarios = sorted(group["scenario"].astype(str).unique())
        expected_scenarios = sorted(scenarios or found_scenarios)
        expected_rows = len(fsas) * int(horizons) * len(expected_scenarios)

        checks = {
            "row_count": len(group) == expected_rows,
            "scenario_set": found_scenarios == expected_scenarios,
            "forecast_delta_non_null": group["forecast_delta_kwh"].notna().all(),
            "risk_delta_non_null": group["peak_risk_delta"].notna().all(),
        }

        baseline = group[group["temperature_delta_c"].eq(0)]
        if not baseline.empty:
            checks["baseline_forecast_delta_zero"] = (
                baseline["forecast_delta_kwh"].abs().lt(1e-9).all()
            )
            checks["baseline_risk_delta_zero"] = (
                baseline["peak_risk_delta"].abs().lt(1e-12).all()
            )

        for check, passed in checks.items():
            rows.append(
                {
                    "forecast_origin": origin,
                    "check": check,
                    "status": "PASS" if passed else "FAIL",
                }
            )

    return pd.DataFrame(rows)
