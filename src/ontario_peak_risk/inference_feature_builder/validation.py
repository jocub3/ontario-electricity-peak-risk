"""Operational input and data-availability validation."""

from __future__ import annotations
import numpy as np
import pandas as pd


def demand_history_requirements(
    observed_history: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    fsas: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    ts, fsa_col, demand = cols["timestamp"], cols["fsa"], cols["demand"]

    lookback = int(config["inference"]["required_history_hours_before_origin"])
    required_index = pd.date_range(
        forecast_origin - pd.Timedelta(hours=lookback),
        forecast_origin,
        freq="h",
    )

    rows, missing_rows = [], []

    for fsa in fsas:
        group = observed_history.loc[
            observed_history[fsa_col].eq(fsa),
            [ts, demand],
        ].dropna(subset=[ts])

        actual = set(group.loc[group[demand].notna(), ts].tolist())
        missing = [stamp for stamp in required_index if stamp not in actual]
        # Recency must be evaluated as-of the forecast origin.
        # Future observations may exist in storage (e.g., for a post-holdout
        # simulation), but they were not available at the simulated origin.
        available_as_of_origin = group.loc[
            group[demand].notna() & group[ts].le(forecast_origin)
        ]
        latest = (
            available_as_of_origin[ts].max()
            if not available_as_of_origin.empty else pd.NaT
        )

        rows.append({
            "fsa": fsa,
            "required_start": required_index.min(),
            "required_end": required_index.max(),
            "required_hours": len(required_index),
            "available_hours": len(required_index) - len(missing),
            "missing_hours": len(missing),
            "latest_available_demand": latest,
            "gap_from_latest_to_origin_hours": (
                (forecast_origin - latest) / pd.Timedelta(hours=1)
                if pd.notna(latest) else np.nan
            ),
            "status": "PASS" if not missing else "FAIL",
        })

        for stamp in missing:
            missing_rows.append({
                "fsa": fsa,
                "missing_timestamp": stamp,
            })

    return pd.DataFrame(rows), pd.DataFrame(missing_rows)


def validate_weather_grid(
    weather_forecast: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    fsas: list[str],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    ts, fsa_col = cols["timestamp"], cols["fsa"]
    weather_columns = [
        cols["temperature"], cols["humidity"], cols["pressure"]
    ]

    required = pd.date_range(
        forecast_origin,
        forecast_origin + pd.Timedelta(
            hours=int(config["inference"]["horizons"])
        ),
        freq="h",
    )

    rows, missing_rows = [], []

    for fsa in fsas:
        group = weather_forecast.loc[
            weather_forecast.get(fsa_col, pd.Series(dtype=str)).eq(fsa)
        ].copy() if not weather_forecast.empty else pd.DataFrame()

        fsa_missing = 0
        for stamp in required:
            match = (
                group.loc[group[ts].eq(stamp)]
                if (not group.empty and ts in group.columns)
                else pd.DataFrame()
            )

            complete = (
                len(match) == 1
                and all(
                    column in match.columns
                    and pd.notna(match.iloc[0][column])
                    for column in weather_columns
                )
            )

            if not complete:
                fsa_missing += 1
                missing_rows.append({
                    "fsa": fsa,
                    "timestamp": stamp,
                    "reason": "missing row or required weather value",
                })

        rows.append({
            "fsa": fsa,
            "required_rows_h0_h24": len(required),
            "available_complete_rows": len(required) - fsa_missing,
            "missing_rows": fsa_missing,
            "status": "PASS" if fsa_missing == 0 else "FAIL",
        })

    return pd.DataFrame(rows), pd.DataFrame(missing_rows)


def domain_reference(
    project_history: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    rows = []

    for variable in config["domain_checks"]["variables"]:
        if variable not in project_history.columns:
            continue

        values = pd.to_numeric(
            project_history[variable], errors="coerce"
        ).dropna()
        if values.empty:
            continue

        rows.append({
            "variable": variable,
            "historical_min": values.min(),
            "historical_p01": values.quantile(0.01),
            "historical_p99": values.quantile(0.99),
            "historical_max": values.max(),
        })

    return pd.DataFrame(rows)


def domain_check(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    rows = []

    for _, ref in reference.iterrows():
        variable = ref["variable"]
        if variable not in frame.columns:
            continue

        values = pd.to_numeric(frame[variable], errors="coerce")

        for idx, value in values.items():
            if pd.isna(value):
                continue

            if value < ref["historical_min"] or value > ref["historical_max"]:
                level = "OUT_OF_RANGE"
            elif value < ref["historical_p01"] or value > ref["historical_p99"]:
                level = "CAUTION"
            else:
                level = "NORMAL"

            if level != "NORMAL":
                rows.append({
                    "context": context,
                    "row_index": idx,
                    "variable": variable,
                    "value": value,
                    "historical_min": ref["historical_min"],
                    "historical_p01": ref["historical_p01"],
                    "historical_p99": ref["historical_p99"],
                    "historical_max": ref["historical_max"],
                    "domain_status": level,
                })

    return pd.DataFrame(rows)


def validate_prediction_output(
    prediction: pd.DataFrame,
    fsas: list[str],
    horizons: int,
    threshold: float,
) -> pd.DataFrame:
    """Validate the final integrated operational prediction table."""
    required_columns = [
        "fsa",
        "forecast_origin",
        "target_timestamp",
        "horizon",
        "forecast_consumption_kwh",
        "peak_risk_score",
        "peak_alert",
    ]

    rows = []

    missing_columns = [
        col for col in required_columns if col not in prediction.columns
    ]
    rows.append({
        "check": "required_columns_present",
        "status": "PASS" if not missing_columns else "FAIL",
        "detail": (
            "all required columns available"
            if not missing_columns
            else f"missing: {missing_columns}"
        ),
    })

    if missing_columns:
        return pd.DataFrame(rows)

    expected_rows = len(fsas) * int(horizons)
    rows.append({
        "check": "expected_row_count",
        "status": "PASS" if len(prediction) == expected_rows else "FAIL",
        "detail": f"actual={len(prediction)}, expected={expected_rows}",
    })

    key = ["fsa", "forecast_origin", "target_timestamp", "horizon"]
    duplicate_count = int(prediction.duplicated(key).sum())
    rows.append({
        "check": "unique_prediction_key",
        "status": "PASS" if duplicate_count == 0 else "FAIL",
        "detail": f"duplicate_rows={duplicate_count}",
    })

    actual_fsas = set(prediction["fsa"].dropna().astype(str))
    expected_fsas = set(map(str, fsas))
    rows.append({
        "check": "fsa_coverage",
        "status": "PASS" if actual_fsas == expected_fsas else "FAIL",
        "detail": (
            f"actual={sorted(actual_fsas)}, "
            f"expected={sorted(expected_fsas)}"
        ),
    })

    expected_horizons = set(range(1, int(horizons) + 1))
    horizon_ok = True
    horizon_detail = []
    for fsa, group in prediction.groupby("fsa", observed=True):
        actual = set(pd.to_numeric(group["horizon"], errors="coerce").dropna())
        if actual != expected_horizons:
            horizon_ok = False
            horizon_detail.append(str(fsa))
    rows.append({
        "check": "complete_horizon_coverage_by_fsa",
        "status": "PASS" if horizon_ok else "FAIL",
        "detail": (
            "h1..h24 complete for every FSA"
            if horizon_ok
            else f"incomplete FSAs: {horizon_detail}"
        ),
    })

    missing_values = int(
        prediction[
            ["forecast_consumption_kwh", "peak_risk_score", "peak_alert"]
        ].isna().sum().sum()
    )
    rows.append({
        "check": "prediction_values_not_missing",
        "status": "PASS" if missing_values == 0 else "FAIL",
        "detail": f"missing_values={missing_values}",
    })

    demand = pd.to_numeric(
        prediction["forecast_consumption_kwh"], errors="coerce"
    )
    invalid_demand = int((demand <= 0).sum() + demand.isna().sum())
    rows.append({
        "check": "forecast_consumption_positive",
        "status": "PASS" if invalid_demand == 0 else "FAIL",
        "detail": f"invalid_rows={invalid_demand}",
    })

    score = pd.to_numeric(prediction["peak_risk_score"], errors="coerce")
    invalid_score = int((~score.between(0, 1, inclusive="both")).sum())
    rows.append({
        "check": "peak_risk_score_in_0_1",
        "status": "PASS" if invalid_score == 0 else "FAIL",
        "detail": f"invalid_rows={invalid_score}",
    })

    expected_alert = score.ge(float(threshold))
    actual_alert = prediction["peak_alert"].astype(bool)
    inconsistent = int((expected_alert != actual_alert).sum())
    rows.append({
        "check": "peak_alert_threshold_consistency",
        "status": "PASS" if inconsistent == 0 else "FAIL",
        "detail": (
            f"inconsistent_rows={inconsistent}, "
            f"threshold={float(threshold):.6g}"
        ),
    })

    return pd.DataFrame(rows)
