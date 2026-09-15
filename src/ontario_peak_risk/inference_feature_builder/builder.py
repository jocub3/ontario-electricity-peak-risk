"""Build exact frozen model features from operational inputs."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_FEATURES = {
    "target_hour",
    "target_weekday",
    "target_month",
    "target_is_weekend",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_month_sin",
    "target_month_cos",
    "target_lag_24h",
    "target_lag_48h",
    "target_lag_168h",
    "origin_rolling_mean_24h",
    "origin_rolling_std_24h",
    "origin_rolling_mean_168h",
    "origin__Temp (°C)",
    "origin__Rel Hum (%)",
    "origin__Stn Press (kPa)",
    "origin__reported_premise_count",
    "fsa",
    "target_season",
}


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def build_forecast_grid(
    forecast_origin: str | pd.Timestamp,
    fsas: list[str],
    horizons: int = 24,
) -> pd.DataFrame:
    origin = pd.Timestamp(forecast_origin)
    rows = []

    for fsa in fsas:
        for horizon in range(1, horizons + 1):
            rows.append({
                "fsa": fsa,
                "forecast_origin": origin,
                "horizon": horizon,
                "target_timestamp": origin + pd.Timedelta(hours=horizon),
            })

    return pd.DataFrame(rows)


def _demand_lookup(
    observed_history: pd.DataFrame,
    config: dict,
) -> pd.Series:
    cols = config["columns"]
    return (
        observed_history[
            [cols["fsa"], cols["timestamp"], cols["demand"]]
        ]
        .dropna(subset=[cols["fsa"], cols["timestamp"]])
        .drop_duplicates([cols["fsa"], cols["timestamp"]], keep="last")
        .set_index([cols["fsa"], cols["timestamp"]])[cols["demand"]]
    )


def _origin_values(
    observed_history: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    cols = config["columns"]
    fsa, ts = cols["fsa"], cols["timestamp"]

    source_cols = [
        fsa, ts, cols["temperature"], cols["humidity"],
        cols["pressure"], cols["premise_count"],
    ]
    observed_cols = [c for c in source_cols if c in observed_history.columns]

    observed = observed_history.loc[
        observed_history[ts].eq(forecast_origin),
        observed_cols,
    ].drop_duplicates([fsa, ts], keep="last").copy()

    forecast_cols = [
        c for c in [
            fsa, ts, cols["temperature"], cols["humidity"], cols["pressure"]
        ]
        if c in weather_forecast.columns
    ]
    forecast_h0 = (
        weather_forecast.loc[
            weather_forecast[ts].eq(forecast_origin),
            forecast_cols,
        ]
        .drop_duplicates([fsa, ts], keep="last")
        .copy()
        if (not weather_forecast.empty and ts in weather_forecast.columns)
        else pd.DataFrame(columns=forecast_cols)
    )

    all_fsas = sorted(
        set(observed.get(fsa, pd.Series(dtype=str)).dropna().astype(str))
        | set(forecast_h0.get(fsa, pd.Series(dtype=str)).dropna().astype(str))
    )
    result = pd.DataFrame({fsa: all_fsas})

    observed_idx = (
        observed.set_index(fsa) if not observed.empty else pd.DataFrame()
    )
    forecast_idx = (
        forecast_h0.set_index(fsa) if not forecast_h0.empty else pd.DataFrame()
    )

    for column in [cols["temperature"], cols["humidity"], cols["pressure"]]:
        observed_series = (
            observed_idx[column]
            if (not observed_idx.empty and column in observed_idx.columns)
            else pd.Series(dtype=float)
        )
        forecast_series = (
            forecast_idx[column]
            if (not forecast_idx.empty and column in forecast_idx.columns)
            else pd.Series(dtype=float)
        )
        result[column] = result[fsa].map(
            observed_series.combine_first(forecast_series)
        )

    if (
        not observed_idx.empty
        and cols["premise_count"] in observed_idx.columns
    ):
        result[cols["premise_count"]] = result[fsa].map(
            observed_idx[cols["premise_count"]]
        )
    else:
        result[cols["premise_count"]] = pd.NA

    return result


def build_feature_matrix(
    observed_history: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    forecast_origin: str | pd.Timestamp,
    fsas: list[str],
    *,
    config: dict,
) -> pd.DataFrame:
    origin = pd.Timestamp(forecast_origin)
    cols = config["columns"]
    grid = build_forecast_grid(
        origin, fsas, int(config["inference"]["horizons"])
    )

    target = pd.to_datetime(grid["target_timestamp"])
    grid["target_hour"] = target.dt.hour
    grid["target_weekday"] = target.dt.weekday
    grid["target_month"] = target.dt.month
    grid["target_is_weekend"] = (
        grid["target_weekday"] >= 5
    ).astype("int8")

    grid["target_hour_sin"] = np.sin(
        2 * np.pi * grid["target_hour"] / 24
    )
    grid["target_hour_cos"] = np.cos(
        2 * np.pi * grid["target_hour"] / 24
    )
    grid["target_weekday_sin"] = np.sin(
        2 * np.pi * grid["target_weekday"] / 7
    )
    grid["target_weekday_cos"] = np.cos(
        2 * np.pi * grid["target_weekday"] / 7
    )
    grid["target_month_sin"] = np.sin(
        2 * np.pi * (grid["target_month"] - 1) / 12
    )
    grid["target_month_cos"] = np.cos(
        2 * np.pi * (grid["target_month"] - 1) / 12
    )
    grid["target_season"] = grid["target_month"].map(season_from_month)

    demand_lookup = _demand_lookup(observed_history, config)

    for lag in [24, 48, 168]:
        lookup_times = target - pd.to_timedelta(lag, unit="h")
        keys = pd.MultiIndex.from_arrays(
            [grid["fsa"].to_numpy(), lookup_times.to_numpy()],
            names=[cols["fsa"], cols["timestamp"]],
        )
        grid[f"target_lag_{lag}h"] = demand_lookup.reindex(keys).to_numpy()

    rolling_rows = []
    for fsa in fsas:
        # Build one FSA demand series through the forecast origin.
        # Keeping the origin observation is necessary because the frozen
        # Random Forest forecasting branch uses origin-inclusive rolling
        # summaries, while the XGBoost Peak-Risk branch uses strictly
        # pre-origin summaries.
        group_through_origin = (
            observed_history.loc[
                observed_history[cols["fsa"]].eq(fsa)
                & observed_history[cols["timestamp"]].le(origin),
                [cols["timestamp"], cols["demand"]],
            ]
            .dropna()
            .drop_duplicates(cols["timestamp"], keep="last")
            .sort_values(cols["timestamp"])
        )
        series_through_origin = (
            group_through_origin
            .set_index(cols["timestamp"])[cols["demand"]]
        )

        # XGBoost semantics: strictly before forecast_origin.
        last24 = series_through_origin.reindex(
            pd.date_range(
                origin - pd.Timedelta(hours=24),
                origin - pd.Timedelta(hours=1),
                freq="h",
            )
        )
        last168 = series_through_origin.reindex(
            pd.date_range(
                origin - pd.Timedelta(hours=168),
                origin - pd.Timedelta(hours=1),
                freq="h",
            )
        )

        # Random Forest semantics: include the observed value at
        # forecast_origin. These windows therefore contain exactly
        # 24 and 168 timestamps respectively when history is complete.
        rf_last24 = series_through_origin.reindex(
            pd.date_range(
                origin - pd.Timedelta(hours=23),
                origin,
                freq="h",
            )
        )
        rf_last168 = series_through_origin.reindex(
            pd.date_range(
                origin - pd.Timedelta(hours=167),
                origin,
                freq="h",
            )
        )

        rolling_rows.append({
            "fsa": fsa,

            # Frozen XGBoost semantics (exclusive of origin).
            "origin_rolling_mean_24h": last24.mean(),
            "origin_rolling_std_24h": last24.std(),
            "origin_rolling_mean_168h": last168.mean(),

            # Frozen Random Forest semantics (inclusive of origin).
            "_rf_origin_rolling_mean_24h": rf_last24.mean(),
            "_rf_origin_rolling_std_24h": rf_last24.std(),
            "_rf_origin_rolling_mean_168h": rf_last168.mean(),
        })

    grid = grid.merge(
        pd.DataFrame(rolling_rows),
        on="fsa",
        how="left",
        validate="many_to_one",
    )

    origin_values = _origin_values(
        observed_history, weather_forecast, origin, config
    )
    origin_values = origin_values.rename(columns={
        cols["temperature"]: "origin__Temp (°C)",
        cols["humidity"]: "origin__Rel Hum (%)",
        cols["pressure"]: "origin__Stn Press (kPa)",
        cols["premise_count"]: "origin__reported_premise_count",
    })

    grid = grid.merge(
        origin_values,
        on="fsa",
        how="left",
        validate="many_to_one",
    )

    return grid.sort_values(["fsa", "horizon"]).reset_index(drop=True)


def load_model_contracts(artifacts_dir: Path) -> dict[str, dict]:
    rf = json.loads(
        (artifacts_dir / "forecasting_random_forest" / "metadata.json")
        .read_text(encoding="utf-8")
    )
    xgb = json.loads(
        (artifacts_dir / "peak_risk_xgboost" / "metadata.json")
        .read_text(encoding="utf-8")
    )
    return {"rf": rf, "xgb": xgb}


def expected_features(metadata: dict) -> list[str]:
    return list(metadata["numeric_features"]) + list(
        metadata["categorical_features"]
    )


def feature_contract_report(
    generated: pd.DataFrame,
    metadata: dict,
    *,
    model_name: str,
) -> pd.DataFrame:
    expected = expected_features(metadata)
    rows = []

    for feature in expected:
        if feature not in generated.columns:
            status, detail = "FAIL", "feature not generated"
        elif feature not in SUPPORTED_FEATURES:
            status, detail = "FAIL", "feature unsupported by current IFB"
        elif generated[feature].isna().all():
            status, detail = "FAIL", "all values missing"
        elif generated[feature].isna().any():
            status = "WARN"
            detail = f"{int(generated[feature].isna().sum())} missing values"
        else:
            status, detail = "PASS", "available"

        rows.append({
            "model": model_name,
            "feature": feature,
            "status": status,
            "detail": detail,
        })

    return pd.DataFrame(rows)


def model_feature_frames(
    generated: pd.DataFrame,
    contracts: dict[str, dict],
) -> dict[str, dict[int, pd.DataFrame]]:
    output = {"rf": {}, "xgb": {}}

    for task in ["rf", "xgb"]:
        features = expected_features(contracts[task])

        for horizon in range(1, 25):
            frame = generated.loc[
                generated["horizon"].eq(horizon)
            ].copy()

            # The final RF and XGBoost artifacts use the same public rolling
            # feature names but were trained with different rolling-window
            # semantics. Historical replay proves:
            #
            # RF  : origin-inclusive window
            # XGB : strictly pre-origin window
            #
            # Replace only the RF values immediately before model inference.
            if task == "rf":
                rf_overrides = {
                    "origin_rolling_mean_24h":
                        "_rf_origin_rolling_mean_24h",
                    "origin_rolling_std_24h":
                        "_rf_origin_rolling_std_24h",
                    "origin_rolling_mean_168h":
                        "_rf_origin_rolling_mean_168h",
                }

                for public_name, internal_name in rf_overrides.items():
                    if public_name in features:
                        if internal_name not in frame.columns:
                            raise ValueError(
                                f"Missing RF-specific inference feature: "
                                f"{internal_name}"
                            )
                        frame[public_name] = frame[internal_name]

            key_columns = [
                "fsa",
                "forecast_origin",
                "target_timestamp",
                "horizon",
            ]
            selected_columns = list(
                dict.fromkeys(key_columns + features)
            )

            output[task][horizon] = frame[
                selected_columns
            ].reset_index(drop=True)

    return output
