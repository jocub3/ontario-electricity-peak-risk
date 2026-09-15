"""Direct multi-horizon SARIMAX candidate model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
    resolve_project_path,
    save_csv,
)
from src.ontario_peak_risk.modeling.metrics import forecasting_metrics
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_forecast_origin_fold,
)

from .data import build_sarimax_exog, sarimax_exog_columns


def _load_branch_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config, path.parent.parent


def _prepare_xy(
    targets: pd.DataFrame,
    observed_frame: pd.DataFrame,
    horizon: int,
    target_prefix: str,
    fsa: str,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    subset = (
        targets.loc[
            targets["fsa"].eq(fsa),
            ["fsa", "forecast_origin", f"{target_prefix}{horizon:02d}"],
        ]
        .sort_values("forecast_origin")
        .reset_index(drop=True)
    )

    exog_frame = build_sarimax_exog(
        subset[["fsa", "forecast_origin"]],
        observed_frame,
        horizon=horizon,
    )

    exog_columns = sarimax_exog_columns(exog_frame)
    exog = exog_frame[exog_columns].copy()

    # Median imputation is fit from the current dataset passed to this helper.
    # For validation, callers replace missing values using training medians.
    y = subset[f"{target_prefix}{horizon:02d}"].astype(float)

    keys = subset[["fsa", "forecast_origin"]].copy()

    return y, exog, keys


def _fit_predict(
    train_y: pd.Series,
    train_exog: pd.DataFrame,
    validation_exog: pd.DataFrame,
    specification: dict,
    fit_cfg: dict,
) -> np.ndarray:
    train_medians = train_exog.median(numeric_only=True)
    train_exog = train_exog.fillna(train_medians).fillna(0.0)
    validation_exog = validation_exog.fillna(train_medians).fillna(0.0)

    model = SARIMAX(
        endog=train_y,
        exog=train_exog,
        order=tuple(specification["order"]),
        seasonal_order=tuple(specification["seasonal_order"]),
        trend=specification.get("trend", "c"),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False,
        maxiter=int(fit_cfg["maxiter"]),
        method=str(fit_cfg["method"]),
    )

    forecast = fitted.get_forecast(
        steps=len(validation_exog),
        exog=validation_exog,
    )

    return np.asarray(forecast.predicted_mean)


def _tune(
    feature_dataset: pd.DataFrame,
    forecast_targets: pd.DataFrame,
    modeling_config: dict,
    branch_config: dict,
) -> pd.DataFrame:
    cfg = branch_config["model"]
    folds = build_validation_folds(modeling_config)
    rows = []

    for spec_id, specification in enumerate(
        cfg["tuning"]["specifications"],
        start=1,
    ):
        for fold in folds:
            train_targets, validation_targets = slice_forecast_origin_fold(
                forecast_targets,
                fold,
                time_column="forecast_origin",
                max_horizon_hours=int(cfg["horizons"]),
            )

            for fsa in cfg["tuning"]["representative_fsas"]:
                for horizon in cfg["tuning"]["representative_horizons"]:
                    train_y, train_exog, _ = _prepare_xy(
                        train_targets,
                        feature_dataset,
                        int(horizon),
                        cfg["target_prefix"],
                        fsa,
                    )
                    validation_y, validation_exog, _ = _prepare_xy(
                        validation_targets,
                        feature_dataset,
                        int(horizon),
                        cfg["target_prefix"],
                        fsa,
                    )

                    prediction = _fit_predict(
                        train_y,
                        train_exog,
                        validation_exog,
                        specification,
                        cfg["fit"],
                    )

                    rows.append(
                        {
                            "specification_id": spec_id,
                            "fold": fold.name,
                            "fsa": fsa,
                            "horizon": int(horizon),
                            "order": str(tuple(specification["order"])),
                            "seasonal_order": str(
                                tuple(specification["seasonal_order"])
                            ),
                            "trend": specification.get("trend", "c"),
                            **forecasting_metrics(validation_y, prediction),
                        }
                    )

    return pd.DataFrame(rows)


def _select_specification(
    tuning_results: pd.DataFrame,
    specifications: list[dict],
) -> tuple[dict, pd.DataFrame]:
    summary = (
        tuning_results
        .groupby("specification_id", observed=True)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
        )
        .reset_index()
        .sort_values(["mean_mae", "mean_rmse"])
        .reset_index(drop=True)
    )

    best_id = int(summary.iloc[0]["specification_id"])
    return specifications[best_id - 1], summary


def run_sarimax(
    config_path: str | Path = "configs/sarimax.yaml",
) -> dict[str, object]:
    branch_config, project_root = _load_branch_config(config_path)

    foundation_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(foundation_path)

    feature_dataset = load_feature_dataset(modeling_config, project_root)
    forecast_targets = load_forecast_targets(modeling_config, project_root)

    cfg = branch_config["model"]

    if bool(cfg.get("evaluate_final_holdout", False)):
        raise ValueError("SARIMAX candidate must not evaluate the 2025 holdout.")

    paths = {
        key: resolve_project_path(project_root, value)
        for key, value in branch_config["paths"].items()
        if key.endswith("_dir")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    tuning_results = _tune(
        feature_dataset,
        forecast_targets,
        modeling_config,
        branch_config,
    )

    best_spec, tuning_summary = _select_specification(
        tuning_results,
        cfg["tuning"]["specifications"],
    )

    save_csv(
        tuning_results,
        paths["reports_dir"] / "SARIMAX_05_tuning_details.csv",
    )
    save_csv(
        tuning_summary,
        paths["reports_dir"] / "SARIMAX_05_tuning_summary.csv",
    )

    (
        paths["outputs_dir"] / "best_specification.json"
    ).write_text(
        json.dumps(best_spec, indent=2),
        encoding="utf-8",
    )

    folds = build_validation_folds(modeling_config)
    prediction_parts = []

    for fold in folds:
        train_targets, validation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=int(cfg["horizons"]),
        )

        for fsa in sorted(validation_targets["fsa"].dropna().unique()):
            for horizon in range(1, int(cfg["horizons"]) + 1):
                train_y, train_exog, _ = _prepare_xy(
                    train_targets,
                    feature_dataset,
                    horizon,
                    cfg["target_prefix"],
                    fsa,
                )
                validation_y, validation_exog, validation_keys = _prepare_xy(
                    validation_targets,
                    feature_dataset,
                    horizon,
                    cfg["target_prefix"],
                    fsa,
                )

                prediction = _fit_predict(
                    train_y,
                    train_exog,
                    validation_exog,
                    best_spec,
                    cfg["fit"],
                )

                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "fold": fold.name,
                            "fsa": fsa,
                            "forecast_origin": validation_keys["forecast_origin"],
                            "horizon": horizon,
                            "actual": validation_y.to_numpy(),
                            "predicted": prediction,
                        }
                    )
                )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["residual"] = predictions["actual"] - predictions["predicted"]

    fold_rows = []
    fsa_rows = []
    horizon_rows = []

    for fold, group in predictions.groupby("fold", observed=True):
        fold_rows.append(
            {"fold": fold, **forecasting_metrics(group["actual"], group["predicted"])}
        )

    for (fold, fsa), group in predictions.groupby(
        ["fold", "fsa"], observed=True
    ):
        fsa_rows.append(
            {
                "fold": fold,
                "fsa": fsa,
                **forecasting_metrics(group["actual"], group["predicted"]),
            }
        )

    for (fold, horizon), group in predictions.groupby(
        ["fold", "horizon"], observed=True
    ):
        horizon_rows.append(
            {
                "fold": fold,
                "horizon": horizon,
                **forecasting_metrics(group["actual"], group["predicted"]),
            }
        )

    global_metrics = pd.DataFrame(
        [
            {
                "model": cfg["name"],
                **forecasting_metrics(
                    predictions["actual"],
                    predictions["predicted"],
                ),
            }
        ]
    )
    fold_metrics = pd.DataFrame(fold_rows)
    fsa_metrics = pd.DataFrame(fsa_rows)
    horizon_metrics = pd.DataFrame(horizon_rows)

    save_csv(fold_metrics, paths["reports_dir"] / "SARIMAX_06_fold_metrics.csv")
    save_csv(fsa_metrics, paths["reports_dir"] / "SARIMAX_06_fsa_metrics.csv")
    save_csv(
        horizon_metrics,
        paths["reports_dir"] / "SARIMAX_06_horizon_metrics.csv",
    )
    save_csv(
        global_metrics,
        paths["reports_dir"] / "SARIMAX_06_global_metrics.csv",
    )

    predictions.to_parquet(
        paths["outputs_dir"] / "sarimax_validation_predictions.parquet",
        index=False,
    )

    print("SARIMAX completed successfully.")
    print("Best specification:", best_spec)
    print("Final 2025 holdout evaluated: NO")

    return {
        "tuning_details": tuning_results,
        "tuning_summary": tuning_summary,
        "best_specification": best_spec,
        "global_metrics": global_metrics,
        "fold_metrics": fold_metrics,
        "fsa_metrics": fsa_metrics,
        "horizon_metrics": horizon_metrics,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARIMAX candidate model.")
    parser.add_argument("--config", default="configs/sarimax.yaml")
    args = parser.parse_args()
    run_sarimax(args.config)


if __name__ == "__main__":
    main()
