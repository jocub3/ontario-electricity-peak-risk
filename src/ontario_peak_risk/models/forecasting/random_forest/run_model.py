"""Train, tune, evaluate, and document Random Forest Regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
    resolve_project_path,
    save_csv,
)
from src.ontario_peak_risk.modeling.evaluation import (
    evaluate_forecasting_by_group,
    evaluate_forecasting_by_horizon,
    evaluate_forecasting_global,
)
from src.ontario_peak_risk.modeling.metrics import forecasting_metrics
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_forecast_origin_fold,
)

from .features import (
    build_horizon_features,
    feature_dictionary,
    model_feature_columns,
)
from .preprocessing import build_random_forest_preprocessor


def _load_branch_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config, path.parent.parent


def _build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    parameters: dict,
    random_seed: int,
) -> Pipeline:
    preprocessor = build_random_forest_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    model = RandomForestRegressor(
        n_estimators=int(parameters["n_estimators"]),
        max_depth=parameters["max_depth"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=parameters["max_features"],
        random_state=random_seed,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _tune_parameters(
    feature_dataset: pd.DataFrame,
    forecast_targets: pd.DataFrame,
    modeling_config: dict,
    branch_config: dict,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Moderate tuning using the shared 2023 and 2024 validation folds.

    Only representative horizons are used during tuning to limit runtime.
    The selected configuration is later evaluated on all 24 horizons.
    """
    model_cfg = branch_config["model"]
    tuning_cfg = model_cfg["tuning"]
    horizons = int(model_cfg["horizons"])
    folds = build_validation_folds(modeling_config)

    rows = []

    for parameter_id, parameters in enumerate(
        tuning_cfg["parameter_grid"],
        start=1,
    ):
        for fold in folds:
            train_targets, validation_targets = slice_forecast_origin_fold(
                forecast_targets,
                fold,
                time_column="forecast_origin",
                max_horizon_hours=horizons,
            )

            train_targets = train_targets.reset_index(drop=True)
            validation_targets = validation_targets.reset_index(drop=True)

            for horizon in tuning_cfg["representative_horizons"]:
                target_column = (
                    f"{model_cfg['target_prefix']}{int(horizon):02d}"
                )

                train_features = build_horizon_features(
                    train_targets[["fsa", "forecast_origin"]],
                    feature_dataset,
                    horizon=int(horizon),
                )
                validation_features = build_horizon_features(
                    validation_targets[["fsa", "forecast_origin"]],
                    feature_dataset,
                    horizon=int(horizon),
                )

                X_train = train_features[
                    numeric_columns + categorical_columns
                ]
                y_train = train_targets[target_column]

                X_validation = validation_features[
                    numeric_columns + categorical_columns
                ]
                y_validation = validation_targets[target_column]

                pipeline = _build_pipeline(
                    numeric_columns,
                    categorical_columns,
                    parameters,
                    random_seed=int(model_cfg["random_seed"]),
                )

                pipeline.fit(X_train, y_train)
                prediction = pipeline.predict(X_validation)

                metrics = forecasting_metrics(
                    y_validation,
                    prediction,
                )

                rows.append(
                    {
                        "parameter_id": parameter_id,
                        "fold": fold.name,
                        "horizon": int(horizon),
                        **parameters,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def _select_best_parameters(
    tuning_results: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    summary = (
        tuning_results
        .groupby(
            [
                "parameter_id",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_mae", "mean_rmse"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    best = summary.iloc[0]

    best_parameters = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": (
            None
            if pd.isna(best["max_depth"])
            else int(best["max_depth"])
        ),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": (
            str(best["max_features"])
            if isinstance(best["max_features"], str)
            else float(best["max_features"])
        ),
    }

    return best_parameters, summary


def run_random_forest_regressor(
    config_path: str | Path = "configs/random_forest_regressor.yaml",
) -> dict[str, object]:
    """Execute Random Forest tuning and full development-fold evaluation."""
    branch_config, project_root = _load_branch_config(config_path)

    foundation_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(foundation_path)

    feature_dataset = load_feature_dataset(modeling_config, project_root)
    forecast_targets = load_forecast_targets(modeling_config, project_root)

    model_cfg = branch_config["model"]
    horizons = int(model_cfg["horizons"])

    if bool(model_cfg.get("evaluate_final_holdout", False)):
        raise ValueError(
            "Random Forest candidate branch must not evaluate the final 2025 holdout."
        )

    paths = {
        key: resolve_project_path(project_root, value)
        for key, value in branch_config["paths"].items()
        if key.endswith("_dir")
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    numeric_columns, categorical_columns = model_feature_columns(
        feature_dataset
    )

    feature_report = feature_dictionary(feature_dataset)
    save_csv(
        feature_report,
        paths["reports_dir"] / "RF_02_feature_dictionary.csv",
    )

    tuning_results = _tune_parameters(
        feature_dataset,
        forecast_targets,
        modeling_config,
        branch_config,
        numeric_columns,
        categorical_columns,
    )

    best_parameters, tuning_summary = _select_best_parameters(
        tuning_results
    )

    save_csv(
        tuning_results,
        paths["reports_dir"] / "RF_04_tuning_details.csv",
    )
    save_csv(
        tuning_summary,
        paths["reports_dir"] / "RF_04_tuning_summary.csv",
    )

    (
        paths["outputs_dir"] / "best_parameters.json"
    ).write_text(
        json.dumps(best_parameters, indent=2),
        encoding="utf-8",
    )

    folds = build_validation_folds(modeling_config)
    target_columns = [
        f"{model_cfg['target_prefix']}{horizon:02d}"
        for horizon in range(1, horizons + 1)
    ]

    fold_metric_parts = []
    fsa_metric_parts = []
    horizon_metric_parts = []
    prediction_parts = []
    importance_parts = []

    for fold in folds:
        train_targets, validation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=horizons,
        )

        train_targets = train_targets.reset_index(drop=True)
        validation_targets = validation_targets.reset_index(drop=True)

        fold_actual = validation_targets[
            ["fsa", "forecast_origin", *target_columns]
        ].copy()

        fold_predicted = validation_targets[
            ["fsa", "forecast_origin"]
        ].copy()

        for horizon in range(1, horizons + 1):
            target_column = f"{model_cfg['target_prefix']}{horizon:02d}"

            train_features = build_horizon_features(
                train_targets[["fsa", "forecast_origin"]],
                feature_dataset,
                horizon=horizon,
            )
            validation_features = build_horizon_features(
                validation_targets[["fsa", "forecast_origin"]],
                feature_dataset,
                horizon=horizon,
            )

            X_train = train_features[
                numeric_columns + categorical_columns
            ]
            y_train = train_targets[target_column]

            X_validation = validation_features[
                numeric_columns + categorical_columns
            ]

            pipeline = _build_pipeline(
                numeric_columns,
                categorical_columns,
                best_parameters,
                random_seed=int(model_cfg["random_seed"]),
            )

            pipeline.fit(X_train, y_train)
            prediction = pipeline.predict(X_validation)

            fold_predicted[target_column] = prediction

            preprocessor = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]
            feature_names = preprocessor.get_feature_names_out()

            importance_parts.append(
                pd.DataFrame(
                    {
                        "fold": fold.name,
                        "horizon": horizon,
                        "feature": feature_names,
                        "importance": model.feature_importances_,
                    }
                )
            )

            prediction_parts.append(
                pd.DataFrame(
                    {
                        "fold": fold.name,
                        "fsa": validation_targets["fsa"],
                        "forecast_origin": validation_targets["forecast_origin"],
                        "horizon": horizon,
                        "actual": validation_targets[target_column],
                        "predicted": prediction,
                    }
                )
            )

        predicted_targets = fold_predicted[target_columns]

        fold_metrics = evaluate_forecasting_global(
            fold_actual,
            predicted_targets,
            target_prefix=model_cfg["target_prefix"],
        )
        fold_metrics.insert(0, "fold", fold.name)
        fold_metric_parts.append(fold_metrics)

        fsa_metrics = evaluate_forecasting_by_group(
            fold_actual,
            predicted_targets,
            group_columns=["fsa"],
            target_prefix=model_cfg["target_prefix"],
        )
        fsa_metrics.insert(0, "fold", fold.name)
        fsa_metric_parts.append(fsa_metrics)

        horizon_metrics = evaluate_forecasting_by_horizon(
            fold_actual,
            predicted_targets,
            target_prefix=model_cfg["target_prefix"],
        )
        horizon_metrics.insert(0, "fold", fold.name)
        horizon_metric_parts.append(horizon_metrics)

    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    fsa_metrics = pd.concat(fsa_metric_parts, ignore_index=True)
    horizon_metrics = pd.concat(horizon_metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    importances = pd.concat(importance_parts, ignore_index=True)

    global_metrics = pd.DataFrame(
        [
            {
                "model": model_cfg["name"],
                **forecasting_metrics(
                    predictions["actual"],
                    predictions["predicted"],
                ),
            }
        ]
    )

    importance_summary = (
        importances.groupby("feature", observed=True)["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    save_csv(
        fold_metrics,
        paths["reports_dir"] / "RF_05_fold_metrics.csv",
    )
    save_csv(
        fsa_metrics,
        paths["reports_dir"] / "RF_05_fsa_metrics.csv",
    )
    save_csv(
        horizon_metrics,
        paths["reports_dir"] / "RF_05_horizon_metrics.csv",
    )
    save_csv(
        global_metrics,
        paths["reports_dir"] / "RF_05_global_metrics.csv",
    )
    save_csv(
        importance_summary,
        paths["reports_dir"] / "RF_06_feature_importance.csv",
    )

    predictions.to_parquet(
        paths["outputs_dir"] / "random_forest_validation_predictions.parquet",
        index=False,
    )

    print("Random Forest Regressor completed successfully.")
    print("Best parameters:", best_parameters)
    print("Final 2025 holdout evaluated: NO")

    return {
        "feature_dictionary": feature_report,
        "tuning_details": tuning_results,
        "tuning_summary": tuning_summary,
        "best_parameters": best_parameters,
        "fold_metrics": fold_metrics,
        "fsa_metrics": fsa_metrics,
        "horizon_metrics": horizon_metrics,
        "global_metrics": global_metrics,
        "feature_importance": importance_summary,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Random Forest Regressor candidate model."
    )
    parser.add_argument(
        "--config",
        default="configs/random_forest_regressor.yaml",
    )
    args = parser.parse_args()
    run_random_forest_regressor(args.config)


if __name__ == "__main__":
    main()
