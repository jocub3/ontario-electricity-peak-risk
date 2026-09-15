"""Train, tune, evaluate, and interpret XGBRegressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from xgboost import XGBRegressor
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
from .preprocessing import build_tree_preprocessor


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
    preprocessor = build_tree_preprocessor(
        numeric_columns,
        categorical_columns,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(parameters["n_estimators"]),
        max_depth=int(parameters["max_depth"]),
        learning_rate=float(parameters["learning_rate"]),
        subsample=float(parameters["subsample"]),
        colsample_bytree=float(parameters["colsample_bytree"]),
        min_child_weight=float(parameters["min_child_weight"]),
        reg_lambda=float(parameters["reg_lambda"]),
        random_state=random_seed,
        n_jobs=-1,
        tree_method="hist",
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
    cfg = branch_config["model"]
    folds = build_validation_folds(modeling_config)
    rows = []

    for parameter_id, parameters in enumerate(
        cfg["tuning"]["parameter_grid"],
        start=1,
    ):
        for fold in folds:
            train_targets, validation_targets = slice_forecast_origin_fold(
                forecast_targets,
                fold,
                time_column="forecast_origin",
                max_horizon_hours=int(cfg["horizons"]),
            )
            train_targets = train_targets.reset_index(drop=True)
            validation_targets = validation_targets.reset_index(drop=True)

            for horizon in cfg["tuning"]["representative_horizons"]:
                target_column = f"{cfg['target_prefix']}{int(horizon):02d}"

                train_features = build_horizon_features(
                    train_targets[["fsa", "forecast_origin"]],
                    feature_dataset,
                    int(horizon),
                )
                validation_features = build_horizon_features(
                    validation_targets[["fsa", "forecast_origin"]],
                    feature_dataset,
                    int(horizon),
                )

                X_train = train_features[numeric_columns + categorical_columns]
                y_train = train_targets[target_column]
                X_validation = validation_features[numeric_columns + categorical_columns]
                y_validation = validation_targets[target_column]

                pipeline = _build_pipeline(
                    numeric_columns,
                    categorical_columns,
                    parameters,
                    int(cfg["random_seed"]),
                )
                pipeline.fit(X_train, y_train)
                prediction = pipeline.predict(X_validation)

                rows.append(
                    {
                        "parameter_id": parameter_id,
                        "fold": fold.name,
                        "horizon": int(horizon),
                        **parameters,
                        **forecasting_metrics(y_validation, prediction),
                    }
                )

    return pd.DataFrame(rows)


def _select_best_parameters(
    tuning_results: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    grouping = ['parameter_id', 'n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'reg_lambda']

    summary = (
        tuning_results.groupby(grouping, dropna=False, observed=True)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape=("mape", "mean"),
        )
        .reset_index()
        .sort_values(["mean_mae", "mean_rmse"])
        .reset_index(drop=True)
    )

    best = summary.iloc[0]
    best_parameters = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": float(best["min_child_weight"]),
        "reg_lambda": float(best["reg_lambda"]),
    }
    return best_parameters, summary


def run_model(
    config_path: str | Path = "configs/xgboost_regressor.yaml",
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
        raise ValueError(
            "XGBRegressor candidate must not evaluate the final 2025 holdout."
        )

    paths = {
        key: resolve_project_path(project_root, value)
        for key, value in branch_config["paths"].items()
        if key.endswith("_dir")
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    numeric_columns, categorical_columns = model_feature_columns(feature_dataset)

    feature_report = feature_dictionary(feature_dataset)
    save_csv(
        feature_report,
        paths["reports_dir"] / "XGB_02_feature_dictionary.csv",
    )

    tuning_results = _tune_parameters(
        feature_dataset,
        forecast_targets,
        modeling_config,
        branch_config,
        numeric_columns,
        categorical_columns,
    )

    best_parameters, tuning_summary = _select_best_parameters(tuning_results)

    save_csv(
        tuning_results,
        paths["reports_dir"] / "XGB_04_tuning_details.csv",
    )
    save_csv(
        tuning_summary,
        paths["reports_dir"] / "XGB_04_tuning_summary.csv",
    )

    (
        paths["outputs_dir"] / "best_parameters.json"
    ).write_text(
        json.dumps(best_parameters, indent=2),
        encoding="utf-8",
    )

    folds = build_validation_folds(modeling_config)
    target_columns = [
        f"{cfg['target_prefix']}{horizon:02d}"
        for horizon in range(1, int(cfg["horizons"]) + 1)
    ]

    prediction_parts = []
    importance_parts = []
    fold_metric_parts = []
    fsa_metric_parts = []
    horizon_metric_parts = []

    for fold in folds:
        train_targets, validation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=int(cfg["horizons"]),
        )

        train_targets = train_targets.reset_index(drop=True)
        validation_targets = validation_targets.reset_index(drop=True)

        actual = validation_targets[
            ["fsa", "forecast_origin", *target_columns]
        ].copy()

        predicted_frame = validation_targets[
            ["fsa", "forecast_origin"]
        ].copy()

        for horizon in range(1, int(cfg["horizons"]) + 1):
            target_column = f"{cfg['target_prefix']}{horizon:02d}"

            train_features = build_horizon_features(
                train_targets[["fsa", "forecast_origin"]],
                feature_dataset,
                horizon,
            )
            validation_features = build_horizon_features(
                validation_targets[["fsa", "forecast_origin"]],
                feature_dataset,
                horizon,
            )

            X_train = train_features[numeric_columns + categorical_columns]
            y_train = train_targets[target_column]
            X_validation = validation_features[numeric_columns + categorical_columns]

            pipeline = _build_pipeline(
                numeric_columns,
                categorical_columns,
                best_parameters,
                int(cfg["random_seed"]),
            )

            pipeline.fit(X_train, y_train)
            prediction = pipeline.predict(X_validation)

            predicted_frame[target_column] = prediction

            model = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()

            if hasattr(model, "feature_importances_"):
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

        predicted_targets = predicted_frame[target_columns]

        fold_metrics = evaluate_forecasting_global(
            actual,
            predicted_targets,
            target_prefix=cfg["target_prefix"],
        )
        fold_metrics.insert(0, "fold", fold.name)
        fold_metric_parts.append(fold_metrics)

        fsa_metrics = evaluate_forecasting_by_group(
            actual,
            predicted_targets,
            group_columns=["fsa"],
            target_prefix=cfg["target_prefix"],
        )
        fsa_metrics.insert(0, "fold", fold.name)
        fsa_metric_parts.append(fsa_metrics)

        horizon_metrics = evaluate_forecasting_by_horizon(
            actual,
            predicted_targets,
            target_prefix=cfg["target_prefix"],
        )
        horizon_metrics.insert(0, "fold", fold.name)
        horizon_metric_parts.append(horizon_metrics)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["residual"] = predictions["actual"] - predictions["predicted"]

    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    fsa_metrics = pd.concat(fsa_metric_parts, ignore_index=True)
    horizon_metrics = pd.concat(horizon_metric_parts, ignore_index=True)

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

    if importance_parts:
        importance = pd.concat(importance_parts, ignore_index=True)
        importance_summary = (
            importance.groupby("feature", observed=True)["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
    else:
        importance_summary = pd.DataFrame(columns=["feature", "importance"])

    save_csv(fold_metrics, paths["reports_dir"] / "XGB_05_fold_metrics.csv")
    save_csv(fsa_metrics, paths["reports_dir"] / "XGB_05_fsa_metrics.csv")
    save_csv(
        horizon_metrics,
        paths["reports_dir"] / "XGB_05_horizon_metrics.csv",
    )
    save_csv(
        global_metrics,
        paths["reports_dir"] / "XGB_05_global_metrics.csv",
    )
    save_csv(
        importance_summary,
        paths["reports_dir"] / "XGB_06_feature_importance.csv",
    )

    predictions.to_parquet(
        paths["outputs_dir"] / "xgboost_validation_predictions.parquet",
        index=False,
    )

    print("XGBRegressor completed successfully.")
    print("Best parameters:", best_parameters)
    print("Final 2025 holdout evaluated: NO")

    return {
        "feature_dictionary": feature_report,
        "tuning_details": tuning_results,
        "tuning_summary": tuning_summary,
        "best_parameters": best_parameters,
        "global_metrics": global_metrics,
        "fold_metrics": fold_metrics,
        "fsa_metrics": fsa_metrics,
        "horizon_metrics": horizon_metrics,
        "feature_importance": importance_summary,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run XGBRegressor candidate model."
    )
    parser.add_argument("--config", default="configs/xgboost_regressor.yaml")
    args = parser.parse_args()
    run_model(args.config)


if __name__ == "__main__":
    main()
