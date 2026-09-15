"""Train, tune, evaluate, and document HistGradientBoostingClassifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_modeling_config,
    resolve_project_path,
    save_csv,
)
from src.ontario_peak_risk.modeling.evaluation import (
    probabilities_to_binary_predictions,
)
from src.ontario_peak_risk.modeling.metrics import peak_risk_metrics
from src.ontario_peak_risk.modeling.peak_policy import (
    build_peak_horizon_targets,
    fit_peak_thresholds,
    label_peak_evaluation_data,
)
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_forecast_origin_fold,
    slice_observation_fold,
)

from .features import (
    build_horizon_features,
    feature_dictionary,
    model_feature_columns,
)
from .preprocessing import build_hist_gradient_boosting_preprocessor


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
    preprocessor = build_hist_gradient_boosting_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=float(parameters["learning_rate"]),
        max_iter=int(parameters["max_iter"]),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=float(parameters["l2_regularization"]),
        early_stopping=False,
        random_state=random_seed,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _evaluate_group(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for key, group in frame.groupby(
        group_columns,
        observed=True,
        sort=True,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        rows.append(
            {
                **dict(zip(group_columns, key)),
                **peak_risk_metrics(
                    group["actual"],
                    group["predicted"],
                    group["probability"],
                ),
            }
        )

    return pd.DataFrame(rows)


def _prepare_peak_targets(
    feature_dataset: pd.DataFrame,
    fold,
    peak_cfg: dict,
    model_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_observations, validation_observations = slice_observation_fold(
        feature_dataset,
        fold,
        time_column="timestamp",
    )

    estimator = fit_peak_thresholds(
        training_observations,
        percentile=float(peak_cfg["percentile_threshold"]),
        target_column=peak_cfg["target_column"],
        grouping_columns=list(peak_cfg["threshold_grouping"]),
    )

    labeled_training = label_peak_evaluation_data(
        estimator,
        training_observations,
    )
    labeled_validation = label_peak_evaluation_data(
        estimator,
        validation_observations,
    )

    training_targets = build_peak_horizon_targets(
        labeled_training,
        horizons=int(model_cfg["horizons"]),
        target_prefix=model_cfg["target_prefix"],
        require_complete_horizon=True,
    )
    validation_targets = build_peak_horizon_targets(
        labeled_validation,
        horizons=int(model_cfg["horizons"]),
        target_prefix=model_cfg["target_prefix"],
        require_complete_horizon=True,
    )

    safe_training, _ = slice_forecast_origin_fold(
        training_targets,
        fold,
        time_column="forecast_origin",
        max_horizon_hours=int(model_cfg["horizons"]),
    )
    _, safe_validation = slice_forecast_origin_fold(
        validation_targets,
        fold,
        time_column="forecast_origin",
        max_horizon_hours=int(model_cfg["horizons"]),
    )

    return (
        safe_training.reset_index(drop=True),
        safe_validation.reset_index(drop=True),
    )


def _tune_parameters(
    feature_dataset: pd.DataFrame,
    modeling_config: dict,
    branch_config: dict,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    model_cfg = branch_config["model"]
    peak_cfg = modeling_config["modeling"]["peak_risk"]
    folds = build_validation_folds(modeling_config)

    rows = []

    for parameter_id, parameters in enumerate(
        model_cfg["tuning"]["parameter_grid"],
        start=1,
    ):
        for fold in folds:
            training_targets, validation_targets = _prepare_peak_targets(
                feature_dataset,
                fold,
                peak_cfg,
                model_cfg,
            )

            for horizon in model_cfg["tuning"]["representative_horizons"]:
                target_column = (
                    f"{model_cfg['target_prefix']}{int(horizon):02d}"
                )

                train_features = build_horizon_features(
                    training_targets[["fsa", "forecast_origin"]],
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
                y_train = training_targets[target_column].astype("int8")

                X_validation = validation_features[
                    numeric_columns + categorical_columns
                ]
                y_validation = validation_targets[target_column].astype("int8")

                pipeline = _build_pipeline(
                    numeric_columns,
                    categorical_columns,
                    parameters,
                    random_seed=int(model_cfg["random_seed"]),
                )

                pipeline.fit(X_train, y_train)

                probability = pipeline.predict_proba(X_validation)[:, 1]
                predicted = probabilities_to_binary_predictions(
                    probability,
                    threshold=float(model_cfg["probability_threshold"]),
                )

                metrics = peak_risk_metrics(
                    y_validation,
                    predicted,
                    probability,
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
    grouping = [
        "parameter_id",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    ]

    summary = (
        tuning_results.groupby(
            grouping,
            observed=True,
        )
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_recall=("recall", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_pr_auc", "mean_f1"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    best = summary.iloc[0]

    parameters = {
        "learning_rate": float(best["learning_rate"]),
        "max_iter": int(best["max_iter"]),
        "max_leaf_nodes": int(best["max_leaf_nodes"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "l2_regularization": float(best["l2_regularization"]),
    }

    return parameters, summary


def run_hist_gradient_boosting_classifier(
    config_path: str | Path = "configs/hist_gradient_boosting_classifier.yaml",
) -> dict[str, object]:
    """Execute HGB tuning and full development-fold evaluation."""
    branch_config, project_root = _load_branch_config(config_path)

    foundation_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(foundation_path)

    feature_dataset = load_feature_dataset(modeling_config, project_root)

    model_cfg = branch_config["model"]
    peak_cfg = modeling_config["modeling"]["peak_risk"]

    if bool(model_cfg.get("evaluate_final_holdout", False)):
        raise ValueError(
            "HistGradientBoosting candidate branch must not evaluate the final 2025 holdout."
        )

    if float(model_cfg["probability_threshold"]) != float(
        peak_cfg["probability_threshold"]
    ):
        raise ValueError(
            "Candidate probability threshold must match Modeling Foundation."
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
        paths["reports_dir"] / "HGB_02_feature_dictionary.csv",
    )

    tuning_results = _tune_parameters(
        feature_dataset,
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
        paths["reports_dir"] / "HGB_04_tuning_details.csv",
    )
    save_csv(
        tuning_summary,
        paths["reports_dir"] / "HGB_04_tuning_summary.csv",
    )

    (
        paths["outputs_dir"] / "best_parameters.json"
    ).write_text(
        json.dumps(best_parameters, indent=2),
        encoding="utf-8",
    )

    folds = build_validation_folds(modeling_config)

    fold_metric_parts = []
    fsa_metric_parts = []
    horizon_metric_parts = []
    balance_parts = []
    prediction_parts = []
    importance_parts = []

    for fold in folds:
        training_targets, validation_targets = _prepare_peak_targets(
            feature_dataset,
            fold,
            peak_cfg,
            model_cfg,
        )

        fold_prediction_parts = []

        for horizon in range(1, int(model_cfg["horizons"]) + 1):
            target_column = (
                f"{model_cfg['target_prefix']}{horizon:02d}"
            )

            train_features = build_horizon_features(
                training_targets[["fsa", "forecast_origin"]],
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
            y_train = training_targets[target_column].astype("int8")

            X_validation = validation_features[
                numeric_columns + categorical_columns
            ]
            y_validation = validation_targets[target_column].astype("int8")

            if y_train.nunique() < 2:
                raise ValueError(
                    f"{fold.name} {target_column} contains only one training class."
                )

            pipeline = _build_pipeline(
                numeric_columns,
                categorical_columns,
                best_parameters,
                random_seed=int(model_cfg["random_seed"]),
            )

            pipeline.fit(X_train, y_train)

            probability = pipeline.predict_proba(X_validation)[:, 1]
            predicted = probabilities_to_binary_predictions(
                probability,
                threshold=float(model_cfg["probability_threshold"]),
            )

            horizon_predictions = pd.DataFrame(
                {
                    "fold": fold.name,
                    "fsa": validation_targets["fsa"].to_numpy(),
                    "forecast_origin": validation_targets[
                        "forecast_origin"
                    ].to_numpy(),
                    "horizon": horizon,
                    "actual": y_validation.to_numpy(),
                    "predicted": predicted,
                    "probability": probability,
                }
            )

            fold_prediction_parts.append(horizon_predictions)

            horizon_metric_parts.append(
                pd.DataFrame(
                    [
                        {
                            "fold": fold.name,
                            "horizon": horizon,
                            **peak_risk_metrics(
                                y_validation,
                                predicted,
                                probability,
                            ),
                        }
                    ]
                )
            )

            # Lightweight permutation importance for interpretation.
            if horizon in {1, 12, 24}:
                sample_n = min(3000, len(X_validation))
                sample_index = X_validation.sample(
                    n=sample_n,
                    random_state=int(model_cfg["random_seed"]),
                ).index

                perm = permutation_importance(
                    pipeline,
                    X_validation.loc[sample_index],
                    y_validation.loc[sample_index],
                    scoring="average_precision",
                    n_repeats=3,
                    random_state=int(model_cfg["random_seed"]),
                    n_jobs=-1,
                )

                importance_parts.append(
                    pd.DataFrame(
                        {
                            "fold": fold.name,
                            "horizon": horizon,
                            "feature": (
                                numeric_columns + categorical_columns
                            ),
                            "importance": perm.importances_mean,
                        }
                    )
                )

        fold_predictions = pd.concat(
            fold_prediction_parts,
            ignore_index=True,
        )
        prediction_parts.append(fold_predictions)

        fold_metric_parts.append(
            pd.DataFrame(
                [
                    {
                        "fold": fold.name,
                        **peak_risk_metrics(
                            fold_predictions["actual"],
                            fold_predictions["predicted"],
                            fold_predictions["probability"],
                        ),
                    }
                ]
            )
        )

        fsa_metric_parts.append(
            _evaluate_group(
                fold_predictions,
                ["fold", "fsa"],
            )
        )

        balance = (
            fold_predictions.groupby(
                ["fold", "horizon"],
                observed=True,
            )
            .agg(
                observations=("actual", "size"),
                peak_hours=("actual", "sum"),
            )
            .reset_index()
        )

        balance["peak_rate_pct"] = (
            balance["peak_hours"]
            / balance["observations"]
            * 100
        )
        balance_parts.append(balance)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    fsa_metrics = pd.concat(fsa_metric_parts, ignore_index=True)
    horizon_metrics = pd.concat(horizon_metric_parts, ignore_index=True)
    class_balance = pd.concat(balance_parts, ignore_index=True)

    global_metrics = pd.DataFrame(
        [
            {
                "model": model_cfg["name"],
                **peak_risk_metrics(
                    predictions["actual"],
                    predictions["predicted"],
                    predictions["probability"],
                ),
            }
        ]
    )

    if importance_parts:
        importance = pd.concat(
            importance_parts,
            ignore_index=True,
        )
        importance_summary = (
            importance.groupby("feature", observed=True)["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
    else:
        importance_summary = pd.DataFrame(
            columns=["feature", "importance"]
        )

    save_csv(
        fold_metrics,
        paths["reports_dir"] / "HGB_05_fold_metrics.csv",
    )
    save_csv(
        fsa_metrics,
        paths["reports_dir"] / "HGB_05_fsa_metrics.csv",
    )
    save_csv(
        horizon_metrics,
        paths["reports_dir"] / "HGB_05_horizon_metrics.csv",
    )
    save_csv(
        global_metrics,
        paths["reports_dir"] / "HGB_05_global_metrics.csv",
    )
    save_csv(
        class_balance,
        paths["reports_dir"] / "HGB_05_class_balance.csv",
    )
    save_csv(
        importance_summary,
        paths["reports_dir"] / "HGB_06_permutation_importance.csv",
    )

    predictions.to_parquet(
        paths["outputs_dir"]
        / "hist_gradient_boosting_validation_predictions.parquet",
        index=False,
    )

    print("HistGradientBoostingClassifier completed successfully.")
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
        "class_balance": class_balance,
        "permutation_importance": importance_summary,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HistGradientBoostingClassifier candidate model."
    )
    parser.add_argument(
        "--config",
        default="configs/hist_gradient_boosting_classifier.yaml",
    )
    args = parser.parse_args()
    run_hist_gradient_boosting_classifier(args.config)


if __name__ == "__main__":
    main()
