"""Train, tune, evaluate, and interpret RandomForestClassifier for Peak-Risk."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
    resolve_project_path,
    save_csv,
)
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_fold,
    slice_forecast_origin_fold,
)

from .diagnostics import (
    classification_metrics,
    threshold_sweep,
)
from .features import (
    build_horizon_features,
    feature_dictionary,
    model_feature_columns,
)
from .preprocessing import build_tree_preprocessor
from .targets import (
    build_peak_labels,
    fit_peak_threshold_table,
)


def _load_branch_config(
    config_path: str | Path,
) -> tuple[dict, Path]:
    path = Path(config_path).resolve()

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config, path.parent.parent


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    parameters: dict,
    *,
    random_seed: int,
    class_ratio: float,
) -> Pipeline:
    preprocessor = build_tree_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    model = RandomForestClassifier(
        n_estimators=int(parameters["n_estimators"]),
        max_depth=(
            None
            if parameters["max_depth"] is None
            else int(parameters["max_depth"])
        ),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=parameters["max_features"],
        class_weight=parameters["class_weight"],
        random_state=random_seed,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _prepare_horizon(
    target_frame: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    threshold_table: pd.DataFrame,
    horizon: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    labels = build_peak_labels(
        target_frame,
        threshold_table,
        horizon,
    )

    feature_frame = build_horizon_features(
        target_frame[["fsa", "forecast_origin"]],
        feature_dataset,
        horizon,
    )

    X = feature_frame[
        numeric_columns + categorical_columns
    ].copy()

    y = labels["actual_peak"].astype("int8")

    keys = labels[
        [
            "fsa",
            "forecast_origin",
            "target_timestamp",
            "season",
            "peak_threshold",
        ]
    ].copy()

    return X, y, keys


def _class_ratio(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


def _tune(
    *,
    feature_dataset: pd.DataFrame,
    forecast_targets: pd.DataFrame,
    modeling_config: dict,
    branch_config: dict,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    cfg = branch_config["model"]
    tuning = cfg["tuning"]

    folds = build_validation_folds(
        modeling_config
    )

    tasks = [
        (parameter_id, parameters, fold, int(horizon))
        for parameter_id, parameters in enumerate(
            tuning["parameter_grid"],
            start=1,
        )
        for fold in folds
        for horizon in tuning["representative_horizons"]
    ]

    rows = []
    started_all = time.perf_counter()

    print(
        f"Tuning tasks: {len(tasks)} "
        f"({len(tuning['parameter_grid'])} parameter sets × "
        f"{len(folds)} folds × "
        f"{len(tuning['representative_horizons'])} horizons)",
        flush=True,
    )

    for task_number, (
        parameter_id,
        parameters,
        fold,
        horizon,
    ) in enumerate(tasks, start=1):

        task_started = time.perf_counter()

        train_observations, _ = slice_fold(
            feature_dataset,
            fold,
            time_column="timestamp",
        )

        train_targets, validation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=int(cfg["horizons"]),
        )

        threshold_table = fit_peak_threshold_table(
            train_observations,
            percentile=float(cfg["peak_percentile"]),
        )

        X_train, y_train, _ = _prepare_horizon(
            train_targets.reset_index(drop=True),
            feature_dataset,
            threshold_table,
            horizon,
            numeric_columns,
            categorical_columns,
        )

        X_validation, y_validation, _ = _prepare_horizon(
            validation_targets.reset_index(drop=True),
            feature_dataset,
            threshold_table,
            horizon,
            numeric_columns,
            categorical_columns,
        )

        class_ratio = _class_ratio(y_train)

        pipeline = _build_pipeline(
            numeric_columns,
            categorical_columns,
            parameters,
            random_seed=int(cfg["random_seed"]),
            class_ratio=class_ratio,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        probability = pipeline.predict_proba(
            X_validation
        )[:, 1]

        predicted = (
            probability
            >= float(cfg["probability_threshold"])
        ).astype("int8")

        metrics = classification_metrics(
            y_validation,
            predicted,
            probability,
        )

        task_seconds = time.perf_counter() - task_started
        elapsed = time.perf_counter() - started_all
        average = elapsed / task_number
        eta = average * (len(tasks) - task_number)

        rows.append(
            {
                "parameter_id": parameter_id,
                "fold": fold.name,
                "horizon": horizon,
                "class_ratio": class_ratio,
                "task_seconds": task_seconds,
                **parameters,
                **metrics,
            }
        )

        print(
            f"[Tuning {task_number}/{len(tasks)}] "
            f"params={parameter_id} | {fold.name} | h+{horizon} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"time={_format_seconds(task_seconds)} | "
            f"ETA={_format_seconds(eta)}",
            flush=True,
        )

    return pd.DataFrame(rows)


def _select_best_parameters(
    tuning_results: pd.DataFrame,
    parameter_grid: list[dict],
) -> tuple[dict, pd.DataFrame]:
    summary = (
        tuning_results
        .groupby("parameter_id", observed=True)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_recall=("recall", "mean"),
            mean_precision=("precision", "mean"),
            total_seconds=("task_seconds", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "mean_pr_auc",
                "mean_f1",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_id = int(
        summary.iloc[0]["parameter_id"]
    )

    best_parameters = dict(
        parameter_grid[best_id - 1]
    )

    return best_parameters, summary


def _evaluate_grouped(
    predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    global_metrics = pd.DataFrame(
        [
            {
                "model": "random_forest_classifier",
                **classification_metrics(
                    predictions["actual"],
                    predictions["predicted"],
                    predictions["probability"],
                ),
            }
        ]
    )

    fold_rows = []
    fsa_rows = []
    horizon_rows = []
    class_balance_rows = []

    for fold, group in predictions.groupby(
        "fold",
        observed=True,
    ):
        fold_rows.append(
            {
                "fold": fold,
                **classification_metrics(
                    group["actual"],
                    group["predicted"],
                    group["probability"],
                ),
            }
        )

    for (fold, fsa), group in predictions.groupby(
        ["fold", "fsa"],
        observed=True,
    ):
        fsa_rows.append(
            {
                "fold": fold,
                "fsa": fsa,
                **classification_metrics(
                    group["actual"],
                    group["predicted"],
                    group["probability"],
                ),
            }
        )

    for (fold, horizon), group in predictions.groupby(
        ["fold", "horizon"],
        observed=True,
    ):
        horizon_rows.append(
            {
                "fold": fold,
                "horizon": int(horizon),
                **classification_metrics(
                    group["actual"],
                    group["predicted"],
                    group["probability"],
                ),
            }
        )

        class_balance_rows.append(
            {
                "fold": fold,
                "horizon": int(horizon),
                "observations": len(group),
                "actual_peaks": int(group["actual"].sum()),
                "actual_peak_rate_pct": float(
                    group["actual"].mean() * 100
                ),
            }
        )

    return {
        "global_metrics": global_metrics,
        "fold_metrics": pd.DataFrame(fold_rows),
        "fsa_metrics": pd.DataFrame(fsa_rows),
        "horizon_metrics": pd.DataFrame(horizon_rows),
        "class_balance": pd.DataFrame(class_balance_rows),
    }


def run_model(
    config_path: str | Path = "configs/random_forest_classifier.yaml",
    *,
    resume: bool = True,
) -> dict[str, object]:
    branch_config, project_root = _load_branch_config(
        config_path
    )

    foundation_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )

    modeling_config, _ = load_modeling_config(
        foundation_path
    )

    feature_dataset = load_feature_dataset(
        modeling_config,
        project_root,
    )

    forecast_targets = load_forecast_targets(
        modeling_config,
        project_root,
    )

    cfg = branch_config["model"]

    if bool(cfg.get("evaluate_final_holdout", False)):
        raise ValueError(
            "Candidate-model development must not evaluate the 2025 holdout."
        )

    if float(cfg["probability_threshold"]) != 0.50:
        raise ValueError(
            "The official comparison threshold must remain 0.50. "
            "Alternative thresholds belong to diagnostic analysis."
        )

    paths = {
        key: resolve_project_path(project_root, value)
        for key, value in branch_config["paths"].items()
        if key.endswith("_dir")
    }

    for path in paths.values():
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    numeric_columns, categorical_columns = model_feature_columns(
        feature_dataset
    )

    feature_report = feature_dictionary(
        feature_dataset
    )

    save_csv(
        feature_report,
        paths["reports_dir"] / "RFC_02_feature_dictionary.csv",
    )

    print("=" * 72, flush=True)
    print("RandomForestClassifier — PEAK-RISK CANDIDATE", flush=True)
    print("Official threshold: 0.50", flush=True)
    print("Selection metric: PR-AUC", flush=True)
    print("Final 2025 holdout evaluated: NO", flush=True)
    print("=" * 72, flush=True)

    tuning_results = _tune(
        feature_dataset=feature_dataset,
        forecast_targets=forecast_targets,
        modeling_config=modeling_config,
        branch_config=branch_config,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    best_parameters, tuning_summary = _select_best_parameters(
        tuning_results,
        cfg["tuning"]["parameter_grid"],
    )

    save_csv(
        tuning_results,
        paths["reports_dir"] / "RFC_04_tuning_details.csv",
    )

    save_csv(
        tuning_summary,
        paths["reports_dir"] / "RFC_04_tuning_summary.csv",
    )

    (
        paths["outputs_dir"] / "best_parameters.json"
    ).write_text(
        json.dumps(
            best_parameters,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Selected parameters: {best_parameters}",
        flush=True,
    )

    folds = build_validation_folds(
        modeling_config
    )

    checkpoint_dir = (
        paths["outputs_dir"] / "checkpoints"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_tasks = [
        (fold, horizon)
        for fold in folds
        for horizon in range(
            1,
            int(cfg["horizons"]) + 1,
        )
    ]

    prediction_parts = []
    importance_parts = []
    full_started = time.perf_counter()

    for task_number, (
        fold,
        horizon,
    ) in enumerate(full_tasks, start=1):

        checkpoint_path = (
            checkpoint_dir
            / f"{fold.name}__h{horizon:02d}.parquet"
        )

        importance_checkpoint = (
            checkpoint_dir
            / f"{fold.name}__h{horizon:02d}__importance.csv"
        )

        if resume and checkpoint_path.exists():
            part = pd.read_parquet(
                checkpoint_path
            )
            prediction_parts.append(part)

            if importance_checkpoint.exists():
                importance_parts.append(
                    pd.read_csv(
                        importance_checkpoint
                    )
                )

            print(
                f"[Full {task_number}/{len(full_tasks)}] "
                f"[resume] {fold.name} h+{horizon}",
                flush=True,
            )
            continue

        task_started = time.perf_counter()

        train_observations, _ = slice_fold(
            feature_dataset,
            fold,
            time_column="timestamp",
        )

        train_targets, validation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=int(cfg["horizons"]),
        )

        train_targets = train_targets.reset_index(
            drop=True
        )
        validation_targets = validation_targets.reset_index(
            drop=True
        )

        threshold_table = fit_peak_threshold_table(
            train_observations,
            percentile=float(cfg["peak_percentile"]),
        )

        X_train, y_train, _ = _prepare_horizon(
            train_targets,
            feature_dataset,
            threshold_table,
            horizon,
            numeric_columns,
            categorical_columns,
        )

        X_validation, y_validation, keys = _prepare_horizon(
            validation_targets,
            feature_dataset,
            threshold_table,
            horizon,
            numeric_columns,
            categorical_columns,
        )

        class_ratio = _class_ratio(
            y_train
        )

        pipeline = _build_pipeline(
            numeric_columns,
            categorical_columns,
            best_parameters,
            random_seed=int(cfg["random_seed"]),
            class_ratio=class_ratio,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        probability = pipeline.predict_proba(
            X_validation
        )[:, 1]

        predicted = (
            probability
            >= float(cfg["probability_threshold"])
        ).astype("int8")

        part = pd.DataFrame(
            {
                "fold": fold.name,
                "fsa": keys["fsa"].to_numpy(),
                "forecast_origin": keys["forecast_origin"].to_numpy(),
                "target_timestamp": keys["target_timestamp"].to_numpy(),
                "season": keys["season"].to_numpy(),
                "peak_threshold": keys["peak_threshold"].to_numpy(),
                "horizon": horizon,
                "actual": y_validation.to_numpy(),
                "predicted": predicted,
                "probability": probability,
            }
        )

        part.to_parquet(
            checkpoint_path,
            index=False,
        )
        prediction_parts.append(part)

        # Model-native feature importance.
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame(
                {
                    "fold": fold.name,
                    "horizon": horizon,
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }
            )
            importance.to_csv(
                importance_checkpoint,
                index=False,
            )
            importance_parts.append(
                importance
            )

        task_seconds = time.perf_counter() - task_started
        elapsed = time.perf_counter() - full_started
        average = elapsed / task_number
        eta = average * (
            len(full_tasks) - task_number
        )

        task_metrics = classification_metrics(
            y_validation,
            predicted,
            probability,
        )

        print(
            f"[Full {task_number}/{len(full_tasks)}] "
            f"{fold.name} h+{horizon} | "
            f"PR-AUC={task_metrics['pr_auc']:.4f} | "
            f"Recall={task_metrics['recall']:.4f} | "
            f"time={_format_seconds(task_seconds)} | "
            f"ETA={_format_seconds(eta)}",
            flush=True,
        )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    grouped = _evaluate_grouped(
        predictions
    )

    threshold_analysis = threshold_sweep(
        predictions
    )

    if importance_parts:
        importance = pd.concat(
            importance_parts,
            ignore_index=True,
        )

        importance_summary = (
            importance
            .groupby("feature", observed=True)["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
    else:
        importance_summary = pd.DataFrame(
            columns=["feature", "importance"]
        )

    save_csv(
        grouped["global_metrics"],
        paths["reports_dir"] / "RFC_05_global_metrics.csv",
    )
    save_csv(
        grouped["fold_metrics"],
        paths["reports_dir"] / "RFC_05_fold_metrics.csv",
    )
    save_csv(
        grouped["fsa_metrics"],
        paths["reports_dir"] / "RFC_05_fsa_metrics.csv",
    )
    save_csv(
        grouped["horizon_metrics"],
        paths["reports_dir"] / "RFC_05_horizon_metrics.csv",
    )
    save_csv(
        grouped["class_balance"],
        paths["reports_dir"] / "RFC_05_class_balance.csv",
    )
    save_csv(
        threshold_analysis,
        paths["reports_dir"] / "RFC_05_threshold_analysis.csv",
    )
    save_csv(
        importance_summary,
        paths["reports_dir"] / "RFC_06_feature_importance.csv",
    )

    predictions.to_parquet(
        paths["outputs_dir"]
        / "random_forest_classifier_validation_predictions.parquet",
        index=False,
    )

    print("=" * 72, flush=True)
    print("RandomForestClassifier completed successfully.", flush=True)
    print(f"Best parameters: {best_parameters}", flush=True)
    print("Official comparison threshold: 0.50", flush=True)
    print("Final 2025 holdout evaluated: NO", flush=True)
    print("=" * 72, flush=True)

    return {
        "feature_dictionary": feature_report,
        "tuning_details": tuning_results,
        "tuning_summary": tuning_summary,
        "best_parameters": best_parameters,
        **grouped,
        "threshold_analysis": threshold_analysis,
        "feature_importance": importance_summary,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RandomForestClassifier Peak-Risk candidate."
    )
    parser.add_argument(
        "--config",
        default="configs/random_forest_classifier.yaml",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    run_model(
        args.config,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
