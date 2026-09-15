"""Run the Daily Seasonal Naive Forecasting baseline on development folds only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

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
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_forecast_origin_fold,
)

from .model import (
    build_daily_seasonal_naive_predictions,
    validate_daily_seasonal_naive_availability,
)


def _load_branch_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config, path.parent.parent


def _write_summary(
    docs_dir: Path,
    fold_metrics: pd.DataFrame,
    global_metrics: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    try:
        fold_table = fold_metrics.to_markdown(index=False)
        global_table = global_metrics.to_markdown(index=False)
        availability_table = availability.to_markdown(index=False)
    except ImportError:
        fold_table = fold_metrics.to_string(index=False)
        global_table = global_metrics.to_string(index=False)
        availability_table = availability.to_string(index=False)

    content = f"""# Daily Seasonal Naive Forecasting Baseline

## Baseline definition

For each future target hour, the model predicts electricity demand using the
observed demand from the same hour 24 hours earlier.

No learned preprocessing, feature scaling, parameter fitting, or hyperparameter
tuning is used.

## Development policy

Only the shared validation folds are evaluated in this branch.

The final 2025 holdout is not evaluated and remains protected for the final
model-selection stage.

## Fold metrics

{fold_table}

## Pooled development metrics

{global_table}

## Prediction coverage

{availability_table}

## Interpretation

This model is the Forecasting reference benchmark. Candidate Forecasting models
must demonstrate improvement relative to this baseline under the same common
splits, horizons, and evaluation metrics.
"""

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "SNB_05_Seasonal_Naive_Baseline_Summary.md").write_text(
        content,
        encoding="utf-8",
    )


def run_seasonal_naive_baseline(
    config_path: str | Path = "configs/seasonal_naive_baseline.yaml",
) -> dict[str, pd.DataFrame]:
    """Execute the Seasonal Naive baseline using validation folds only."""
    branch_config, project_root = _load_branch_config(config_path)

    foundation_config_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(foundation_config_path)

    feature_dataset = load_feature_dataset(modeling_config, project_root)
    forecast_targets = load_forecast_targets(modeling_config, project_root)

    model_cfg = branch_config["model"]
    horizons = int(model_cfg["horizons"])

    if bool(model_cfg.get("evaluate_final_holdout", False)):
        raise ValueError(
            "The Seasonal Naive baseline branch must not evaluate the final 2025 holdout."
        )

    reports_dir = resolve_project_path(
        project_root,
        branch_config["paths"]["reports_dir"],
    )
    docs_dir = resolve_project_path(
        project_root,
        branch_config["paths"]["docs_dir"],
    )
    outputs_dir = resolve_project_path(
        project_root,
        branch_config["paths"]["outputs_dir"],
    )

    for path in (reports_dir, docs_dir, outputs_dir):
        path.mkdir(parents=True, exist_ok=True)

    validation_folds = build_validation_folds(modeling_config)

    target_columns = [
        f"{model_cfg['target_prefix']}{horizon:02d}"
        for horizon in range(1, horizons + 1)
    ]

    fold_metric_parts = []
    fsa_metric_parts = []
    horizon_metric_parts = []
    availability_parts = []
    pooled_actual = []
    pooled_predicted = []

    for fold in validation_folds:
        _, evaluation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=horizons,
        )

        evaluation_targets = evaluation_targets.reset_index(drop=True)

        predictions = build_daily_seasonal_naive_predictions(
            evaluation_targets[["fsa", "forecast_origin"]],
            feature_dataset,
            horizons=horizons,
            seasonal_period_hours=int(model_cfg["seasonal_period_hours"]),
            target_prefix=model_cfg["target_prefix"],
        )

        availability = validate_daily_seasonal_naive_availability(
            predictions,
            target_prefix=model_cfg["target_prefix"],
        )
        availability.insert(0, "fold", fold.name)
        availability_parts.append(availability)

        failures = availability.loc[availability["status"] == "FAIL"]
        if not failures.empty:
            raise ValueError(
                f"Seasonal Naive prediction coverage failed for {fold.name}."
            )

        actual = evaluation_targets[
            ["fsa", "forecast_origin", *target_columns]
        ].copy()

        predicted_targets = predictions[target_columns].copy()

        fold_metrics = evaluate_forecasting_global(
            actual,
            predicted_targets,
            target_prefix=model_cfg["target_prefix"],
        )
        fold_metrics.insert(0, "fold", fold.name)
        fold_metric_parts.append(fold_metrics)

        by_fsa = evaluate_forecasting_by_group(
            actual,
            predicted_targets,
            group_columns=["fsa"],
            target_prefix=model_cfg["target_prefix"],
        )
        by_fsa.insert(0, "fold", fold.name)
        fsa_metric_parts.append(by_fsa)

        by_horizon = evaluate_forecasting_by_horizon(
            actual,
            predicted_targets,
            target_prefix=model_cfg["target_prefix"],
        )
        by_horizon.insert(0, "fold", fold.name)
        horizon_metric_parts.append(by_horizon)

        pooled_actual.append(actual[target_columns])
        pooled_predicted.append(predicted_targets)

    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    fsa_metrics = pd.concat(fsa_metric_parts, ignore_index=True)
    horizon_metrics = pd.concat(horizon_metric_parts, ignore_index=True)
    availability = pd.concat(availability_parts, ignore_index=True)

    pooled_actual_frame = pd.concat(pooled_actual, ignore_index=True)
    pooled_predicted_frame = pd.concat(pooled_predicted, ignore_index=True)

    global_metrics = evaluate_forecasting_global(
        pooled_actual_frame,
        pooled_predicted_frame,
        target_prefix=model_cfg["target_prefix"],
    )
    global_metrics.insert(0, "model", model_cfg["name"])

    save_csv(fold_metrics, reports_dir / "SNB_01_fold_metrics.csv")
    save_csv(fsa_metrics, reports_dir / "SNB_02_fsa_metrics.csv")
    save_csv(horizon_metrics, reports_dir / "SNB_03_horizon_metrics.csv")
    save_csv(global_metrics, reports_dir / "SNB_04_global_metrics.csv")
    save_csv(availability, reports_dir / "SNB_05_prediction_coverage.csv")

    _write_summary(
        docs_dir,
        fold_metrics,
        global_metrics,
        availability,
    )

    print("Seasonal Naive Forecasting baseline completed successfully.")
    print("Development folds evaluated:", [fold.name for fold in validation_folds])
    print("Final 2025 holdout evaluated: NO")
    print(f"Reports: {reports_dir.resolve()}")

    return {
        "fold_metrics": fold_metrics,
        "fsa_metrics": fsa_metrics,
        "horizon_metrics": horizon_metrics,
        "global_metrics": global_metrics,
        "prediction_coverage": availability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Daily Seasonal Naive Forecasting baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/seasonal_naive_baseline.yaml",
    )
    args = parser.parse_args()
    run_seasonal_naive_baseline(args.config)


if __name__ == "__main__":
    main()
