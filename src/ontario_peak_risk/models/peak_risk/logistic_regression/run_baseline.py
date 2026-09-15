"""Run the Logistic Regression Peak-Risk baseline on development folds only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_modeling_config,
    resolve_project_path,
    save_csv,
)
from src.ontario_peak_risk.modeling.evaluation import (
    peak_risk_metrics,
    probabilities_to_binary_predictions,
)
from src.ontario_peak_risk.modeling.peak_policy import (
    build_peak_horizon_targets,
    fit_peak_thresholds,
    label_peak_evaluation_data,
)
from src.ontario_peak_risk.modeling.preprocessing import (
    build_sklearn_preprocessor,
    get_preprocessing_profile,
)
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_forecast_origin_fold,
    slice_observation_fold,
)

from .features import (
    build_logistic_horizon_features,
    logistic_feature_columns,
)


def _load_branch_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config, path.parent.parent


def _evaluate_group(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Evaluate Peak-Risk metrics by one or more grouping columns."""
    rows = []

    for key, group in frame.groupby(group_columns, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        result = peak_risk_metrics(
            group["actual"],
            group["predicted"],
            group["probability"],
        )

        rows.append(
            {
                **dict(zip(group_columns, key)),
                **result,
            }
        )

    return pd.DataFrame(rows)


def _write_summary(
    docs_dir: Path,
    fold_metrics: pd.DataFrame,
    global_metrics: pd.DataFrame,
    class_balance: pd.DataFrame,
) -> None:
    try:
        fold_table = fold_metrics.to_markdown(index=False)
        global_table = global_metrics.to_markdown(index=False)
        balance_table = class_balance.to_markdown(index=False)
    except ImportError:
        fold_table = fold_metrics.to_string(index=False)
        global_table = global_metrics.to_string(index=False)
        balance_table = class_balance.to_string(index=False)

    content = f"""# Logistic Regression Peak-Risk Baseline

## Baseline definition

The baseline fits one Logistic Regression classifier for each forecast horizon
from `peak_h01` through `peak_h24`.

The model uses a deliberately small, interpretable predictor set:

- FSA;
- target season;
- cyclic target-hour, weekday, and month representations;
- weekend status;
- same target hour's electricity demand 24 hours earlier;
- same target hour's electricity demand 168 hours earlier.

No observed future electricity demand is used as a predictor.

## Preprocessing

Numeric features use the shared Logistic Regression preprocessing profile:
training-only imputation and standardization.

Categorical variables use training-only imputation and one-hot encoding.

No hyperparameter search is performed. This is a fixed baseline configuration.

## Peak target policy

Peak thresholds are fitted independently inside each training fold using the
shared 97.5th-percentile FSA + season policy.

The frozen training thresholds are then applied to validation observations.

## Development policy

Only the 2023 and 2024 validation folds are evaluated.

The final 2025 holdout remains untouched.

## Fold metrics

{fold_table}

## Pooled development metrics

{global_table}

## Class balance

{balance_table}

## Interpretation

This model is the classification reference benchmark. Candidate Peak-Risk
models must demonstrate improvement under the same folds, target methodology,
probability threshold, and official evaluation metrics.
"""

    docs_dir.mkdir(parents=True, exist_ok=True)
    (
        docs_dir
        / "LRB_06_Logistic_Regression_Peak_Baseline_Summary.md"
    ).write_text(
        content,
        encoding="utf-8",
    )


def run_logistic_regression_peak_baseline(
    config_path: str | Path = "configs/logistic_regression_peak_baseline.yaml",
) -> dict[str, pd.DataFrame]:
    """Execute the Peak-Risk Logistic Regression baseline."""
    branch_config, project_root = _load_branch_config(config_path)

    foundation_config_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(foundation_config_path)

    feature_dataset = load_feature_dataset(modeling_config, project_root)

    model_cfg = branch_config["model"]
    peak_cfg = modeling_config["modeling"]["peak_risk"]

    horizons = int(model_cfg["horizons"])
    probability_threshold = float(model_cfg["probability_threshold"])

    if bool(model_cfg.get("evaluate_final_holdout", False)):
        raise ValueError(
            "The Logistic Regression baseline branch must not evaluate "
            "the final 2025 holdout."
        )

    if probability_threshold != float(peak_cfg["probability_threshold"]):
        raise ValueError(
            "Baseline probability threshold must match Modeling Foundation."
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

    numeric_columns, categorical_columns = logistic_feature_columns()
    profile = get_preprocessing_profile("logistic_regression")

    logistic_cfg = model_cfg["logistic_regression"]

    fold_metric_parts = []
    horizon_metric_parts = []
    fsa_metric_parts = []
    class_balance_parts = []
    all_validation_predictions = []

    for fold in validation_folds:
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
            horizons=horizons,
            target_prefix=model_cfg["target_prefix"],
            require_complete_horizon=True,
        )
        validation_targets = build_peak_horizon_targets(
            labeled_validation,
            horizons=horizons,
            target_prefix=model_cfg["target_prefix"],
            require_complete_horizon=True,
        )

        safe_training_targets, _ = slice_forecast_origin_fold(
            training_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=horizons,
        )
        _, safe_validation_targets = slice_forecast_origin_fold(
            validation_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=horizons,
        )

        safe_training_targets = safe_training_targets.reset_index(drop=True)
        safe_validation_targets = safe_validation_targets.reset_index(drop=True)

        fold_prediction_parts = []

        for horizon in range(1, horizons + 1):
            target_column = f"{model_cfg['target_prefix']}{horizon:02d}"

            train_features = build_logistic_horizon_features(
                safe_training_targets[["fsa", "forecast_origin"]],
                feature_dataset,
                horizon=horizon,
            )

            validation_features = build_logistic_horizon_features(
                safe_validation_targets[["fsa", "forecast_origin"]],
                feature_dataset,
                horizon=horizon,
            )

            preprocessor = build_sklearn_preprocessor(
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                profile=profile,
            )

            classifier = LogisticRegression(
                C=float(logistic_cfg["C"]),
                max_iter=int(logistic_cfg["max_iter"]),
                solver=str(logistic_cfg["solver"]),
                class_weight=logistic_cfg["class_weight"],
            )

            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", classifier),
                ]
            )

            # Split data in train and test
            X_train = train_features[numeric_columns + categorical_columns]
            y_train = safe_training_targets[target_column].astype("int8")

            X_validation = validation_features[numeric_columns + categorical_columns]
            y_validation = safe_validation_targets[target_column].astype("int8")

            if y_train.nunique() < 2:
                raise ValueError(
                    f"{fold.name} {target_column} contains only one training class."
                )

            # Run model
            pipeline.fit(X_train, y_train)

            probability = pipeline.predict_proba(X_validation)[:, 1]
            predicted = probabilities_to_binary_predictions(
                probability,
                threshold=probability_threshold,
            )

            horizon_predictions = pd.DataFrame(
                {
                    "fold": fold.name,
                    "fsa": safe_validation_targets["fsa"].to_numpy(),
                    "forecast_origin": safe_validation_targets[
                        "forecast_origin"
                    ].to_numpy(),
                    "horizon": horizon,
                    "actual": y_validation.to_numpy(),
                    "predicted": predicted,
                    "probability": probability,
                }
            )

            fold_prediction_parts.append(horizon_predictions)

            horizon_metrics = peak_risk_metrics(
                y_validation,
                predicted,
                probability,
            )
            horizon_metric_parts.append(
                pd.DataFrame(
                    [
                        {
                            "fold": fold.name,
                            "horizon": horizon,
                            **horizon_metrics,
                        }
                    ]
                )
            )

        fold_predictions = pd.concat(
            fold_prediction_parts,
            ignore_index=True,
        )

        all_validation_predictions.append(fold_predictions)

        fold_metrics = peak_risk_metrics(
            fold_predictions["actual"],
            fold_predictions["predicted"],
            fold_predictions["probability"],
        )
        fold_metric_parts.append(
            pd.DataFrame([{"fold": fold.name, **fold_metrics}])
        )

        fsa_metrics = _evaluate_group(
            fold_predictions,
            ["fold", "fsa"],
        )
        fsa_metric_parts.append(fsa_metrics)

        class_balance = (
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
        class_balance["peak_rate_pct"] = (
            class_balance["peak_hours"]
            / class_balance["observations"]
            * 100
        )
        class_balance_parts.append(class_balance)

    predictions = pd.concat(
        all_validation_predictions,
        ignore_index=True,
    )

    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    horizon_metrics = pd.concat(horizon_metric_parts, ignore_index=True)
    fsa_metrics = pd.concat(fsa_metric_parts, ignore_index=True)
    class_balance = pd.concat(class_balance_parts, ignore_index=True)

    global_result = peak_risk_metrics(
        predictions["actual"],
        predictions["predicted"],
        predictions["probability"],
    )
    global_metrics = pd.DataFrame(
        [{"model": model_cfg["name"], **global_result}]
    )

    save_csv(fold_metrics, reports_dir / "LRB_01_fold_metrics.csv")
    save_csv(fsa_metrics, reports_dir / "LRB_02_fsa_metrics.csv")
    save_csv(horizon_metrics, reports_dir / "LRB_03_horizon_metrics.csv")
    save_csv(global_metrics, reports_dir / "LRB_04_global_metrics.csv")
    save_csv(class_balance, reports_dir / "LRB_05_class_balance.csv")

    # Full predictions are intentionally stored outside reports and should
    # normally remain excluded from Git.
    predictions.to_parquet(
        outputs_dir / "logistic_regression_validation_predictions.parquet",
        index=False,
    )

    _write_summary(
        docs_dir,
        fold_metrics,
        global_metrics,
        class_balance,
    )

    print("Logistic Regression Peak-Risk baseline completed successfully.")
    print("Development folds evaluated:", [fold.name for fold in validation_folds])
    print("Final 2025 holdout evaluated: NO")
    print(f"Reports: {reports_dir.resolve()}")

    return {
        "fold_metrics": fold_metrics,
        "fsa_metrics": fsa_metrics,
        "horizon_metrics": horizon_metrics,
        "global_metrics": global_metrics,
        "class_balance": class_balance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Logistic Regression Peak-Risk baseline."
    )
    parser.add_argument(
        "--config",
        default="configs/logistic_regression_peak_baseline.yaml",
    )
    args = parser.parse_args()
    run_logistic_regression_peak_baseline(args.config)


if __name__ == "__main__":
    main()
