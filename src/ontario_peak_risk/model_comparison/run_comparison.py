"""Complete Model Comparison & Selection phase runner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import (
    load_config,
    load_model_reports,
    resolve_project_path,
    validate_loaded_reports,
)
from .metrics import (
    combine_group_metrics,
    forecasting_global_table,
    forecasting_stability,
    m9w_forecasting_stress,
    m9w_peak_stress,
    peak_global_table,
    peak_stability,
)
from .plotting import (
    calibration_comparison_plot,
    forecasting_global_metrics_plot,
    forecasting_percentage_metrics_plot,
    fsa_heatmap,
    metric_by_fold_plot,
    metric_by_horizon_plot,
    m9w_bar_plot,
    peak_global_plot,
    runtime_plot,
)
from .runtime import runtime_table
from .selection import (
    forecasting_selection_matrix,
    peak_selection_matrix,
    selection_recommendation,
)


def _save(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_model_comparison(
    config_path: str | Path = "configs/model_comparison.yaml",
) -> dict[str, object]:
    config, project_root = load_config(config_path)

    if bool(config["phase"].get("evaluate_final_holdout", False)):
        raise ValueError(
            "Model Comparison & Selection must not evaluate the 2025 holdout."
        )

    output_reports = resolve_project_path(
        project_root,
        config["paths"]["output_reports_dir"],
    )
    output_figures = resolve_project_path(
        project_root,
        config["paths"]["output_figures_dir"],
    )
    output_data = resolve_project_path(
        project_root,
        config["paths"]["output_data_dir"],
    )

    for path in [
        output_reports,
        output_figures,
        output_data,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    forecasting_reports = load_model_reports(
        config,
        project_root,
        "forecasting",
    )
    peak_reports = load_model_reports(
        config,
        project_root,
        "peak_risk",
    )

    validation = pd.concat(
        [
            validate_loaded_reports(forecasting_reports),
            validate_loaded_reports(peak_reports),
        ],
        ignore_index=True,
    )
    _save(validation, output_reports / "MC_01_input_validation.csv")

    missing_core = validation.loc[
        ~(
            validation["global_metrics"]
            & validation["fold_metrics"]
            & validation["fsa_metrics"]
            & validation["horizon_metrics"]
        )
    ]

    if not missing_core.empty:
        print(
            "WARNING: Some models are missing one or more core reports. "
            "Review MC_01_input_validation.csv before final selection."
        )
        print(
            missing_core[
                [
                    "model",
                    "task",
                    "report_dir",
                    "global_metrics",
                    "fold_metrics",
                    "fsa_metrics",
                    "horizon_metrics",
                ]
            ].to_string(index=False)
        )

    # Forecasting
    fc_global = forecasting_global_table(
        forecasting_reports
    )
    fc_fold = combine_group_metrics(
        forecasting_reports,
        "fold_metrics",
    )
    fc_fsa = combine_group_metrics(
        forecasting_reports,
        "fsa_metrics",
    )
    fc_horizon = combine_group_metrics(
        forecasting_reports,
        "horizon_metrics",
    )
    fc_stability = forecasting_stability(
        fc_fold,
        fc_horizon,
    )
    fc_m9w = m9w_forecasting_stress(fc_fsa)

    # Peak-Risk
    pk_global = peak_global_table(peak_reports)
    pk_fold = combine_group_metrics(
        peak_reports,
        "fold_metrics",
    )
    pk_fsa = combine_group_metrics(
        peak_reports,
        "fsa_metrics",
    )
    pk_horizon = combine_group_metrics(
        peak_reports,
        "horizon_metrics",
    )
    pk_stability = peak_stability(
        pk_fold,
        pk_horizon,
    )
    pk_m9w = m9w_peak_stress(pk_fsa)

    manual_runtime = config.get(
        "runtime", {}
    ).get("manual_overrides", {})

    fc_runtime = runtime_table(
        forecasting_reports,
        manual_overrides=manual_runtime.get("forecasting", {}),
    )
    pk_runtime = runtime_table(
        peak_reports,
        manual_overrides=manual_runtime.get("peak_risk", {}),
    )

    fc_selection = forecasting_selection_matrix(
        fc_global,
        fc_stability,
        fc_m9w,
        fc_runtime,
    )
    pk_selection = peak_selection_matrix(
        pk_global,
        pk_stability,
        pk_m9w,
        pk_runtime,
    )

    recommendation = selection_recommendation(
        fc_selection,
        pk_selection,
    )

    tables = {
        "MC_02_forecasting_global_comparison.csv": fc_global,
        "MC_02_forecasting_fold_comparison.csv": fc_fold,
        "MC_02_forecasting_fsa_comparison.csv": fc_fsa,
        "MC_02_forecasting_horizon_comparison.csv": fc_horizon,
        "MC_03_forecasting_stability.csv": fc_stability,
        "MC_03_forecasting_m9w_2023.csv": fc_m9w,
        "MC_04_peak_risk_global_comparison.csv": pk_global,
        "MC_04_peak_risk_fold_comparison.csv": pk_fold,
        "MC_04_peak_risk_fsa_comparison.csv": pk_fsa,
        "MC_04_peak_risk_horizon_comparison.csv": pk_horizon,
        "MC_05_peak_risk_stability.csv": pk_stability,
        "MC_05_peak_risk_m9w_2023.csv": pk_m9w,
        "MC_06_forecasting_runtime_evidence.csv": fc_runtime,
        "MC_06_peak_risk_runtime_evidence.csv": pk_runtime,
        "MC_07_forecasting_selection_matrix.csv": fc_selection,
        "MC_07_peak_risk_selection_matrix.csv": pk_selection,
    }

    for filename, frame in tables.items():
        _save(frame, output_reports / filename)

    (
        output_data / "MC_07_selection_recommendation.json"
    ).write_text(
        json.dumps(
            recommendation,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Figures
    forecasting_global_metrics_plot(
        fc_global,
        output_figures,
    )
    forecasting_percentage_metrics_plot(
        fc_global,
        output_figures,
    )
    metric_by_fold_plot(
        fc_fold,
        "mae",
        "Forecasting — MAE by Validation Fold",
        output_figures,
        "forecasting_mae_by_fold.png",
    )
    metric_by_horizon_plot(
        fc_horizon,
        "mae",
        "Forecasting — MAE by Forecast Horizon",
        output_figures,
        "forecasting_mae_by_horizon.png",
    )
    fsa_heatmap(
        fc_fsa,
        "mae",
        "fold_2023",
        "Forecasting — MAE by FSA (2023)",
        output_figures,
        "forecasting_fsa_mae_2023.png",
    )
    fsa_heatmap(
        fc_fsa,
        "mae",
        "fold_2024",
        "Forecasting — MAE by FSA (2024)",
        output_figures,
        "forecasting_fsa_mae_2024.png",
    )
    if not fc_m9w.empty:
        m9w_bar_plot(
            fc_m9w,
            "mae",
            "Forecasting — M9W-2023 Stress Test",
            output_figures,
            "forecasting_m9w_2023_mae.png",
        )

    peak_global_plot(
        pk_global,
        output_figures,
    )
    calibration_comparison_plot(
        pk_global,
        output_figures,
    )
    metric_by_fold_plot(
        pk_fold,
        "pr_auc",
        "Peak-Risk — PR-AUC by Validation Fold",
        output_figures,
        "peak_risk_pr_auc_by_fold.png",
    )
    metric_by_fold_plot(
        pk_fold,
        "f1",
        "Peak-Risk — F1 by Validation Fold",
        output_figures,
        "peak_risk_f1_by_fold.png",
    )
    metric_by_horizon_plot(
        pk_horizon,
        "pr_auc",
        "Peak-Risk — PR-AUC by Forecast Horizon",
        output_figures,
        "peak_risk_pr_auc_by_horizon.png",
    )
    metric_by_horizon_plot(
        pk_horizon,
        "recall",
        "Peak-Risk — Recall by Forecast Horizon",
        output_figures,
        "peak_risk_recall_by_horizon.png",
    )
    fsa_heatmap(
        pk_fsa,
        "pr_auc",
        "fold_2023",
        "Peak-Risk — PR-AUC by FSA (2023)",
        output_figures,
        "peak_risk_fsa_pr_auc_2023.png",
    )
    fsa_heatmap(
        pk_fsa,
        "pr_auc",
        "fold_2024",
        "Peak-Risk — PR-AUC by FSA (2024)",
        output_figures,
        "peak_risk_fsa_pr_auc_2024.png",
    )
    if not pk_m9w.empty:
        m9w_bar_plot(
            pk_m9w,
            "pr_auc",
            "Peak-Risk — M9W-2023 PR-AUC Stress Test",
            output_figures,
            "peak_risk_m9w_2023_pr_auc.png",
        )
        m9w_bar_plot(
            pk_m9w,
            "recall",
            "Peak-Risk — M9W-2023 Recall Stress Test",
            output_figures,
            "peak_risk_m9w_2023_recall.png",
        )

    runtime_plot(
        fc_runtime,
        "Forecasting — Available Runtime Evidence",
        output_figures,
        "forecasting_runtime_evidence.png",
    )
    runtime_plot(
        pk_runtime,
        "Peak-Risk — Available Runtime Evidence",
        output_figures,
        "peak_risk_runtime_evidence.png",
    )

    print("=" * 72)
    print("MODEL COMPARISON & SELECTION COMPLETED")
    print("2025 final holdout evaluated: NO")
    print(
        "Forecasting primary candidate:",
        recommendation["forecasting_primary_candidate"],
    )
    print(
        "Peak-Risk primary candidate:",
        recommendation["peak_risk_primary_candidate"],
    )
    print(
        "IMPORTANT: Review guardrails and stress tests before approving "
        "the final candidate decision."
    )
    print("=" * 72)

    return {
        "input_validation": validation,
        "forecasting_global": fc_global,
        "forecasting_fold": fc_fold,
        "forecasting_fsa": fc_fsa,
        "forecasting_horizon": fc_horizon,
        "forecasting_stability": fc_stability,
        "forecasting_m9w": fc_m9w,
        "forecasting_runtime": fc_runtime,
        "forecasting_selection": fc_selection,
        "peak_global": pk_global,
        "peak_fold": pk_fold,
        "peak_fsa": pk_fsa,
        "peak_horizon": pk_horizon,
        "peak_stability": pk_stability,
        "peak_m9w": pk_m9w,
        "peak_runtime": pk_runtime,
        "peak_selection": pk_selection,
        "recommendation": recommendation,
    }
