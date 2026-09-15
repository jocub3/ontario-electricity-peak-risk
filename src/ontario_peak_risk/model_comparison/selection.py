"""Evidence-based candidate selection.

Primary metrics establish the ranking. Guardrails are presented as evidence
and are not collapsed into an arbitrary weighted score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecasting_selection_matrix(
    global_table: pd.DataFrame,
    stability: pd.DataFrame,
    m9w: pd.DataFrame,
    runtime: pd.DataFrame,
) -> pd.DataFrame:
    matrix = global_table.merge(
        stability,
        on="model",
        how="left",
    ).merge(
        m9w.rename(
            columns={
                "mae": "m9w_2023_mae",
                "mape": "m9w_2023_mape",
                "bias": "m9w_2023_bias",
            }
        )[
            [
                c for c in [
                    "model",
                    "m9w_2023_mae",
                    "m9w_2023_mape",
                    "m9w_2023_bias",
                ]
                if c in m9w.rename(
                    columns={
                        "mae": "m9w_2023_mae",
                        "mape": "m9w_2023_mape",
                        "bias": "m9w_2023_bias",
                    }
                ).columns
            ]
        ],
        on="model",
        how="left",
    ).merge(
        runtime,
        on=["model", "family"],
        how="left",
    )

    matrix["primary_rank_mae"] = matrix["mae"].rank(
        method="min",
        ascending=True,
    )

    baseline = matrix.loc[
        matrix["model"].str.contains(
            "Seasonal Naive",
            case=False,
            regex=False,
        )
    ]

    if not baseline.empty:
        baseline_mae = float(baseline.iloc[0]["mae"])
        matrix["mae_improvement_vs_baseline_pct"] = (
            (baseline_mae - matrix["mae"])
            / baseline_mae
            * 100
        )
    else:
        matrix["mae_improvement_vs_baseline_pct"] = np.nan

    return matrix.sort_values(
        ["primary_rank_mae", "rmse", "mape"],
        kind="stable",
    ).reset_index(drop=True)


def peak_selection_matrix(
    global_table: pd.DataFrame,
    stability: pd.DataFrame,
    m9w: pd.DataFrame,
    runtime: pd.DataFrame,
) -> pd.DataFrame:
    renamed = m9w.rename(
        columns={
            "precision": "m9w_2023_precision",
            "recall": "m9w_2023_recall",
            "f1": "m9w_2023_f1",
            "pr_auc": "m9w_2023_pr_auc",
            "roc_auc": "m9w_2023_roc_auc",
            "brier_score": "m9w_2023_brier_score",
        }
    )

    stress_columns = [
        c for c in [
            "model",
            "m9w_2023_precision",
            "m9w_2023_recall",
            "m9w_2023_f1",
            "m9w_2023_pr_auc",
            "m9w_2023_roc_auc",
            "m9w_2023_brier_score",
        ]
        if c in renamed.columns
    ]

    matrix = global_table.merge(
        stability,
        on="model",
        how="left",
    ).merge(
        renamed[stress_columns],
        on="model",
        how="left",
    ).merge(
        runtime,
        on=["model", "family"],
        how="left",
    )

    matrix["primary_rank_pr_auc"] = matrix["pr_auc"].rank(
        method="min",
        ascending=False,
    )

    return matrix.sort_values(
        [
            "primary_rank_pr_auc",
            "f1",
            "recall",
        ],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def selection_recommendation(
    forecasting_matrix: pd.DataFrame,
    peak_matrix: pd.DataFrame,
) -> dict:
    recommendation = {
        "forecasting_primary_candidate": None,
        "peak_risk_primary_candidate": None,
        "notes": [],
    }

    if not forecasting_matrix.empty:
        recommendation["forecasting_primary_candidate"] = (
            forecasting_matrix.iloc[0]["model"]
        )

    if not peak_matrix.empty:
        recommendation["peak_risk_primary_candidate"] = (
            peak_matrix.iloc[0]["model"]
        )

    recommendation["notes"].append(
        "Primary candidates are selected by the pre-declared primary metric "
        "(Forecasting: MAE; Peak-Risk: PR-AUC)."
    )
    recommendation["notes"].append(
        "Guardrails must be reviewed before the selection is considered final."
    )
    recommendation["notes"].append(
        "The 2025 holdout remains untouched during this phase."
    )

    return recommendation
