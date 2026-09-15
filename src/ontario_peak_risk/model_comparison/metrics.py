"""Standardized comparison metrics and stability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import ModelReports


def _first_row(frame: pd.DataFrame | None) -> dict:
    if frame is None or frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _normalize_fold(value) -> str:
    text = str(value)
    if text.startswith("fold_"):
        return text
    if text in {"2023", "2024"}:
        return f"fold_{text}"
    return text


def forecasting_global_table(
    reports: dict[str, ModelReports],
) -> pd.DataFrame:
    rows = []
    for model in reports.values():
        row = _first_row(model.global_metrics)
        if not row:
            continue

        bias = row.get("bias", np.nan)
        rows.append(
            {
                "model_key": model.key,
                "model": model.display_name,
                "family": model.family,
                "mae": row.get("mae", np.nan),
                "rmse": row.get("rmse", np.nan),
                "mape": row.get("mape", np.nan),
                "wape": row.get("wape", np.nan),
                "bias": bias,
                "abs_bias": abs(bias) if pd.notna(bias) else np.nan,
                "n_observations": row.get(
                    "n_observations",
                    row.get(
                        "validation_predictions",
                        row.get("observations", np.nan),
                    ),
                ),
            }
        )
    return pd.DataFrame(rows)


def peak_global_table(
    reports: dict[str, ModelReports],
) -> pd.DataFrame:
    rows = []
    for model in reports.values():
        row = _first_row(model.global_metrics)
        if not row:
            continue

        rows.append(
            {
                "model_key": model.key,
                "model": model.display_name,
                "family": model.family,
                "precision": row.get("precision", np.nan),
                "recall": row.get("recall", np.nan),
                "f1": row.get("f1", np.nan),
                "pr_auc": row.get("pr_auc", np.nan),
                "roc_auc": row.get("roc_auc", np.nan),
                "balanced_accuracy": row.get(
                    "balanced_accuracy", np.nan
                ),
                "brier_score": row.get("brier_score", np.nan),
                "positive_rate_pct": row.get(
                    "positive_rate_pct",
                    row.get("actual_peak_rate_pct", np.nan),
                ),
                "predicted_positive_rate_pct": row.get(
                    "predicted_positive_rate_pct", np.nan
                ),
                "n_observations": row.get(
                    "n_observations",
                    row.get(
                        "validation_predictions",
                        row.get("observations", np.nan),
                    ),
                ),
            }
        )
    return pd.DataFrame(rows)


def combine_group_metrics(
    reports: dict[str, ModelReports],
    attribute: str,
) -> pd.DataFrame:
    frames = []

    for model in reports.values():
        frame = getattr(model, attribute)
        if frame is None or frame.empty:
            continue

        work = frame.copy()
        work["model_key"] = model.key
        work["model"] = model.display_name
        work["family"] = model.family

        if "fold" in work.columns:
            work["fold"] = work["fold"].map(_normalize_fold)

        frames.append(work)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def forecasting_stability(
    fold_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, fold_group in fold_metrics.groupby(
        "model", observed=True
    ):
        mae = fold_group["mae"].astype(float)

        horizon_group = horizon_metrics[
            horizon_metrics["model"].eq(model)
        ].copy()

        if horizon_group.empty:
            slope = np.nan
            short_mae = np.nan
            long_mae = np.nan
        else:
            x = horizon_group["horizon"].astype(float).to_numpy()
            y = horizon_group["mae"].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            slope = (
                float(np.polyfit(x[mask], y[mask], 1)[0])
                if mask.sum() >= 2
                else np.nan
            )
            short_mae = horizon_group.loc[
                horizon_group["horizon"].between(1, 6),
                "mae",
            ].mean()
            long_mae = horizon_group.loc[
                horizon_group["horizon"].between(19, 24),
                "mae",
            ].mean()

        rows.append(
            {
                "model": model,
                "fold_mae_mean": mae.mean(),
                "fold_mae_std": mae.std(ddof=0),
                "fold_mae_cv": (
                    mae.std(ddof=0) / mae.mean()
                    if mae.mean() != 0
                    else np.nan
                ),
                "fold_mae_range": mae.max() - mae.min(),
                "horizon_mae_slope": slope,
                "short_horizon_mae_h01_h06": short_mae,
                "long_horizon_mae_h19_h24": long_mae,
            }
        )

    return pd.DataFrame(rows)


def peak_stability(
    fold_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, fold_group in fold_metrics.groupby(
        "model", observed=True
    ):
        pr = fold_group["pr_auc"].astype(float)

        horizon_group = horizon_metrics[
            horizon_metrics["model"].eq(model)
        ].copy()

        if horizon_group.empty:
            slope = np.nan
            short_pr = np.nan
            long_pr = np.nan
        else:
            x = horizon_group["horizon"].astype(float).to_numpy()
            y = horizon_group["pr_auc"].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            slope = (
                float(np.polyfit(x[mask], y[mask], 1)[0])
                if mask.sum() >= 2
                else np.nan
            )
            short_pr = horizon_group.loc[
                horizon_group["horizon"].between(1, 6),
                "pr_auc",
            ].mean()
            long_pr = horizon_group.loc[
                horizon_group["horizon"].between(19, 24),
                "pr_auc",
            ].mean()

        rows.append(
            {
                "model": model,
                "fold_pr_auc_mean": pr.mean(),
                "fold_pr_auc_std": pr.std(ddof=0),
                "fold_pr_auc_range": pr.max() - pr.min(),
                "horizon_pr_auc_slope": slope,
                "short_horizon_pr_auc_h01_h06": short_pr,
                "long_horizon_pr_auc_h19_h24": long_pr,
            }
        )

    return pd.DataFrame(rows)


def m9w_forecasting_stress(
    fsa_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if fsa_metrics.empty:
        return pd.DataFrame()

    target = fsa_metrics[
        fsa_metrics["fsa"].astype(str).eq("M9W")
        & fsa_metrics["fold"].astype(str).eq("fold_2023")
    ].copy()

    metric_columns = [
        col
        for col in [
            "mae",
            "rmse",
            "mape",
            "wape",
            "bias",
        ]
        if col in target.columns
    ]

    if target.empty:
        return pd.DataFrame()

    result = (
        target
        .groupby(
            ["model", "family"],
            observed=True,
            as_index=False,
        )[metric_columns]
        .mean()
    )

    return result


def m9w_peak_stress(
    fsa_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if fsa_metrics.empty:
        return pd.DataFrame()

    target = fsa_metrics[
        fsa_metrics["fsa"].astype(str).eq("M9W")
        & fsa_metrics["fold"].astype(str).eq("fold_2023")
    ].copy()

    metric_columns = [
        col
        for col in [
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "brier_score",
            "positive_rate_pct",
        ]
        if col in target.columns
    ]

    if target.empty:
        return pd.DataFrame()

    result = (
        target
        .groupby(
            ["model", "family"],
            observed=True,
            as_index=False,
        )[metric_columns]
        .mean()
    )

    return result


def rank_table(
    table: pd.DataFrame,
    metrics_direction: dict[str, str],
) -> pd.DataFrame:
    """Create metric-specific ranks. No opaque weighted composite is used."""
    result = table.copy()

    for metric, direction in metrics_direction.items():
        if metric not in result.columns:
            continue

        result[f"rank_{metric}"] = result[metric].rank(
            method="min",
            ascending=(direction == "lower"),
            na_option="bottom",
        )

    rank_cols = [
        col for col in result.columns if col.startswith("rank_")
    ]

    if rank_cols:
        result["mean_metric_rank"] = result[rank_cols].mean(
            axis=1
        )

    return result
