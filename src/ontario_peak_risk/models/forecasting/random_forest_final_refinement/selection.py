"""Selection logic for the small Random Forest forecasting refinement."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_refinement(
    details: pd.DataFrame,
) -> pd.DataFrame:
    required = {"refinement_id", "mae"}
    missing = sorted(required.difference(details.columns))
    if missing:
        raise ValueError(
            f"Refinement results missing required columns: {missing}"
        )

    aggregations = {
        "mean_mae": ("mae", "mean"),
        "std_mae": ("mae", "std"),
    }

    if "rmse" in details.columns:
        aggregations["mean_rmse"] = ("rmse", "mean")
    if "mape" in details.columns:
        aggregations["mean_mape"] = ("mape", "mean")

    summary = (
        details
        .groupby("refinement_id", observed=True)
        .agg(**aggregations)
        .reset_index()
    )

    if "fold" in details.columns:
        fold = (
            details
            .groupby(["refinement_id", "fold"], observed=True)["mae"]
            .mean()
            .reset_index()
        )

        fold_range = (
            fold
            .groupby("refinement_id", observed=True)["mae"]
            .agg(lambda s: s.max() - s.min())
            .rename("fold_mae_range")
            .reset_index()
        )

        summary = summary.merge(
            fold_range,
            on="refinement_id",
            how="left",
        )

    if "horizon" in details.columns:
        horizon = (
            details
            .groupby(["refinement_id", "horizon"], observed=True)["mae"]
            .mean()
            .reset_index()
        )

        horizon_range = (
            horizon
            .groupby("refinement_id", observed=True)["mae"]
            .agg(lambda s: s.max() - s.min())
            .rename("horizon_mae_range")
            .reset_index()
        )

        summary = summary.merge(
            horizon_range,
            on="refinement_id",
            how="left",
        )

    return summary.sort_values(
        ["mean_mae", "std_mae"],
        kind="stable",
    ).reset_index(drop=True)


def select_final_configuration(
    summary: pd.DataFrame,
    *,
    incumbent_id: str,
    minimum_improvement_pct: float,
) -> tuple[pd.Series, pd.DataFrame]:
    if incumbent_id not in set(summary["refinement_id"]):
        raise ValueError(
            f"Incumbent {incumbent_id!r} is missing from refinement results."
        )

    incumbent = summary.loc[
        summary["refinement_id"].eq(incumbent_id)
    ].iloc[0]

    incumbent_mae = float(incumbent["mean_mae"])

    ranked = summary.copy()
    ranked["mae_improvement_vs_incumbent_pct"] = (
        (incumbent_mae - ranked["mean_mae"])
        / incumbent_mae
        * 100
    )

    best = ranked.sort_values(
        ["mean_mae", "std_mae"],
        kind="stable",
    ).iloc[0]

    if (
        best["refinement_id"] != incumbent_id
        and float(best["mae_improvement_vs_incumbent_pct"])
        >= float(minimum_improvement_pct)
    ):
        selected_id = str(best["refinement_id"])
        decision = (
            "REPLACE_INCUMBENT: local refinement produced a material "
            "MAE improvement."
        )
    else:
        selected_id = incumbent_id
        decision = (
            "KEEP_INCUMBENT: no local alternative exceeded the configured "
            "minimum MAE improvement required to justify changing the "
            "already-selected forecasting model."
        )

    ranked["selected"] = ranked["refinement_id"].eq(selected_id)
    ranked["decision"] = decision

    selected = ranked.loc[
        ranked["refinement_id"].eq(selected_id)
    ].iloc[0]

    return selected, ranked
