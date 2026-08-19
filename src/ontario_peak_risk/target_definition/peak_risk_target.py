"""Leakage-safe Peak-Risk target construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PeakRiskThresholdEstimator:
    """
    Fit FSA + seasonal Peak thresholds using training data only.

    The estimator deliberately separates fitting from labeling so that
    validation/test data never influence the threshold.
    """

    percentile: float = 0.975
    target_column: str = "total_consumption_kwh"
    grouping_columns: tuple[str, ...] = ("fsa", "season")
    thresholds_: pd.DataFrame | None = field(
        default=None,
        init=False,
    )

    def fit(
        self,
        training_frame: pd.DataFrame,
    ) -> "PeakRiskThresholdEstimator":
        """Estimate thresholds from training observations only."""
        thresholds = (
            training_frame
            .groupby(
                list(self.grouping_columns),
                observed=True,
            )[self.target_column]
            .quantile(self.percentile)
            .rename("peak_threshold_kwh")
            .reset_index()
        )

        self.thresholds_ = thresholds
        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply previously fitted thresholds to another dataset."""
        if self.thresholds_ is None:
            raise RuntimeError(
                "PeakRiskThresholdEstimator must be fitted before transform()."
            )

        working = frame.copy()

        labeled = working.merge(
            self.thresholds_,
            on=list(self.grouping_columns),
            how="left",
            validate="many_to_one",
        )

        labeled["peak_threshold_available"] = (
            labeled["peak_threshold_kwh"]
            .notna()
            .astype("int8")
        )

        label = pd.Series(
            pd.NA,
            index=labeled.index,
            dtype="Int8",
        )

        available = labeled[
            "peak_threshold_kwh"
        ].notna()

        label.loc[available] = (
            labeled.loc[
                available,
                self.target_column,
            ]
            > labeled.loc[
                available,
                "peak_threshold_kwh",
            ]
        ).astype("int8")

        labeled["peak_risk_target"] = label

        return labeled


def build_walk_forward_peak_diagnostics(
    frame: pd.DataFrame,
    percentile: float,
    target_column: str,
    grouping_columns: list[str],
    time_column: str = "year",
    minimum_training_years: int = 1,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create leakage-safe diagnostic historical Peak labels.

    For each evaluation year Y:
      thresholds are fitted using years strictly before Y;
      labels are then generated for year Y.

    These labels are diagnostic only and are not the final modeling labels.
    """
    years = sorted(
        pd.Series(frame[time_column])
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    label_parts = []
    threshold_parts = []

    for index, evaluation_year in enumerate(years):
        training_years = years[:index]

        if len(training_years) < minimum_training_years:
            continue

        train = frame.loc[
            frame[time_column].isin(training_years)
        ].copy()

        evaluation = frame.loc[
            frame[time_column] == evaluation_year
        ].copy()

        estimator = PeakRiskThresholdEstimator(
            percentile=percentile,
            target_column=target_column,
            grouping_columns=tuple(grouping_columns),
        ).fit(train)

        thresholds = estimator.thresholds_.copy()
        thresholds["evaluation_year"] = evaluation_year
        thresholds["training_year_start"] = min(training_years)
        thresholds["training_year_end"] = max(training_years)
        threshold_parts.append(thresholds)

        labeled = estimator.transform(evaluation)

        columns = [
            "fsa",
            "timestamp",
            time_column,
            "season",
            target_column,
            "peak_threshold_kwh",
            "peak_threshold_available",
            "peak_risk_target",
        ]

        label_parts.append(
            labeled[
                [column for column in columns if column in labeled.columns]
            ]
        )

    labels = (
        pd.concat(label_parts, ignore_index=True)
        if label_parts
        else pd.DataFrame()
    )

    thresholds = (
        pd.concat(threshold_parts, ignore_index=True)
        if threshold_parts
        else pd.DataFrame()
    )

    return labels, thresholds


def peak_distribution_reports(
    labels: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Summarize leakage-safe diagnostic Peak labels."""
    valid = labels.loc[
        labels["peak_risk_target"].notna()
    ].copy()

    if valid.empty:
        empty = pd.DataFrame()
        return {
            "overall": empty,
            "by_fsa": empty,
            "by_year": empty,
            "by_season": empty,
        }

    def summarize(
        grouping: list[str] | None,
    ) -> pd.DataFrame:
        if grouping:
            result = (
                valid
                .groupby(
                    grouping,
                    observed=True,
                )
                .agg(
                    observations=("peak_risk_target", "size"),
                    peak_hours=("peak_risk_target", "sum"),
                )
                .reset_index()
            )
        else:
            result = pd.DataFrame(
                [
                    {
                        "observations": len(valid),
                        "peak_hours": int(
                            valid["peak_risk_target"].sum()
                        ),
                    }
                ]
            )

        result["peak_rate_pct"] = (
            result["peak_hours"]
            / result["observations"]
            * 100
        )

        return result

    return {
        "overall": summarize(None),
        "by_fsa": summarize(["fsa"]),
        "by_year": summarize(["year"]),
        "by_season": summarize(["season"]),
    }
