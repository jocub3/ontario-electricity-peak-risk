"""Dataset overview, temporal coverage, continuity, and base missingness."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    categorical_columns,
    numeric_columns,
    save_figure,
    save_table,
    write_section_markdown,
)


def dataset_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a compact overview table."""
    return pd.DataFrame(
        [
            {
                "rows": len(frame),
                "columns": frame.shape[1],
                "memory_mb": round(
                    frame.memory_usage(deep=True).sum() / 1024**2,
                    3,
                ),
                "numeric_columns": len(numeric_columns(frame)),
                "categorical_columns": len(categorical_columns(frame)),
                "minimum_timestamp": frame["timestamp"].min(),
                "maximum_timestamp": frame["timestamp"].max(),
                "unique_fsas": frame["fsa"].nunique(),
            }
        ]
    )


def dtype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize every column's dtype and cardinality."""
    return pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "non_null_count": [int(frame[column].notna().sum()) for column in frame.columns],
            "missing_pct": [
                round(frame[column].isna().mean() * 100, 6)
                for column in frame.columns
            ],
            "unique_count": [
                int(frame[column].nunique(dropna=True))
                for column in frame.columns
            ],
        }
    )


def temporal_coverage_by_fsa(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate complete, continuous hourly coverage for each FSA."""
    rows = []

    for fsa, group in frame.groupby("fsa", observed=True, sort=True):
        timestamps = pd.DatetimeIndex(
            group["timestamp"].dropna().drop_duplicates().sort_values()
        )
        expected = pd.date_range(
            timestamps.min(),
            timestamps.max(),
            freq="h",
        )
        missing = expected.difference(timestamps)
        deltas = pd.Series(timestamps).diff().dropna()

        rows.append(
            {
                "fsa": str(fsa),
                "rows": len(group),
                "unique_hours": len(timestamps),
                "minimum_timestamp": timestamps.min(),
                "maximum_timestamp": timestamps.max(),
                "expected_continuous_hours": len(expected),
                "missing_hour_count": len(missing),
                "duplicate_timestamp_rows": int(
                    group.duplicated(["timestamp"], keep=False).sum()
                ),
                "non_one_hour_gaps": int((deltas != pd.Timedelta(hours=1)).sum()),
                "first_missing_examples": " | ".join(
                    missing[:5].strftime("%Y-%m-%d %H:%M:%S").tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness across the complete dataset."""
    return (
        pd.DataFrame(
            {
                "column": frame.columns,
                "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
                "missing_pct": [
                    round(frame[column].isna().mean() * 100, 6)
                    for column in frame.columns
                ],
            }
        )
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )


def plot_continuity(
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot observed versus expected hourly coverage by FSA."""
    figure, axis = plt.subplots(figsize=(10, 5))
    x = range(len(coverage))
    width = 0.38

    axis.bar(
        [value - width / 2 for value in x],
        coverage["unique_hours"],
        width=width,
        label="Observed",
    )
    axis.bar(
        [value + width / 2 for value in x],
        coverage["expected_continuous_hours"],
        width=width,
        label="Expected",
    )
    axis.set_xticks(list(x))
    axis.set_xticklabels(coverage["fsa"])
    axis.set_xlabel("FSA")
    axis.set_ylabel("Hourly observations")
    axis.set_title("Temporal coverage by FSA")
    axis.legend()

    save_figure(figure, output_path)


def plot_missingness(
    missingness: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot columns containing missing values."""
    subset = missingness.loc[missingness["missing_pct"] > 0].copy()

    figure, axis = plt.subplots(figsize=(11, max(4, len(subset) * 0.35)))
    axis.barh(subset["column"], subset["missing_pct"])
    axis.invert_yaxis()
    axis.set_xlabel("Missing values (%)")
    axis.set_ylabel("Column")
    axis.set_title("Missingness by variable")

    save_figure(figure, output_path)


def run_overview(
    frame: pd.DataFrame,
    reports_dir: Path,
    figures_dir: Path,
    docs_dir: Path,
) -> dict[str, pd.DataFrame]:
    overview = dataset_overview(frame)
    dtypes = dtype_summary(frame)
    coverage = temporal_coverage_by_fsa(frame)
    missingness = missingness_summary(frame)

    save_table(overview, reports_dir / "05_01_dataset_overview.csv")
    save_table(dtypes, reports_dir / "05_01_dtype_summary.csv")
    save_table(coverage, reports_dir / "05_01_temporal_coverage_by_fsa.csv")
    save_table(missingness, reports_dir / "05_01_missingness_summary.csv")

    plot_continuity(
        coverage,
        figures_dir / "05_01_temporal_coverage_by_fsa.png",
    )
    plot_missingness(
        missingness,
        figures_dir / "05_01_missingness_by_variable.png",
    )

    write_section_markdown(
        docs_dir / "05_01_Dataset_Overview_and_Coverage.md",
        "EDA 05.01 — Dataset Overview and Coverage",
        [
            "This section validates the analytical dataset dimensions, data types, "
            "memory footprint, temporal coverage, hourly continuity, and missingness "
            "before substantive exploration.",
        ],
        [
            ("Dataset overview", overview),
            ("Temporal coverage by FSA", coverage),
            ("Variables with missing values", missingness.loc[missingness["missing_pct"] > 0]),
        ],
        [
            ("Temporal coverage by FSA", "../../reports/figures/eda/05_01_temporal_coverage_by_fsa.png"),
            ("Missingness by variable", "../../reports/figures/eda/05_01_missingness_by_variable.png"),
        ],
    )

    return {
        "overview": overview,
        "dtypes": dtypes,
        "coverage": coverage,
        "missingness": missingness,
    }
