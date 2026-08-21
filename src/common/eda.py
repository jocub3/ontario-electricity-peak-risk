"""Reusable EDA helpers: distributions, outliers, and seasonal profiles."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_distribution(
    df: pd.DataFrame, column: str, by: str | None = None, bins: int = 60
) -> plt.Figure:
    """Histogram of a numeric column, optionally overlaid per group."""
    fig, ax = plt.subplots(figsize=(9, 4))
    if by is None:
        ax.hist(df[column].dropna(), bins=bins, color="#4C72B0")
    else:
        for key, group in df.groupby(by):
            ax.hist(group[column].dropna(), bins=bins, alpha=0.5, label=str(key))
        ax.legend(title=by, ncol=2, fontsize=8)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {column}" + (f" by {by}" if by else ""))
    fig.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, column: str, by: str) -> plt.Figure:
    """Boxplot of a numeric column split by a categorical group."""
    fig, ax = plt.subplots(figsize=(9, 4))
    df.boxplot(column=column, by=by, ax=ax)
    ax.set_title(f"{column} by {by}")
    ax.set_xlabel(by)
    ax.set_ylabel(column)
    plt.suptitle("")
    fig.tight_layout()
    return fig


def iqr_outlier_summary(df: pd.DataFrame, column: str, group_by: str | None = None) -> pd.DataFrame:
    """Count of IQR-rule outliers (below Q1-1.5*IQR or above Q3+1.5*IQR), overall or per group."""

    def bounds_and_counts(series: pd.Series) -> dict:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_below": int((series < lower).sum()),
            "n_above": int((series > upper).sum()),
            "pct_outliers": round(((series < lower) | (series > upper)).mean() * 100, 2),
        }

    if group_by is None:
        return pd.DataFrame([bounds_and_counts(df[column])])

    rows = []
    for key, group in df.groupby(group_by):
        row = bounds_and_counts(group[column])
        row[group_by] = key
        rows.append(row)
    cols = [group_by, "lower_bound", "upper_bound", "n_below", "n_above", "pct_outliers"]
    return pd.DataFrame(rows)[cols]


def average_profile(
    df: pd.DataFrame, column: str, period_col: str, group_by: str = "FSA"
) -> pd.DataFrame:
    """Average of a numeric column by a time period (hour/weekday/month/...), one column per group."""
    return df.groupby([period_col, group_by])[column].mean().unstack(group_by)


def plot_average_profile(
    df: pd.DataFrame, column: str, period_col: str, group_by: str = "FSA"
) -> plt.Figure:
    """Line plot of the average profile returned by average_profile()."""
    profile = average_profile(df, column, period_col, group_by)
    fig, ax = plt.subplots(figsize=(9, 4))
    profile.plot(ax=ax, marker="o", markersize=3)
    ax.set_xlabel(period_col)
    ax.set_ylabel(f"mean {column}")
    ax.set_title(f"Average {column} by {period_col}")
    ax.legend(title=group_by, ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_time_series(
    df: pd.DataFrame,
    column: str,
    timestamp_col: str = "timestamp_local",
    group_by: str = "FSA",
    freq: str = "D",
) -> plt.Figure:
    """Resampled time series line plot, one line per group."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for key, group in df.groupby(group_by):
        resampled = group.set_index(timestamp_col)[column].resample(freq).mean()
        ax.plot(resampled.index, resampled.values, label=str(key), linewidth=0.8)
    ax.set_xlabel(timestamp_col)
    ax.set_ylabel(f"mean {column} ({freq})")
    ax.set_title(f"{column} over time")
    ax.legend(title=group_by, ncol=2, fontsize=8)
    fig.tight_layout()
    return fig
