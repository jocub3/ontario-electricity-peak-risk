"""Correlation matrices and diagnostic VIF analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(
    frame: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Return a numeric-feature correlation matrix."""
    return frame.select_dtypes(include=[np.number]).corr(method=method)


def vif_analysis(
    frame: pd.DataFrame,
    config: dict,
    maximum_rows: int = 50000,
) -> pd.DataFrame:
    """
    Calculate diagnostic VIF for sufficiently complete numeric predictors.

    VIF is generated for diagnostic purposes only.

    It must not be used as an automatic feature-removal criterion because:
    - several engineered variables are mathematically related by design;
    - tree-based models are generally less sensitive to multicollinearity;
    - final feature selection must be model-specific.

    VIF results are primarily relevant for linear and statistical models.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for VIF analysis. "
            "Install it with: pip install statsmodels"
        ) from exc

    target = config["feature_engineering"]["target_column"]
    traceability = set(config["feature_engineering"]["traceability_columns"])

    numeric = frame.select_dtypes(include=[np.number]).copy()

    excluded = {target}
    excluded.update(
        column
        for column in traceability
        if column in numeric.columns
    )
    numeric = numeric.drop(
        columns=[column for column in excluded if column in numeric.columns],
        errors="ignore",
    )

    # VIF cannot be calculated reliably with missing or constant columns.
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    numeric = numeric.loc[:, numeric.isna().mean() <= 0.05]
    numeric = numeric.dropna()

    if len(numeric) > maximum_rows:
        numeric = numeric.sample(maximum_rows, random_state=42)

    if numeric.shape[1] < 2 or numeric.empty:
        return pd.DataFrame(columns=["feature", "vif"])

    values = numeric.astype(float).values

    return pd.DataFrame(
        {
            "feature": numeric.columns,
            "vif": [
                variance_inflation_factor(values, index)
                for index in range(values.shape[1])
            ],
        }
    ).sort_values("vif", ascending=False).reset_index(drop=True)
