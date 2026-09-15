"""Validation rules for engineered features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_feature_dataset(
    base: pd.DataFrame,
    engineered: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Validate keys, ordering, row preservation, finite values, and lag logic."""
    checks = []
    key = config["feature_engineering"]["key_columns"]
    target = config["feature_engineering"]["target_column"]

    def add(name: str, violations: int, description: str) -> None:
        checks.append(
            {
                "check": name,
                "violations": int(violations),
                "status": "PASS" if violations == 0 else "FAIL",
                "description": description,
            }
        )

    add(
        "row_count_preserved",
        abs(len(base) - len(engineered)),
        "Feature Engineering must not silently add or remove rows.",
    )

    add(
        "unique_key",
        int(engineered.duplicated(key, keep=False).sum()),
        "FSA + timestamp must remain unique.",
    )

    add(
        "key_not_null",
        int(engineered[key].isna().any(axis=1).sum()),
        "Key columns must not contain null values.",
    )

    numeric = engineered.select_dtypes(include=[np.number])
    add(
        "no_infinite_values",
        int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()),
        "Numeric engineered features must not contain positive or negative infinity.",
    )

    if f"{target}_lag_1h" in engineered.columns:
        expected = engineered.groupby("fsa", observed=True)[target].shift(1)
        actual = engineered[f"{target}_lag_1h"]
        mismatch = ~(
            actual.eq(expected)
            | (actual.isna() & expected.isna())
        )
        add(
            "lag_1h_alignment",
            int(mismatch.sum()),
            "One-hour target lag must equal the prior FSA observation.",
        )

    for column in engineered.columns:
        if "_rolling_" in column and column.startswith(target):
            # Ensure the first non-null rolling value is not at the first row in each FSA.
            invalid = 0
            for _, group in engineered.groupby("fsa", observed=True):
                if len(group) and pd.notna(group[column].iloc[0]):
                    invalid += 1
            add(
                f"rolling_history_required::{column}",
                invalid,
                "Rolling target features must require historical observations.",
            )

    return pd.DataFrame(checks)
