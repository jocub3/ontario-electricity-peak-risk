"""Target and categorical dtype transformations for modeling.

Feature scaling and one-hot encoding are deliberately not included here: tree-based
models like XGBoost are invariant to monotonic transformations and scale of individual
features, so those steps would add complexity without changing predictions. The target
transform below is different - it changes the loss the model actually optimizes.
"""

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["FSA", "season"]


def add_log_target(
    df: pd.DataFrame, source_col: str = "TOTAL_CONSUMPTION", target_col: str = "log_consumption"
) -> pd.DataFrame:
    """A log1p-transformed copy of the target is added, keeping the original column.

    Training on log1p(consumption) makes a given percentage error contribute the same
    amount to the loss regardless of an FSA's scale, instead of squared-error naturally
    weighting the largest FSAs more just because their raw values are bigger. Predictions
    made on this target need expm1() applied before reporting them in real kWh.
    """
    df = df.copy()
    df[target_col] = np.log1p(df[source_col])
    return df


def set_categorical_dtypes(
    df: pd.DataFrame, columns: list[str] = CATEGORICAL_COLUMNS
) -> pd.DataFrame:
    """The given columns are cast to pandas category dtype for native XGBoost categorical handling."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category")
    return df
