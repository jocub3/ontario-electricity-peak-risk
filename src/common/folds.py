"""Expanding-window folds locked by the team's methodology - shared by both models and by labeling.py."""

import pandas as pd

FOLDS = {
    "fold_1": {"train_years": (2021, 2022), "test_year": 2023},
    "fold_2": {"train_years": (2021, 2023), "test_year": 2024},
    "fold_3": {"train_years": (2021, 2024), "test_year": 2025},
}


def split_fold(
    df: pd.DataFrame, fold_name: str, year_col: str = "year"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for one fold."""
    fold = FOLDS[fold_name]
    start_year, end_year = fold["train_years"]
    train_df = df[df[year_col].between(start_year, end_year)]
    test_df = df[df[year_col] == fold["test_year"]]
    return train_df, test_df


def iter_folds(df: pd.DataFrame, year_col: str = "year"):
    """Yield (fold_name, train_df, test_df) for all 3 folds, in order."""
    for fold_name in FOLDS:
        train_df, test_df = split_fold(df, fold_name, year_col)
        yield fold_name, train_df, test_df
