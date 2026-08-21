"""Expanding-window folds shared across the forecasting and peak-risk models.

- Fold 1: train 2021-2022 -> test 2023
- Fold 2: train 2021-2023 -> test 2024
- Fold 3: train 2021-2024 -> test 2025
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

import pandas as pd


class Fold(NamedTuple):
    number: int
    train_years: tuple[int, ...]
    test_year: int


FOLDS = [
    Fold(number=1, train_years=(2021, 2022), test_year=2023),
    Fold(number=2, train_years=(2021, 2022, 2023), test_year=2024),
    Fold(number=3, train_years=(2021, 2022, 2023, 2024), test_year=2025),
]


def split_fold(
    df: pd.DataFrame, fold: Fold, timestamp_column: str = "timestamp_local"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, test) for one fold, by calendar year."""
    year = df[timestamp_column].dt.year
    train = df.loc[year.isin(fold.train_years)]
    test = df.loc[year == fold.test_year]
    return train, test


def iter_folds(
    df: pd.DataFrame, timestamp_column: str = "timestamp_local"
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(fold, train, test)`` for each of the 3 folds, in order."""
    for fold in FOLDS:
        train, test = split_fold(df, fold, timestamp_column=timestamp_column)
        yield fold, train, test
