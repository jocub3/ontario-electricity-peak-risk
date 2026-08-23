# Folds and Forecast Targets

## Purpose

1. The 3 expanding-window folds used to evaluate both models are defined in one shared place.
2. The raw hourly consumption series is pivoted into the actual 24h-ahead forecast targets, in both the long form the classifier needs and the wide form the multi-output regressor needs.

## Design

1. `src/common/folds.py` holds `FOLDS`, a fixed dictionary with the 3 official folds (fold_1 train 2021-2022 test 2023, fold_2 train 2021-2023 test 2024, fold_3 train 2021-2024 test 2025), `split_fold()` to get one fold's train/test split, and `iter_folds()` to loop over all 3. The training window only ever grows, it never drops the earliest years the way a rolling window would.
2. `src/common/targets.py` holds `build_long_targets()` (one row per FSA/origin/horizon, used by the classifier) and `build_wide_targets()` (one row per FSA/origin with 24 horizon columns, used by the multi-output regressor).
3. Both target functions rely on the same fact: each FSA is a complete, gap-free hourly grid, so "h hours after a given origin" is exactly "h rows later" in that FSA's own series. A `groupby("FSA").shift(-h)` is enough, no timestamp lookup or join is needed.
4. Both functions trim to the same set of origins: an origin near the end of an FSA's history with only some of the 24 horizons available gets dropped entirely from both tables, rather than kept with a partial set of rows in the long table only.

## Findings

Findings come from `02_02_targets_and_folds.ipynb`, which builds the folds and both target tables and checks them against each other and against known raw values.

1. Every fold's actual `train_years`/`test_years` line up exactly with what `FOLDS` specifies, and there is no overlap between train and test in any fold.
2. Row counts grow as expected across folds (fold_2's train is fold_1's train plus one more year, and so on). fold_2's test year has slightly more rows than fold_1 or fold_3 (52,704 versus 52,560) because 2024 is a leap year, 6 FSAs times 8,784 hours instead of 8,760.
3. `build_long_targets()` produces 6,307,200 rows (5 columns), `build_wide_targets()` produces 262,800 rows (26 columns), consistent with 262,800 valid origins times 24 horizons for the long table.
4. Checking one concrete origin (M5S, 2021-01-01 00:00) against the raw hourly data confirms both tables agree with each other and with the source: horizon 1/2/3 in the long table (3868.8, 3652.6, 3538.7) match `actual_h1`/`actual_h2`/`actual_h3` in the wide table exactly.
5. The long and wide tables cover the exact same set of 262,800 origins, and every origin has exactly 24 rows in the long table, confirming the trimming at the end of each FSA's history is consistent between the two shapes.
