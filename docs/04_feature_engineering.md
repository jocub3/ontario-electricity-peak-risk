# Feature Engineering

## Purpose

1. The target is transformed and `FSA`/`season` are cast to the right dtype for modeling.
2. The full candidate column list is narrowed down to the features actually worth feeding the model.

## Design

1. `src/common/transforms.py` holds `add_log_target()`, which adds `log_consumption = log1p(TOTAL_CONSUMPTION)` while keeping the original column, and `set_categorical_dtypes()`, which casts `FSA` and `season` to pandas `category` for native XGBoost handling.
2. Only the target is transformed, not the features. Tree splits are chosen by scanning one feature's own values, never by comparing raw magnitudes across features, so scaling (`StandardScaler`) and one-hot encoding (`OneHotEncoder`) would add complexity without changing predictions. The target transform is different, it changes the loss the model actually optimizes.
3. Feature selection in `src/common/feature_selection.py` combines three checks: pairwise correlation to flag near-duplicate features, gain-based importance from a quick baseline `XGBRegressor` fit on the whole dataset, and, for every close call the first two flagged, a held-out validation comparison of MAE with and without the feature.
4. The validation split used for that last check is train-2021/validate-2022, not any of the team's 3 official folds (F1 test 2023, F2 test 2024, F3 test 2025). An earlier pass used train-2021-2024/validate-2025, identical to Fold 3's own split, which would have let Fold 3 benefit from having already seen its test year during feature selection. 2021-2022 is the only span never used as a test year in any fold.
5. `SELECTED_FEATURES` is the final list: `FSA`, `season`, 16 numeric/binary features, 18 total, down from 45 candidates.

## Findings

Findings come from `02_01_feature_engineering.ipynb`, which runs the transform, the correlation scan, the baseline gain model, and the validation checks in order.

1. `log1p(TOTAL_CONSUMPTION)` drops overall skew from 0.85 to -0.17, and skew improves for every individual FSA (for example L4T from 1.16 to 0.27, M9R from 1.46 to 0.38).
2. Two correlated clusters show up above 0.85: `month`/`quarter`/`week_of_year`/`day_of_year` (all above 0.94 with each other) and `is_weekend`/`is_workday` (-0.945), along with the already-known `Temp`/`Dew Point Temp` pair (0.92). Correlation alone does not say which side of a pair to keep.
3. Correlation completely misses one redundant pair: `hour` versus `hour_sin`/`hour_cos` never shows up above 0.85 (only -0.10), even though `hour_sin`/`hour_cos` are built directly from `hour`. Pearson only sees straight-line relationships, and `hour` versus `cos(hour)` is far from a straight line.
4. The baseline gain model confirms `Temp` matters (0.040 gain, top 4) despite its weak linear correlation (0.08), the same non-monotonic relationship already found in the EDA. About 25 of the 45 candidate features contribute close to nothing, several with exactly 0.000000 gain, and are dropped outright.
5. Validating the two flagged pairs reverses what gain alone suggested. `hour_sin`/`hour_cos` looked useful by gain (0.027 and 0.007) but add no benefit once `hour` is in the model at any tree depth tested, since they are a deterministic function of `hour` with zero new information by construction. They are dropped. `Dew Point Temp (°C)` looked low-value by gain (0.006) but removing it makes validation MAE worse, since dew point depends on humidity too, not on temperature alone, so its correlation with `Temp` is empirical, not exact.
6. `month`/`quarter`/`week_of_year`/`day_of_year`, `is_weekend`, `Rel Hum (%)`, and the holiday-distance features (`days_to_holiday`, `days_after_holiday`) are all confirmed unhelpful or actively harmful on the corrected split, consistent with the gain screen.
7. Two features gave small, inconsistent results across the two validation splits tried: `weekday` (kept, never hurt in either check) and `Wind Dir (10s deg)` (dropped, never helped by more than a rounding error). Both are flagged as judgment calls to revisit once the real 3-fold evaluation exists, not clean results like the rest.
8. `is_public_holiday` is kept on interpretability grounds despite low gain (0.0007) and a small, direction-inconsistent validation effect, the EDA found a real, if modest, holiday effect, and the feature never showed a meaningful downside.
