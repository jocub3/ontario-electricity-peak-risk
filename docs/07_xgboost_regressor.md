# XGBoost Regressor

## Purpose

1. The pooled, multi-output XGBoost regressor is built, trained, and evaluated on all 3 official folds.
2. Predictions are produced in the exact output schema the team agreed on for comparing all 4 models.

## Design

1. `src/forecast/XGBoost_Regressor.py` holds `build_training_table()`, `make_model()`, and `train_and_predict()`.
2. `build_training_table()` merges the 18 `SELECTED_FEATURES` (taken at `origin_timestamp`) with the 24-horizon wide targets, keeping two versions: the log-scale columns the model is fit on, and the real-kWh columns used only for reporting and metrics.
3. `make_model()` returns an `XGBRegressor` with `multi_strategy="one_output_per_tree"`, XGBoost's native multi-output support, which produces one column per horizon without needing a wrapper like scikit-learn's `MultiOutputRegressor`. `enable_categorical=True` lets `FSA` be used natively as a category column. Hyperparameters (`n_estimators=300`, `max_depth=6`, `learning_rate=0.1`) are a first pass, not tuned.
4. `train_and_predict()` fits on log-scale targets, applies `expm1()` to convert predictions back to real kWh, and returns results already unpivoted into the required long-form schema: `FSA, origin_timestamp, forecast_timestamp, horizon, actual, prediction, model_name`.

## Findings

Findings come from `02_04_xgboost_regressor_training.ipynb`, which trains and evaluates the model on all 3 folds.

1. Overall WAPE is 12.15% for fold_1 (test 2023), 6.08% for fold_2 (test 2024), and 6.35% for fold_3 (test 2025). fold_1 is clearly the weakest of the three across every metric (MAE, WAPE, RMSE, MAPE).
2. fold_1 has the least training data of the three folds (2021-2022 only, versus 3 and 4 years for fold_2 and fold_3), consistent with its weaker result, though the folds also differ in which year they test on, so the two effects are not fully separated by this comparison alone.
3. WAPE grows with horizon in every fold: fold_2 goes from 4.65% at h+1 to 7.14% at h+24, fold_3 from 4.99% to 7.75%, both roughly a 55% relative increase. fold_1 grows from 10.92% to 13.19%, a smaller relative jump (about 21%) but starting from, and staying at, a much higher error throughout. The degradation is smooth and gradual, not a sharp jump at any single horizon.
4. The output dataframe matches the required schema exactly, column names and order both, confirmed directly against `fold_results["fold_3"].head()`.
