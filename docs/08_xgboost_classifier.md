# XGBoost Classifier

## Purpose

1. The pooled, row-wise XGBoost classifier for peak-demand risk is built, trained, and evaluated on all 3 official folds.
2. Predictions are produced in the exact output schema the team agreed on for comparing all 4 models.

## Design

1. `src/peak_risk/XGBoost_Classifier.py` holds `build_training_table()`, `make_model()`, and `train_and_predict()`.
2. `build_training_table()` reshapes the joined hourly dataset into one row per (FSA, origin_timestamp, horizon), keeping two season columns: the origin's season (a model feature) and the forecast_timestamp's season (what a peak threshold must match). `actual_peak` is not added here, since the threshold can only be computed per fold.
3. `CLASSIFIER_FEATURES` is the same 18 `SELECTED_FEATURES` as the regressor plus `horizon` as an explicit feature, since a single row here corresponds to one specific horizon rather than 24 output columns at once.
4. `make_model()` returns an `XGBClassifier` with the same tree settings as the regressor, plus `scale_pos_weight` (computed per fold by the caller as n_negatives/n_positives) to rebalance the loss toward the minority peak class, and `eval_metric="aucpr"` to match PR-AUC, the primary metric.
5. `train_and_predict()` labels train and test with thresholds computed from that fold's own raw hourly training data, not the pivoted table, since a real hour repeats in the pivoted table once per horizon that reaches it, which is not exactly uniform at the edges of the training window.
6. The output keeps `horizon` as a seventh column beyond the required 6, since a single `forecast_timestamp` gets one prediction per horizon that reaches it and the required columns alone cannot tell those rows apart.

## Findings

Findings come from `03_01_xgboost_classifier_training.ipynb`, which trains and evaluates the model on all 3 folds and checks ranking quality, calibration, and top-K capture.

1. PR-AUC is 0.44 for fold_1, 0.58 for fold_2, and 0.58 for fold_3, all well above the no-skill baseline (roughly the base peak rate: 0.11, 0.04, 0.07 respectively). fold_1 is the weakest of the three, the same pattern seen with the regressor, with only 2021-2022 to train on.
2. The actual test peak rate (11.0%, 4.0%, 6.9% by fold) never matches the 2.5% nominal rate the threshold is built from, consistent with what the labeling module showed earlier: the threshold is fixed from older, lower-consumption years while consumption keeps rising.
3. At the default 0.5 probability threshold, fold_1 under-flags peaks (6.7% predicted vs 11.0% actual) while fold_2 and fold_3 over-flag (10.0% vs 4.0%, 11.9% vs 6.9%). This is not fully explained here: a fixed `scale_pos_weight` (about 39 in every fold, since train's positive rate is always about 2.5% by construction) cannot adapt to how far each fold's real test-year prevalence has drifted from that baseline, but that alone does not explain fold_1's under-flagging, which is more likely a symptom of its weaker overall ranking.
4. PR-AUC drops with horizon in fold_1 and fold_3 (0.49 to 0.40, 0.73 to 0.50), while fold_2 drops sharply by h12 then flattens. Recall at the fixed 0.5 threshold barely moves across horizons by comparison (fold_3: 0.82 at h1 to 0.68 at h24), showing that ranking quality erodes with horizon more than the threshold decision does.
5. The model is overconfident, especially at high probabilities: in the top calibration bin, predicted probability averages 0.82 but the observed peak frequency is only 0.48. This is an expected side effect of `scale_pos_weight`, which improves ranking and recall at the cost of calibrated probabilities, the reason PR-AUC rather than Brier score is the primary metric.
6. On fold_3, watching the riskiest 1% of hours by predicted probability catches about 12% of real peaks, 5% catches about 45%, and 10% catches about 69%, each well above what watching the same share of hours at random would catch.
7. The output dataframe matches the required 6-column schema exactly, with `horizon` as the one documented extra column.
