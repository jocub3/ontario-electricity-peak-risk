# Peak Labeling and Evaluation Metrics

## Purpose

1. The peak-risk labels the classifier will learn to predict are defined.
2. The evaluation functions used to score both models are defined and checked against known, hand-calculated values before any real model exists to test them on.

## Design

1. `src/common/labeling.py` holds `compute_peak_thresholds()`, which computes the 97.5th percentile of consumption per (FSA, season) from a fold's train data only, and `label_peaks()`, which adds `actual_peak` (1/0) by comparing a value against its (FSA, season) threshold. `label_peaks()` takes a `season_col` argument so the classifier can label against `forecast_timestamp`'s season while keeping the origin's season as a separate model feature.
2. `src/common/metrics.py` holds the forecasting metrics (`mae`, `wape`, `rmse`, `mape`, `forecast_metrics`, `forecast_metrics_by_horizon`) and the peak-risk metrics (`peak_risk_metrics`, `top_k_capture_rate`, `calibration_data`, `pr_curve_data`). WAPE is the team's primary forecast metric, PR-AUC is the primary peak-risk metric.
3. Every metric function is checked in the notebook against toy data, small inputs where the correct answer can be worked out by hand first, rather than trusting the implementation on real data with no independent way to verify the numbers.

## Findings

Findings come from `02_03_labeling_and_metrics.ipynb`, which runs both modules across the 3 real folds and against hand-picked toy examples.

1. Each fold produces exactly 24 thresholds (6 FSAs times 4 seasons), and the train split always lands almost exactly on 2.5% peak, expected, since the threshold is by definition the 97.5th percentile of that same train data.
2. The test split never lands near 2.5%: fold_1 (test 2023) comes out around 11%, fold_2 (test 2024) around 4%, fold_3 (test 2025) around 7%. This is not a bug, the threshold is fixed from earlier, smaller years, and since consumption trends upward over time (consistent with the `PREMISE_COUNT` step increases already seen in the EDA), a growing share of test-year hours end up crossing a threshold calibrated on the past. The real class imbalance the classifier faces at evaluation time is milder and more fold-dependent than the 97.5/2.5 split the training data always shows.
3. On toy data (`actual = [10, 20, 30]`, `prediction = [12, 18, 33]`), `forecast_metrics()` returns MAE 2.333, WAPE 0.1167, RMSE 2.380, and MAPE 13.33%, matching the hand-calculated values exactly.
4. `forecast_metrics_by_horizon()` computes each horizon's metrics independently and correctly on a toy example with two known-error horizon groups.
5. On a toy example with a perfect classifier (`actual_peak = [0, 0, 1, 1]`, `predicted_peak` matching exactly, probabilities correctly ranking the two peaks highest), `peak_risk_metrics()` returns 1.0 for precision, recall, F1, PR-AUC, and ROC-AUC, and a nonzero Brier score (0.085), since the probabilities are not exactly 0 or 1 even though the classification is perfect.
6. `top_k_capture_rate` and `calibration_data` both behave correctly on the same toy data: capture rate is 1.0 when k equals the number of true peaks and the top-k rows are exactly those peaks, and the two calibration bins show 0 observed frequency for the low-probability bin and 1 for the high-probability bin, exactly matching which toy points are real peaks.
