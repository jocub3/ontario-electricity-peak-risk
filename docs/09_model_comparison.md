# Model Comparison

## Purpose

1. Both models (the XGBoost Regressor and the XGBoost Classifier) are retrained fresh on the same 3 folds and their results are consolidated side by side, as the closing summary of Model 3.
2. Final prediction tables are exported in the exact schema the team agreed on for comparing all 4 teammates' models.

## Design

1. `notebooks/04_evaluation/04_01_model_comparison.ipynb` imports `build_training_table`/`train_and_predict` from both `XGBoost_Regressor.py` and `XGBoost_Classifier.py` directly, so it stands on its own without assuming the earlier training notebooks were already opened.
2. A PR curve (precision and recall at every threshold, via `pr_curve_data()` in `metrics.py`) is added here for the classifier, alongside the metrics and diagnostics already shown in the earlier notebooks, plotted against a no-skill baseline equal to the base peak rate.
3. Both models' full 3-fold predictions are concatenated and exported to `data/interim/results/` (gitignored, generated), one CSV per model, matching the required output schema exactly.
4. "Model comparison" here means the Regressor and Classifier that make up Model 3 compared with each other, not a comparison against the other 3 teammates' algorithms, that comparison happens later, outside this branch, once all 4 models are merged and evaluated under the same conditions.

## Findings

Findings come from `04_01_model_comparison.ipynb`, which reproduces both models' training and consolidates their diagnostics.

1. Both models show the same fold_1-weakest pattern, on every metric and at every horizon, consistent with fold_1 having the least training data (2021-2022 only).
2. The PR curve for the classifier's fold_3 sits well above the no-skill baseline (6.9%) across most of the recall range, the clearest single visual confirmation that the model has real ranking skill, while also making the earlier calibration finding visible in context: the curve never approaches precision=1, consistent with `scale_pos_weight` trading calibrated probabilities for better separation.
3. Neither model has had hyperparameter tuning, this is Model 3's first complete pass, not a tuned final version.
4. The Regressor's WAPE runs around 6% on fold_2/fold_3, growing to about 7-8% by h+24. Checking whether that growth comes from the persisted-weather assumption (origin's weather reused for all 24 horizons) found it is only a partial explanation, hours where the real weather barely changed still show most of the same error growth, so some other source of uncertainty that accumulates with horizon is the bigger contributor. Its `bias` is positive in all 3 folds (under-prediction), consistent with training years always sitting earlier than a still-rising test year.
5. The Classifier's PR-AUC runs around 0.58 on fold_2/fold_3, well above each fold's no-skill baseline, with three known, explainable limitations carried forward rather than hidden: overconfidence at high predicted probabilities, inconsistent over/under-flagging at the default 0.5 threshold across folds, and a `peak_rate_bias` that flips sign between fold_1 (negative) and fold_2/fold_3 (positive).
6. Both models share the same open question about fold_1: this notebook cannot cleanly separate whether its weakness comes from having the least training data or from 2023 being a harder year to predict, since expanding window confounds those two things by design. A separate, informal diagnostic is planned to look into that, outside the scope of this notebook.
7. Both final prediction tables are exported to `data/interim/results/`, matching the required schema exactly (verified column by column).
