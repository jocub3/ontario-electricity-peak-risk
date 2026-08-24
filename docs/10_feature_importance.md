# Feature Importance: Regressor vs. Classifier

## Purpose

1. Gain-based feature importance is extracted from the real, final Regressor and Classifier (both trained on fold_3), not from the throwaway baseline model used earlier for feature selection.
2. The two rankings are compared side by side to check whether the two models, which share the same 18 features, actually lean on them the same way.

## Design

1. `src/common/feature_selection.py` gained one function, `gain_importance(model, features)`, a thin wrapper around `model.feature_importances_` that works for both `XGBRegressor` and `XGBClassifier` since both expose it the same way.
2. `notebooks/04_evaluation/04_02_feature_importance.ipynb` trains both models on fold_3 (train 2021-2024, the largest and most recent training window) using the exact same functions as their own training notebooks, `build_training_table()`/`make_model()` from each model's module, then reads off importance from each fitted model.
3. `horizon` only applies to the Classifier (`CLASSIFIER_FEATURES` adds it to the 18 `SELECTED_FEATURES`), it appears with a blank Regressor rank in the comparison table rather than being left out.
4. This notebook does not re-run feature selection, it only asks how the two models use the features they already have. Whether a different feature set would score better for the Classifier specifically is a separate, unanswered question, `SELECTED_FEATURES` was chosen and validated once, against the Regressor's target.

## Findings

Findings come from `04_02_feature_importance.ipynb`, comparing `feature_importances_` from both fold_3 models.

1. The two rankings barely resemble each other. `FSA` takes 0.746 of all gain in the Regressor, by far the largest share of any feature in either model, then drops to rank 8 (0.061) in the Classifier.
2. `Temp (°C)`, `season`, and `Dew Point Temp (°C)` take the opposite path: ranks 6, 4, and 13 in the Regressor, but the top 3 in the Classifier (0.132, 0.104, 0.079).
3. Both directions have a concrete explanation. `TOTAL_CONSUMPTION` differs enormously in scale between FSAs (M5S averages about 4,100 kWh, M9W about 12,300), so a regressor predicting the raw value leans on `FSA` just to get the right order of magnitude. `actual_peak` is already defined relative to each FSA's own distribution (the 97.5th percentile per FSA and season), so that scale is baked into the label itself before the model sees a row, what is left to explain is which hours cross into an FSA's own extreme, and weather drives that far more than which FSA it is.
4. `horizon` lands at rank 6 for the Classifier (0.063), meaningfully used, consistent with the already-documented finding that PR-AUC erodes with horizon.
5. This does not mean the current 18 features are wrong for the Classifier, only that gain alone cannot say whether a different set would do better, that would need the same correlation, gain, and held-out validation process already run for the Regressor, repeated with `actual_peak` as the target. Left as a documented limitation, not resolved here.
