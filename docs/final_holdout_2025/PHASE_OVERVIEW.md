# Final Holdout Evaluation 2025

This phase opens the protected 2025 period exactly once after all model,
hyperparameter, feature, preprocessing, Peak-definition, and operational
threshold decisions have been frozen.

Forecasting:
- Random Forest Regressor
- MAE, RMSE, MAPE, WAPE, Bias
- global, FSA, horizon, and month analysis
- actual-vs-predicted and residual diagnostics

Peak-Risk:
- XGBoost Classifier
- operational threshold = 0.06
- PR-AUC, ROC-AUC, Precision, Recall, F1, Balanced Accuracy
- **Brier Score**
- global, FSA, horizon, and month analysis
- PR curve, ROC curve, confusion matrix, calibration-related evidence

No post-holdout tuning is permitted.
