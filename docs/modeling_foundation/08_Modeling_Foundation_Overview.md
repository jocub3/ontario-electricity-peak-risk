# Modeling Foundation Phase - Overview

## 1. Purpose of the Phase

Modeling Foundation Phase established the common modeling foundation for the two predictive tasks of the project:

1. **Electricity Demand Forecasting**
2. **Peak-Risk Classification**

The objective of this phase was not to train predictive models. Instead, it created the common experimental framework that all future models must follow.

This ensures that different algorithms can be compared under the same conditions and prevents individual model implementations from using different data splits, target definitions, evaluation metrics, or preprocessing rules.

The Modeling Foundation therefore acts as a shared contract between all future modeling branches.

---

## 2. General Workflow

The phase was organized into a sequence of notebooks:

| Notebook | Purpose |
|---|---|
| `08_00_modeling_design.ipynb` | Define the overall modeling strategy |
| `08_01_common_splits.ipynb` | Define common chronological data splits |
| `08_02_common_metrics.ipynb` | Define official evaluation metrics |
| `08_03_preprocessing_rules.ipynb` | Define preprocessing and anti-leakage rules |
| `08_04_evaluation_framework.ipynb` | Establish common model evaluation procedures |
| `08_05_forecast_horizon_rules.ipynb` | Define the 24-hour Forecasting evaluation structure |
| `08_06_peak_risk_rules.ipynb` | Define Peak-Risk modeling and evaluation rules |
| `08_07_shared_configuration.ipynb` | Review the centralized modeling configuration |
| `08_08_framework_validation.ipynb` | Validate the complete common framework |
| `08_09_modeling_foundation_summary.ipynb` | Review the final Modeling Foundation outputs |
| `08_99_run_complete_modeling_foundation.ipynb` | Execute the complete phase as a reproducible workflow |

The following sections summarize the purpose and outcome of each step.

---

# 3. Step 08_00 – Modeling Design

## Objective

The first step formally defined how the modeling stage of the project will operate.

The main purpose was to establish a common methodology before any individual model is developed.

## Main Decisions

The project contains two predictive tasks:

### Forecasting

Predict electricity consumption for the next 24 hourly periods.

The prediction horizon is represented as:

- `target_h01`
- `target_h02`
- ...
- `target_h24`

### Peak-Risk

Estimate whether each of the following 24 hours will be classified as a Peak period.

The Peak-Risk horizon is represented as:

- `peak_h01`
- `peak_h02`
- ...
- `peak_h24`

Both tasks therefore operate under the same 24-hour prediction horizon.

## Result

A common modeling design was established so that all future algorithms will use the same experimental rules.

The detailed design is documented in:

`docs/modeling_foundation/08_00_Modeling_Foundation_Design.md`

---

# 4. Step 08_01 – Common Temporal Splits

## Objective

This step defined how historical observations will be separated into training, validation, and final test periods.

Because electricity consumption is a time series, random train/test splitting is not appropriate.

The project therefore uses chronological expanding-window validation.

## Validation Strategy

Two development folds were defined:

### Fold 2023

Training:

`2021–2022`

Validation:

`2023`

### Fold 2024

Training:

`2021–2023`

Validation:

`2024`

A final holdout period was also defined:

### Final Test

Training:

`2021–2024`

Test:

`2025`

The 2025 period is reserved for final model evaluation and must not be used for model selection or tuning.

## Horizon-Safe Splitting

An additional safeguard was implemented because the project predicts 24 hours into the future.

A forecast origin is only allowed when all 24 future target hours remain inside the same training, validation, or test period.

For example:

If the training observation period ends at:

`2022-12-31 23:00`

the final valid 24-hour forecast origin is:

`2022-12-30 23:00`

This prevents future target observations from crossing from the training period into the validation period.

## FSA Coverage

The split structure was also validated independently for all six FSA areas.

All FSA areas were present in every training, validation, and test window with consistent temporal coverage.

## Result

The project now has a single chronological split strategy that must be reused by every future model.

Generated reports include:

- `08_01_common_split_summary.csv`
- `08_01_fold_fsa_coverage.csv`

---

# 5. Step 08_02 – Common Metrics

## Objective

This step defined the official metrics that will be used to evaluate and compare models.

Using common metrics is essential because different algorithms must be evaluated under identical criteria.

## Forecasting Metrics

The primary Forecasting metrics are:

- **MAE – Mean Absolute Error**
- **RMSE – Root Mean Squared Error**
- **MAPE – Mean Absolute Percentage Error**

Secondary metrics are:

- **WAPE – Weighted Absolute Percentage Error**
- **Bias**

MAE was selected as the principal model-selection metric.

This means that MAE will provide the main ranking criterion when comparing Forecasting models, while the other metrics will remain important supporting evidence.

## Peak-Risk Metrics

The primary Peak-Risk metrics are:

- **Precision**
- **Recall**
- **F1 Score**
- **PR-AUC**

Secondary metrics are:

- **ROC-AUC**
- **Balanced Accuracy**

PR-AUC was selected as the principal model-selection metric because Peak observations represent a minority class.

## Result

All model branches will report the same metrics and follow the same model-comparison criteria.

The metric definitions are stored in:

`08_02_metric_dictionary.csv`

---

# 6. Step 08_03 – Common Preprocessing Rules

## Objective

Different algorithms may require different preprocessing procedures.

For example, some models may benefit from standardized numerical variables, while tree-based models may not require scaling.

Therefore, the project does not apply one global transformation to the complete Feature Engineering dataset.

## Main Rule

The most important preprocessing rule established during this step is:

> Any transformation that learns information from the data must be fitted using training data only.

This applies to procedures such as:

- numerical scaling;
- missing-value imputation;
- categorical encoding;
- learned transformations.

Validation and test data must only be transformed using parameters learned from the corresponding training data.

## Model-Specific Preprocessing

The framework allows each model to use the preprocessing appropriate for its algorithm.

However, all models must respect the same anti-leakage policy.

## Result

The original `feature_dataset.parquet` remains unchanged by global model-specific transformations.

Preprocessing will instead occur inside each modeling workflow.

---

# 7. Step 08_04 – Common Evaluation Framework

## Objective

This step created the common evaluation structure that future models will use.

The purpose is to ensure that model performance is not evaluated using different procedures across algorithms.

## Forecasting Evaluation

Forecasting results must be evaluated:

- globally;
- by validation fold;
- by FSA;
- by forecast horizon.

This allows the project to determine not only which model performs best overall, but also whether performance changes across geographic areas or as the prediction horizon increases.

## Peak-Risk Evaluation

Peak-Risk models must report the official classification metrics using the same evaluation framework.

Probabilities and binary predictions are treated separately so that probability-based metrics such as PR-AUC can be evaluated independently of a specific classification threshold.

## Result

Reusable evaluation functions were established for future model branches.

This allows model developers to focus on the algorithm itself rather than implementing different evaluation methodologies.

---

# 8. Step 08_05 – Forecast Horizon Rules

## Objective

This step formally established how the 24-hour Forecasting horizon will be evaluated.

The project does not treat the Forecasting task as a single future value.

Instead, each forecast origin produces predictions for:

`h+1 ... h+24`

## Why Horizon-Level Evaluation Matters

Forecasting accuracy may change as the prediction horizon becomes longer.

For example, a model may perform very well at `h+1` but become less accurate at `h+18` or `h+24`.

Therefore, aggregate performance alone is not sufficient.

## Result

Every Forecasting model must preserve and report performance by forecast horizon in addition to global performance.

This allows direct comparison of how prediction accuracy changes throughout the next 24 hours.

---

# 9. Step 08_06 – Peak-Risk Rules

## Objective

This step established the common rules for the Peak-Risk classification problem.

## Peak Definition

Peak periods are based on the selected percentile criterion:

`97.5th percentile`

The threshold is calculated according to:

`FSA + season`

This prevents areas with substantially different electricity-demand levels from being evaluated using a single absolute threshold.

## Training-Only Threshold Rule

Peak thresholds must not be calculated once using the complete historical dataset.

Instead, they must be fitted independently inside each training window.

The resulting threshold is then applied unchanged to the corresponding validation or test observations.

This prevents future information from influencing the definition of Peak events.

## Multi-Horizon Peak-Risk

Peak-Risk follows the same 24-hour prediction structure as Forecasting.

For every forecast origin, the model will estimate Peak-Risk for:

`peak_h01 ... peak_h24`

Each target represents the Peak status of the corresponding FSA at the relevant future hour.

## Classification Probability Threshold

For the initial fair comparison of classifiers, a common probability threshold of:

`0.50`

was established.

This threshold converts predicted probabilities into initial binary Peak / No-Peak predictions.

Alternative operational thresholds may later be investigated using validation data only.

## Result

A complete 24-hour Peak-Risk policy was established and documented in:

`08_06_peak_horizon_policy.csv`

---

# 10. Step 08_07 – Shared Modeling Configuration

## Objective

This step reviewed the centralized configuration used by the Modeling Foundation.

Instead of defining important modeling parameters independently inside different scripts or notebooks, common settings are maintained in a shared configuration file.

## Configuration Includes

The shared configuration contains information such as:

- dataset paths;
- temporal columns;
- FSA grouping;
- Forecasting horizons;
- Peak-Risk horizons;
- evaluation metrics;
- model-selection metrics;
- temporal folds;
- final holdout period;
- preprocessing rules;
- Peak threshold rules;
- classification probability threshold;
- reproducibility settings.

## Result

Future model branches can reuse the same configuration rather than independently defining experimental conditions.

This improves consistency and reproducibility across the project.

---

# 11. Step 08_08 – Framework Validation

## Objective

Before allowing model development to begin, the complete Modeling Foundation was validated.

This step acts as the quality gate for Modeling Foundation Phase.

## Main Validations

The framework verifies that:

- training datasets are not empty;
- validation and test datasets are not empty;
- training occurs before evaluation;
- temporal periods do not overlap;
- FSA coverage is consistent;
- forecast origins are horizon-safe;
- all 24 Forecasting targets are available;
- the complete 24-hour target horizon remains inside the appropriate temporal period;
- the final 2025 holdout occurs after all validation periods;
- preprocessing is fitted using training data only;
- Forecasting and Peak-Risk use compatible horizons;
- the configured model-selection metrics are valid;
- the classification probability threshold is valid.

## Result

All framework validation checks passed.

No blocking issue remained after the final validation.

The validation results are stored in:

`08_08_framework_validation.csv`

---

# 12. Step 08_09 – Modeling Foundation Summary

## Objective

This notebook consolidates the main outputs produced during Modeling Foundation Phase.

It provides a final review of:

- temporal splits;
- FSA coverage;
- official metrics;
- Peak-Risk horizon policy;
- framework validation results.

## Result

The final documentation summarizes the common modeling contract that future model branches must follow.

The generated document is:

`docs/modeling_foundation/08_09_Modeling_Foundation_Summary.md`

This document serves as the technical summary of the final Modeling Foundation configuration.

---

# 13. Step 08_99 – Complete Modeling Foundation Execution

## Objective

The final notebook provides a single reproducible entry point for executing the complete Modeling Foundation workflow.

Instead of relying exclusively on manually executing individual notebooks, the complete framework can be run and validated from one place.

## Process

The complete execution:

1. loads the shared configuration;
2. loads the Feature Engineering and Target Definition outputs;
3. creates the common horizon-safe temporal splits;
4. verifies FSA coverage;
5. loads the official metrics;
6. creates the Peak-Risk multi-horizon policy;
7. performs the framework validation checks;
8. generates the reports;
9. regenerates the Modeling Foundation documentation;
10. applies a final validation gate.

If a required framework validation fails, the process does not silently continue as if the Modeling Foundation were valid.

## Result

The complete Modeling Foundation Phase workflow executed successfully.

This confirms that the shared framework is reproducible and ready to be used by the individual modeling branches.

---

# 14. Final Outputs of Modeling Foundation Phase

The main outputs produced by this phase include:

## Reports

`reports/modeling_foundation/`

- `08_01_common_split_summary.csv`
- `08_01_fold_fsa_coverage.csv`
- `08_02_metric_dictionary.csv`
- `08_06_peak_horizon_policy.csv`
- `08_08_framework_validation.csv`

## Documentation

`docs/modeling_foundation/`

- `08_00_Modeling_Foundation_Design.md`
- `08_09_Modeling_Foundation_Summary.md`
- `08_Modeling_Foundation_Overview.md`

The notebooks provide the interactive and reproducible analysis, while the reports and documentation preserve the final decisions and validation results.

---

# 15. Main Outcome of Modeling Foundation Phase

The most important result of Modeling Foundation Phase is not a predictive model.

The main result is a **common and reproducible experimental framework**.

Before this phase, the project had prepared features and defined the prediction targets.

After this phase, the project additionally has:

- common chronological data splits;
- horizon-safe forecast origins;
- a protected final 2025 holdout;
- common Forecasting targets;
- common Peak-Risk targets;
- common evaluation metrics;
- common model-selection criteria;
- common preprocessing principles;
- common anti-leakage rules;
- common Peak threshold methodology;
- common evaluation functions;
- centralized configuration;
- automated framework validation.

As a result, different models can now be developed independently while remaining directly comparable.

