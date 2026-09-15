# Integrated Prediction Pipeline --- Overview

## Purpose

The Integrated Prediction Pipeline combines the final Forecasting and
Peak-Risk model artifacts into one operational prediction flow.
Internally, the two model families remain independent, but their outputs
are joined into one standardized 24-hour result.

## Validation and artifact checks

The integration phase confirmed that all final artifacts required for
inference are available:

-   24 Random Forest forecasting models;
-   24 XGBoost Peak-Risk models;
-   Random Forest metadata;
-   XGBoost metadata;
-   final Peak threshold table.

The configured prediction horizon is 24 hours and the frozen operational
Peak-Risk threshold is 0.06.

## Historical replay demonstration

An end-to-end historical replay was executed using forecast origin
**2025-12-30 23:00:00** and the six project FSAs.

The pipeline generated exactly **144 rows**:

-   6 FSAs × 24 forecast horizons;
-   horizons h+1 through h+24;
-   one electricity-demand forecast;
-   one Peak-Risk score;
-   one Peak alert per FSA/horizon.

The replay produced **50 Peak alerts** across the 144 FSA-hour
combinations.

Maximum forecast consumption in the demonstration ranged from
approximately 6,963 kWh for M5S to 18,805 kWh for M9W. The maximum
Peak-Risk scores exceeded 0.95 in five of the six FSAs, while L4T
reached approximately 0.416.

This replay is a pipeline smoke test and demonstration, not another
model evaluation. Its results must not be used to retune the final
models.

## Generated outputs

The pipeline generated reusable prediction outputs in CSV and Parquet
formats, together with summary tables by FSA and an executive-style
summary.

The visualization stage generated:

-   24-hour demand heatmap;
-   Peak-Risk score heatmap;
-   binary Peak-alert heatmap;
-   demand forecast lines with Peak markers;
-   Peak-Risk score lines with the operational threshold;
-   maximum forecast demand by FSA;
-   Peak-alert count by FSA.

These outputs demonstrate that the final model artifacts can be loaded
and executed behind one logical interface.

## Important architectural clarification

The current integration is a **parallel two-model architecture**, not a
sequential forecasting-to-classification architecture.

For each horizon, both Random Forest and XGBoost receive
model-compatible features derived from the same forecast origin and
input feature dataset:

``` text
Input features
     ├──> Random Forest ──> Forecast consumption
     └──> XGBoost ────────> Peak-Risk probability ──> Peak alert
                                  threshold = 0.06
```

The Random Forest forecast is therefore **not currently passed as an
input feature to XGBoost**.

The two outputs are combined after prediction because they refer to the
same FSA, forecast origin, target timestamp, and horizon.

This design is consistent with the way the Peak-Risk classifier was
trained. Passing Random Forest forecasts into XGBoost now would change
the classifier's feature space and would require redesigning,
retraining, validating, comparing, and freezing a new Peak-Risk model.

## Current limitation

The historical replay uses an already engineered feature dataset.
Operational forecasting for a genuinely future date still requires an
**Inference Feature Builder**.

That component will convert raw operational inputs---recent demand,
future weather forecasts, forecast date/time, calendar information, and
FSA---into exactly the same leakage-safe feature schema used during
training.


