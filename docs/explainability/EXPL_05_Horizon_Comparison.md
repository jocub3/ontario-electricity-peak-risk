# EXPL_05 — Horizon Comparison

## Purpose

This analysis evaluates the stability of SHAP feature-importance rankings
across the representative horizons h+1, h+6, h+12, h+18, and h+24.

## Results

### Random Forest

- Mean Spearman correlation: 0.812
- Minimum correlation: 0.621
- Maximum correlation: 0.933

The Random Forest explanations remain broadly consistent, but similarity
decreases as the distance between forecast horizons increases. The largest
difference occurs between h+1 and h+24.

### XGBoost

- Mean Spearman correlation: 0.903
- Minimum correlation: 0.832
- Maximum correlation: 0.985

XGBoost shows high explanatory stability across the representative horizons.

## Interpretation

The results support the use of representative horizons for the SHAP analysis.
They also indicate that forecast-horizon effects are more pronounced for the
Random Forest demand model than for the XGBoost Peak-Risk classifier.

These results describe model-behavior stability and should not be interpreted
causally.
