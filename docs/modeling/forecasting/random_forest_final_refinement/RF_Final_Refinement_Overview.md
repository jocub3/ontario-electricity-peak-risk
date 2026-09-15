# Random Forest Regressor --- Final Controlled Refinement Overview

## 1. Purpose

After the model-comparison phase selected **Random Forest Regressor** as
the Forecasting finalist, a small and controlled refinement was
performed before the final 2025 holdout evaluation.

The objective was **not** to restart hyperparameter tuning or conduct a
broad search. Instead, the refinement tested a small set of nearby
configurations to determine whether the already-selected Random Forest
configuration could be improved enough to justify changing it.

The 2025 final holdout was not used during this process.

------------------------------------------------------------------------

## 2. Incumbent Configuration

The Random Forest configuration entering the refinement stage was:

  |Hyperparameter      |   Value|
  |:-------------------|-------:|
  |`n_estimators`      |     160|
  |`max_depth`         |    None|
  |`min_samples_leaf`  |       5|
  |`max_features`      |     0.7|

The existing feature set, preprocessing pipeline, target definition,
temporal folds, and official evaluation metrics were kept unchanged.

------------------------------------------------------------------------

## 3. Refinement Design

Four configurations were evaluated:

  
  |ID          |Description              | `n_estimators` |  `min_samples_leaf` |   `max_features`|
  |:-----------|:------------------------|---------------:|--------------------:|----------------:|
  |R0          |Incumbent                |             160|                    5|              0.7|
  |R1          |More trees               |             220|                    5|              0.7|
  |R2          |Smaller leaf             |             180|                    3|              0.7|
  |R3          |Broader feature sampling |             180|                    5|              0.8|
                                                           

All configurations retained `max_depth=None`.

To keep the refinement computationally controlled, the comparison used
the existing development folds and three representative forecast
horizons:

-   `h+1`
-   `h+12`
-   `h+24`

This produced 24 model fits in total: four configurations × three
representative horizons × two validation folds.

------------------------------------------------------------------------

## 4. Selection Rule

**MAE** remained the primary selection metric.

A challenger was allowed to replace the incumbent only if it reduced
mean MAE by at least **0.25%**. This minimum-improvement rule was
included to avoid changing an already validated model because of a very
small difference that may not be practically meaningful.

RMSE, MAPE, fold variability, and horizon variability were also reviewed
as supporting indicators.

------------------------------------------------------------------------

## 5. Refinement Results

### Aggregate comparison

  
  |Configuration                   |  Mean MAE    |  Mean RMSE   |   Mean MAPE | MAE improvement vs incumbent |
  |:-------------------------------|-------------:|-------------:|------------:|------------------------------|
  |R1 --- More trees               | **527.93**   |  **896.49**  |   **5.404%**|    **+0.150%**               |
  |R0 --- Incumbent                |  528.72      |   898.01     |    5.411%   |      0.000%                  |
  |R2 --- Smaller leaf             | 528.77       |  899.22      |   5.405%    |    -0.009%                   |
  |R3 --- Broader feature sampling |    531.51    |     902.83   |      5.435% |       -0.528%                |
                                                        
                                                       
  ---------------------------------------------------------------------------

R1 obtained the numerically lowest MAE, RMSE, and MAPE among the tested
configurations. However, its MAE improvement over the incumbent was only
approximately **0.15%**, below the predefined **0.25%** threshold
required to justify replacing the existing model.

R2 produced essentially the same average MAE as the incumbent but showed
greater variability. R3 was worse than the incumbent on the main
aggregate metrics.

------------------------------------------------------------------------

## 6. Behavior Across Folds and Horizons

The refinement confirmed the same general pattern already observed in
the Random Forest candidate evaluation:

-   forecasting error is lower in the 2024 validation fold than in 2023;
-   `h+1` is substantially easier to predict than `h+12` and `h+24`;
-   the local hyperparameter changes did not materially alter this
    behavior.

For the incumbent configuration, representative MAE values were:

  |Fold   |     h+1 |    h+12 |    h+24 |
  |:------|--------:|--------:|--------:|
  |2023   |  511.78 |  588.99 |  597.81 |
  |2024   |  366.45 |  538.29 |  569.02 |

This indicates that forecast uncertainty increases with the forecast
horizon, while the 2024 period remains easier for the model than 2023.

A negative bias was present across all six incumbent fold/horizon
combinations, indicating a tendency toward **underprediction**. This is
not a new issue introduced by the refinement; it is a characteristic
that should continue to be monitored in the final holdout evaluation.

------------------------------------------------------------------------

## 7. Final Decision

The final refinement decision was:

> **KEEP INCUMBENT**

Although increasing the number of trees from 160 to 220 produced a small
numerical improvement, the reduction in mean MAE was only approximately
**0.15%**. This did not exceed the predefined 0.25% minimum improvement
required to justify changing the already-selected model.

Therefore, the final Forecasting configuration remains:

``` text
RandomForestRegressor
n_estimators = 160
max_depth = None
min_samples_leaf = 5
max_features = 0.7
```

This result suggests that the selected Random Forest configuration is
already located in a relatively stable local region: modest nearby
hyperparameter changes do not materially improve performance.

------------------------------------------------------------------------

## 8. Freeze Status

The Forecasting model is now frozen for the final holdout stage with:

-   Algorithm: **RandomForestRegressor**
-   Forecast horizons: **24**
-   Hyperparameters: **160 trees / unlimited depth / leaf size 5 /
    max_features 0.7**
-   Feature set: **unchanged**
-   Preprocessing: **unchanged**
-   Development folds used for refinement: **Yes**
-   Final 2025 holdout used for refinement: **No**

The freeze manifest records the selected configuration and the hashes of
the original and refinement configurations for reproducibility.

------------------------------------------------------------------------

## 9. Conclusion

The controlled refinement did not find a sufficiently large improvement
to justify replacing the existing Random Forest configuration.

This is a positive result rather than a failed tuning exercise: it
provides additional evidence that the previously selected model is
reasonably stable around its current hyperparameters and avoids adding
computational complexity for a negligible gain.

**Final Forecasting configuration before 2025 holdout: Random Forest
Regressor with the incumbent hyperparameters.**

No additional hyperparameter tuning is recommended before the final 2025
holdout evaluation. The next stage should evaluate this frozen
configuration on 2025 without modifying the model based on the holdout
result.
