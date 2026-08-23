# Exploratory Data Analysis

## Purpose

1. Distributions, outliers, trend, seasonality, the holiday effect, and the consumption-weather relationship are explored on the cleaned, joined dataset.
2. Calendar coverage (season, weekday, hour) is confirmed to be a complete, balanced hourly grid before it is relied on later.

## Design

1. `src/common/eda.py` holds reusable helpers: `plot_distribution()`, `plot_boxplot()`, `iqr_outlier_summary()`, `average_profile()`/`plot_average_profile()`, and `plot_time_series()`.
2. The notebook reads `fsa_hourly_master.parquet` directly, no rebuilding from the raw files is needed here.
3. IQR outliers for consumption are computed per FSA and per season, not pooled, since a spike that is normal in winter could be an outlier in summer.
4. Weather is checked for physically implausible values rather than statistical outliers: relative humidity over 100%, negative precipitation, wind speed, or visibility, and zero or negative pressure. `Temp` and `Dew Point Temp` are excluded from this check, negative values are normal in an Ontario winter.
5. Seasonality is analyzed at three levels (month, weekday, hour), each pooled across all 5 years, to isolate the repeating seasonal shape from any year-over-year trend.
6. Pearson and Spearman correlation are shown side by side rather than Kendall's tau, which answers the same question at a much higher computational cost. Both are noted as unable to fully capture a non-monotonic relationship like temperature's.

## Findings

1. `TOTAL_CONSUMPTION` is right-skewed in all six FSAs, supporting the log-transform used later, and the six FSAs differ hugely in scale (tied to `PREMISE_COUNT`), confirming `fsa` needs to be a categorical feature in the pooled model. IQR outliers only ever appear above the upper whisker, never below, matching that same right-skew.
2. `PREMISE_COUNT` changes in discrete steps rather than growing organically. Tracing one step (2022-12-15) back to the raw consumption file shows a brand new `CUSTOMER_TYPE`/`PRICE_PLAN` combination starting to appear for both M9W and M9R on that exact date, the jump is a reporting change, not new customers.
3. No zero or negative consumption values exist, and no weather variable has a physically impossible value.
4. No long-term trend distinct from the seasonal pattern is visible, and no data gaps or level shifts remain from cleaning.
5. Seasonality shows a clear annual "U" shape (highest in July, lowest in the May/October shoulder months), a small weekend effect (about 4% higher than workdays, consistent with residential FSAs), and a single broad evening peak each day rather than two separate commute peaks. Weekday and weekend hourly shapes mainly differ in the middle of the day.
6. Public holidays show a small drop in average consumption (8,624 versus 8,797 kWh), a modest effect.
7. Consumption against temperature is an asymmetric valley, not a symmetric U: a minimum around 15°C, a gradual decline on the cold side, and a much sharper, near-exponential rise on the hot side, roughly tripling between 15°C and 36°C.
8. Pearson and Spearman correlation agree that `PREMISE_COUNT` and the hour-of-day features are the strongest predictors. `Temp` stays weak under both measures despite the clear valley shape found directly, confirming that correlation coefficients miss non-monotonic relationships and a model-based feature importance is needed later to evaluate temperature properly.
9. `Visibility` has a reporting artifact: the 25th, 50th, and 75th percentiles are all exactly 24.1 km, most likely a sensor ceiling rather than a real physical distribution.
10. `Temp` and `Dew Point Temp` are very strongly correlated with each other, a candidate for dropping one of them later rather than feeding the model two near-duplicate signals.
