"""Final feature list for modeling, decided in notebooks/02_forecasting/02_01_feature_engineering.ipynb.

Selection combined three checks: pairwise correlation to find near-duplicate features (including
raw-vs-cyclical calendar pairs, where correlation understates the redundancy since the relationship
is non-linear), gain-based importance from a quick baseline XGBoost fit to screen out the long tail
of features the model never found useful, and - for every close call the first two methods flagged -
a held-out validation check comparing MAE with vs without the feature. That last step mattered: gain
from an unvalidated fit turned out to be unreliable for judging correlated features specifically, it
said `hour_sin`/`hour_cos` were useful (they were not) and said `Dew Point Temp (°C)` was not useful
(it was).

The validation split is train-2021/validate-2022, not any of the team's 3 official expanding-window
folds (F1 test=2023, F2 test=2024, F3 test=2025). An earlier pass used train-2021-2024/validate-2025,
which is identical to Fold 3's split, using that to pick features would let Fold 3's later reported
performance benefit from having already "seen" its own test year during selection. 2021-2022 is the
only span that is never a test year in any fold, so it is the only safe choice for this kind of check.

Checked and confirmed on this split: `hour` alone matches `hour + hour_sin + hour_cos` at every tree
depth tested (4/6/10/15), `hour_sin`/`hour_cos` are a deterministic function of `hour` (zero new
information by construction, unlike a correlated-but-distinct measurement), so this is not surprising
in hindsight. `Dew Point Temp (°C)` still improves validation MAE when added back despite its 0.92
correlation with `Temp (°C)`, unlike `hour`/`hour_sin`/`hour_cos`, dew point is not a function of
temperature alone (it also depends on humidity), so the correlation is empirical, not exact, and it
carries real independent information. `month`/`quarter`/`week_of_year`/`day_of_year`, `is_weekend`,
and `Rel Hum (%)` all confirmed unhelpful again. The detailed holiday-distance features
(`days_to_holiday`, `days_after_holiday`) made validation MAE clearly worse when added, confirming
they are noise here, not just low-gain.

Two features gave inconsistent results between the 2021-2022 check and the (since-discarded)
2021-2024/2025 check: `weekday` (clear benefit on the larger split, negligible on this one) and
`Wind Dir (10s deg)` (hurt on the larger split, slightly helped on this one). Both differences are
small either way. Kept `weekday` (never hurt in either check) and dropped `Wind Dir (10s deg)`
(simpler model, no consistent case for it) - a judgment call on genuinely marginal, noisy evidence,
not a clean result like the others. Worth revisiting once the real 3-fold evaluation exists.

`is_public_holiday` is kept as a discretionary call: gain is low (0.0007) and its validation effect
was small and inconsistent in direction across the two checks, but it is a single column, the EDA
found a real (if modest) holiday effect, and it never showed a meaningful downside, kept for
interpretability.

Dropped as unimportant, not re-validated individually (gain too low to be worth the extra check):
`Stn Press (kPa)`, the three rare DST transition flags, `is_day_before_holiday`, `is_day_after_holiday`.
"""

SELECTED_FEATURES = [
    "FSA",
    "season",
    "hour",
    "weekday",
    "is_workday",
    "is_business_hour",
    "is_peak_hour_window",
    "is_long_weekend",
    "is_public_holiday",
    "is_daylight_saving_time",
    "month_cos",
    "year",
    "PREMISE_COUNT",
    "Temp (°C)",
    "Dew Point Temp (°C)",
    "Precip. Amount (mm)",
    "Wind Spd (km/h)",
    "Visibility (km)",
]
