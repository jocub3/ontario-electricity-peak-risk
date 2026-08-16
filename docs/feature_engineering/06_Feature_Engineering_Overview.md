# Feature Engineering Phase - Overview

## Purpose

The Feature Engineering Phase prepared the cleaned project data for the subsequent modeling
stages.

The main objective was to transform the information already understood
during Exploratory Data Analysis (EDA) into a structured set of
variables that can be used by the Forecasting and Peak-Risk models. This
phase did not train or compare predictive models. Instead, it created,
organized, documented, and validated the information that those models
may use later.

A key principle throughout this phase was to preserve the temporal
nature of the project and avoid using information that would not be
available at the time a prediction is made.

------------------------------------------------------------------------

## Phase Workflow

The Feature Engineering phase was organized into a sequence of modular
notebooks. Each notebook addresses a specific part of the preparation
process and contributes to the final modeling dataset.

### `06_00_feature_engineering_design.ipynb` -- Feature Engineering Design

**Purpose:** Define the Feature Engineering strategy before creating new
variables.

This step established the overall design of the phase, including the
types of variables that could be useful for Forecasting and Peak-Risk,
which information could be shared by both modeling tasks, and which
variables require special treatment to avoid using future information.

**Result:** A documented plan for creating and managing features
consistently across the project.

------------------------------------------------------------------------

### `06_01_feature_dataset.ipynb` -- Baseline Feature Dataset

**Purpose:** Establish the starting point for Feature Engineering.

The cleaned master dataset was reviewed and organized before new
variables were introduced. Existing variables were classified according
to their role, while administrative, traceability, target, spatial, and
potentially useful predictor information remained identifiable.

**Result:** A controlled baseline dataset that provides the foundation
for the remaining Feature Engineering steps.

------------------------------------------------------------------------

### `06_02_temporal_features.ipynb` -- Temporal Features

**Purpose:** Represent recurring time patterns more effectively.

Electricity consumption changes according to the hour of the day, day of
the week, month, season, holidays, and other calendar conditions. This
step prepared temporal information so that future models can better
recognize these recurring patterns.

It also created cyclical representations of time, allowing the models to
understand that periods such as the end and beginning of a day, week, or
year are naturally connected.

**Result:** A richer representation of the temporal structure of
electricity consumption.

------------------------------------------------------------------------

### `06_03_lag_features.ipynb` -- Historical Consumption Features

**Purpose:** Provide information about previous electricity consumption.

Electricity demand is strongly related to its own recent and historical
behavior. This step created references to consumption observed at
previous points in time, including recent hours and comparable periods
from previous days and weeks.

**Result:** Historical demand information that can help future models
recognize persistence and recurring consumption patterns.

------------------------------------------------------------------------

### `06_04_rolling_features.ipynb` -- Recent Consumption Behavior

**Purpose:** Summarize how electricity consumption has behaved over
recent periods.

Instead of considering only individual historical observations, this
step created summaries of recent demand behavior. These variables
describe characteristics such as the recent average level, variability,
minimum, and maximum consumption.

**Result:** Context about whether recent electricity demand has been
stable, increasing, variable, high, or low.

------------------------------------------------------------------------

### `06_05_weather_features.ipynb` -- Weather Features

**Purpose:** Improve the representation of meteorological conditions.

The EDA indicated that weather conditions are relevant to electricity
consumption. This step extended the original weather information by
representing changes and recent patterns in variables such as
temperature and humidity, as well as selected non-linear weather
relationships.

**Result:** Weather information that better represents both current
conditions and recent meteorological behavior.

------------------------------------------------------------------------

### `06_06_interaction_features.ipynb` -- Interaction Features

**Purpose:** Represent selected relationships between variables.

Some factors may affect electricity demand differently depending on
other conditions. This step created a limited number of combined
variables justified by the previous analysis rather than generating a
large number of arbitrary combinations.

**Result:** Additional information representing selected relationships
that may not be fully captured by individual variables alone.

------------------------------------------------------------------------

### `06_07_peak_feature_policy.ipynb` -- Peak-Risk Feature Policy

**Purpose:** Define how Peak-Risk information must be handled safely
during modeling.

The final Peak-Risk target was intentionally not created globally during
this phase. Peak thresholds and labels depend on the historical
information available within each training period. Creating them using
the complete dataset could introduce future information into model
development.

This step therefore established the policy that Peak-Risk targets and
related historical information must be generated within the appropriate
training windows during the modeling phase.

**Result:** A documented Peak-Risk methodology designed to prevent data
leakage.

------------------------------------------------------------------------

### `06_08_feature_selection.ipynb` -- Preliminary Feature Selection

**Purpose:** Organize the available variables according to their
expected usefulness and role.

Features were reviewed and classified into preliminary priority groups.
Administrative, redundant, spatial, target-related, and potentially
useful predictor variables were also identified.

This was not a final decision about which variables each model will use.

**Result:** A preliminary feature inventory that will guide
model-specific feature selection in later phases.

------------------------------------------------------------------------

### `06_09_multicollinearity.ipynb` -- Multicollinearity Analysis

**Purpose:** Identify variables that may contain very similar or
redundant information.

Relationships among numerical variables were examined to identify cases
where multiple features may be representing the same underlying
information.

The analysis was used as a diagnostic tool. Variables were not
automatically removed solely because they showed high multicollinearity.

**Result:** Documentation of potentially redundant relationships that
can be considered when developing individual models.

------------------------------------------------------------------------

### `06_10_feature_validation.ipynb` -- Feature Validation

**Purpose:** Verify the integrity of the engineered dataset.

After creating the new variables, this step checked that the Feature
Engineering process had preserved the structure of the data and that the
generated variables were correctly aligned with historical information.

The validation also reviewed unexpected missing values, invalid
numerical values, temporal alignment, lag calculations, rolling
calculations, and key safeguards against data leakage.

**Result:** Confirmation that the engineered dataset is structurally
suitable for the next stages, subject to the modeling-time restrictions
already documented.

------------------------------------------------------------------------

### `06_11_feature_summary.ipynb` -- Feature Engineering Summary

**Purpose:** Consolidate and document the outputs of the entire phase.

The final step summarized the feature families, preliminary priorities,
validation results, and important restrictions that must be respected
during modeling.

**Result:** A consolidated Feature Engineering summary and the final
dataset produced by this phase.

------------------------------------------------------------------------

## Main Output

The principal data product of The Feature Engineering Phase is:

`feature_dataset.parquet`

This dataset contains the existing project information together with the
engineered variables created during the phase. It will serve as the
common data source for the subsequent Forecasting and Peak-Risk modeling
work.

However, inclusion in this dataset does not automatically mean that
every variable can be used by every model or at every forecast horizon.

------------------------------------------------------------------------

## Important Modeling Considerations

Several decisions are intentionally deferred to the modeling stage.

The final predictor set must be selected separately for each candidate
model. Historical demand variables must also be evaluated according to
the forecast horizon because information available for a near-term
prediction may not be directly available for all 24 future hours.

Similarly, Peak-Risk thresholds and labels must be calculated using only
the historical information available within each training window.

Missing-value treatment and other transformations that depend on the
observed data must also be fitted using training data only.

Finally, when weather information is used for future predictions, the
modeling process must distinguish between historically observed weather
and weather information that would actually be available at forecast
time.

------------------------------------------------------------------------

## Feature Engineering Phase Outcome

The Feature Engineering Phase converted the cleaned and explored project data into a
structured and validated feature dataset suitable for model development.

The overall progression can be summarized as:

**Clean Dataset → Feature Design → Temporal Information → Historical
Demand → Recent Demand Behavior → Weather Information → Selected
Interactions → Peak-Risk Policy → Preliminary Feature Review →
Multicollinearity Analysis → Feature Validation → Final Feature
Dataset**

This phase provides the bridge between understanding the data and
building predictive models.

The previous EDA phase answered:

> **What patterns and relationships exist in the data?**

Feature Engineering addressed:

> **How should those patterns and relationships be represented so that
> predictive models can use them effectively?**

The following modeling phases will address:

> **Which models and feature combinations provide the most accurate and
> reliable 24-hour electricity demand forecasts and Peak-Risk
> predictions?**
