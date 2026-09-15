# Integrated Prediction Pipeline

The integrated pipeline loads the final serialized model artifacts and returns
one standardized 24-hour table containing, for each FSA and horizon:

- forecast electricity consumption;
- Peak-Risk score;
- Peak alert using threshold 0.06.

The pipeline is intentionally separate from the future user interface.

A future Streamlit/Power BI decision-support layer can consume the integrated
CSV/Parquet output and add:

- demand heatmaps;
- Peak-Risk heatmaps;
- 24-hour line charts with Peak markers;
- executive KPIs;
- weather-sensitivity scenarios (-5, -2.5, baseline, +2.5, +5 °C).

Operational inference requires a dedicated input/feature-builder layer that
constructs exactly the same model features from recent demand, calendar data,
FSA, and forecast-available weather. Raw future observed weather must never be
used as if it were known at prediction time.
