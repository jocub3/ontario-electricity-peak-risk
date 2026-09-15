# Final Training

After the final 2025 holdout has been documented, the frozen configurations
are trained on all available historical data.

The phase produces 24 serialized Random Forest pipelines and 24 serialized
XGBoost pipelines, one per forecast horizon, plus the final Peak threshold
table and metadata required for reproducible inference.

The holdout results are not used to modify any model-development decision.
