# WSA_00 — Weather Sensitivity Design and Scope

## Objective
Evaluate how frozen Model v1 predictions respond to controlled perturbations of `origin__Temp (°C)` while all other model features remain fixed.

## Scenarios
-5°C, -2.5°C, Baseline, +2.5°C, +5°C

## Interpretation boundary
This is a model sensitivity / what-if analysis, not a causal estimate and not a future target-hour weather experiment. Model v1 only contains forecast-origin weather. Models are not retrained.

## OOD policy
Scenario temperatures are labelled NORMAL, CAUTION, or OUT_OF_RANGE using the model-development feature dataset. Predictions may be retained with warnings; they are not treated as equally reliable outside training support.
