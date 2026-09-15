"""Adapter around the existing Random Forest forecasting candidate implementation.

This refinement reuses the exact candidate pipeline already validated in the
Random Forest branch. It does not rebuild feature engineering, preprocessing,
folds, or metrics.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pandas as pd
import yaml


CANDIDATE_RUNNER_MODULE = (
    "src.ontario_peak_risk.models.forecasting.random_forest.run_model"
)


def resolve_candidate_runner():
    """Import the existing Random Forest candidate runner."""
    try:
        module = importlib.import_module(CANDIDATE_RUNNER_MODULE)
        return module, CANDIDATE_RUNNER_MODULE
    except Exception as exc:
        raise ImportError(
            "Could not import the existing Random Forest candidate runner: "
            f"{CANDIDATE_RUNNER_MODULE}. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_capabilities() -> pd.DataFrame:
    """Report whether the exact functions used by the original branch exist."""
    try:
        module, module_name = resolve_candidate_runner()
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "candidate_module": None,
                    "import_ok": False,
                    "has_tune": False,
                    "has_run_model": False,
                    "tune_function": None,
                    "run_function": None,
                    "message": str(exc),
                }
            ]
        )

    has_tune = hasattr(module, "_tune_parameters")
    has_run_model = hasattr(module, "run_random_forest_regressor")

    return pd.DataFrame(
        [
            {
                "candidate_module": module_name,
                "import_ok": True,
                "has_tune": has_tune,
                "has_run_model": has_run_model,
                "tune_function": (
                    "_tune_parameters" if has_tune else None
                ),
                "run_function": (
                    "run_random_forest_regressor"
                    if has_run_model
                    else None
                ),
                "message": (
                    "Existing candidate implementation is compatible "
                    "with the final-refinement adapter."
                    if has_tune and has_run_model
                    else
                    "Candidate module imported, but expected original "
                    "Random Forest functions were not found."
                ),
            }
        ]
    )


def call_candidate_tune(
    *,
    project_root: Path,
    original_config_path: Path,
    refinement_parameter_grid: list[dict],
    representative_horizons: list[int],
) -> pd.DataFrame:
    """Run only the original candidate's tuning function on a small local grid."""
    module, _ = resolve_candidate_runner()

    if not hasattr(module, "_tune_parameters"):
        raise AttributeError(
            "The original Random Forest runner does not expose "
            "`_tune_parameters`."
        )

    original_cfg = load_yaml(original_config_path)
    branch_cfg = copy.deepcopy(original_cfg)

    branch_cfg["model"]["tuning"]["representative_horizons"] = [
        int(h) for h in representative_horizons
    ]

    cleaned_grid = []
    refinement_ids = []

    for row in refinement_parameter_grid:
        item = dict(row)
        refinement_ids.append(item.pop("refinement_id"))
        cleaned_grid.append(item)

    branch_cfg["model"]["tuning"]["parameter_grid"] = cleaned_grid

    # Reuse exactly the same shared loaders and feature-column policy
    # imported by the original Random Forest candidate runner.
    foundation_path = (
        project_root
        / branch_cfg["paths"]["modeling_foundation_config"]
    )

    modeling_config, _ = module.load_modeling_config(
        foundation_path
    )

    feature_dataset = module.load_feature_dataset(
        modeling_config,
        project_root,
    )

    forecast_targets = module.load_forecast_targets(
        modeling_config,
        project_root,
    )

    numeric_columns, categorical_columns = module.model_feature_columns(
        feature_dataset
    )

    results = module._tune_parameters(
        feature_dataset,
        forecast_targets,
        modeling_config,
        branch_cfg,
        numeric_columns,
        categorical_columns,
    )

    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    if "parameter_id" not in results.columns:
        raise ValueError(
            "Original tuning output does not contain `parameter_id`."
        )

    id_map = {
        index + 1: refinement_id
        for index, refinement_id in enumerate(refinement_ids)
    }

    results["refinement_id"] = (
        results["parameter_id"]
        .astype(int)
        .map(id_map)
    )

    return results
