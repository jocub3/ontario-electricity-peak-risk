"""Run the small controlled Random Forest forecasting refinement."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .adapter import (
    call_candidate_tune,
    candidate_capabilities,
)
from .freeze import (
    build_manifest,
    save_manifest,
)
from .selection import (
    select_final_configuration,
    summarize_refinement,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_refinement(
    config_path: str | Path = "configs/random_forest_final_refinement.yaml",
) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if bool(config["phase"].get("evaluate_final_holdout", False)):
        raise ValueError(
            "Random Forest refinement must not evaluate the final 2025 holdout."
        )

    original_config_path = _resolve(
        project_root,
        config["paths"]["original_model_config"],
    )

    if not original_config_path.exists():
        raise FileNotFoundError(
            f"Original Random Forest config not found: {original_config_path}"
        )

    capabilities = candidate_capabilities()

    if (
        capabilities.empty
        or not bool(capabilities.iloc[0]["import_ok"])
        or not bool(capabilities.iloc[0]["has_tune"])
    ):
        raise RuntimeError(
            "The existing Random Forest candidate pipeline could not be "
            "reused safely. Run RFFR_01_input_validation.ipynb and share "
            "the result / original run_model.py before proceeding."
        )

    reports_dir = _resolve(
        project_root,
        config["paths"]["reports_dir"],
    )
    outputs_dir = _resolve(
        project_root,
        config["paths"]["outputs_dir"],
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    details = call_candidate_tune(
        project_root=project_root,
        original_config_path=original_config_path,
        refinement_parameter_grid=config["refinement"]["parameter_grid"],
        representative_horizons=config["refinement"]["representative_horizons"],
    )

    summary = summarize_refinement(details)

    selected, ranking = select_final_configuration(
        summary,
        incumbent_id="R0_incumbent",
        minimum_improvement_pct=float(
            config["refinement"][
                "minimum_mae_improvement_pct_to_replace_incumbent"
            ]
        ),
    )

    parameter_lookup = {
        row["refinement_id"]: {
            key: value
            for key, value in row.items()
            if key != "refinement_id"
        }
        for row in config["refinement"]["parameter_grid"]
    }

    selected_parameters = parameter_lookup[
        str(selected["refinement_id"])
    ]

    selection_evidence = {
        "selected_refinement_id": str(selected["refinement_id"]),
        "mean_mae": float(selected["mean_mae"]),
        "minimum_improvement_pct_required": float(
            config["refinement"][
                "minimum_mae_improvement_pct_to_replace_incumbent"
            ]
        ),
        "decision": str(selected["decision"]),
        "representative_horizons": [
            int(value)
            for value in config["refinement"]["representative_horizons"]
        ],
        "development_folds_only": True,
    }

    manifest = build_manifest(
        selected_parameters=selected_parameters,
        selection_evidence=selection_evidence,
        original_config_path=original_config_path,
        refinement_config_path=config_path,
        freeze_config=config["freeze"],
    )

    manifest_path = save_manifest(
        manifest,
        outputs_dir / "RF_FORECASTING_FINAL_FREEZE_MANIFEST.json",
    )

    details.to_csv(
        reports_dir / "RFFR_02_refinement_details.csv",
        index=False,
    )
    ranking.to_csv(
        reports_dir / "RFFR_03_refinement_ranking.csv",
        index=False,
    )

    (
        outputs_dir / "selected_random_forest_parameters.json"
    ).write_text(
        json.dumps(selected_parameters, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("RANDOM FOREST FORECASTING FINAL REFINEMENT COMPLETED")
    print("Selected refinement:", selected["refinement_id"])
    print("Selected parameters:", selected_parameters)
    print("Decision:", selected["decision"])
    print(f"Freeze manifest: {manifest_path}")
    print("2025 final holdout evaluated: NO")
    print("=" * 72)

    return {
        "capabilities": capabilities,
        "details": details,
        "summary": summary,
        "ranking": ranking,
        "selected": selected,
        "selected_parameters": selected_parameters,
        "freeze_manifest": manifest,
    }
