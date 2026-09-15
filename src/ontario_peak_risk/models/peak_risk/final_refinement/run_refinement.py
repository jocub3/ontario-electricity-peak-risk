"""Run the XGBoost Peak-Risk threshold refinement and freeze phase."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .analysis import (
    evaluate_selected_threshold,
    fold_threshold_summary,
    local_threshold_sensitivity,
    m9w_stress_sweep,
    select_operational_threshold,
    threshold_grid,
    threshold_sweep,
)
from .freeze import (
    build_freeze_manifest,
    save_freeze_manifest,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_final_refinement(
    config_path: str | Path = "configs/peak_risk_final_refinement.yaml",
) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if bool(config["phase"].get("evaluate_final_holdout", False)):
        raise ValueError(
            "This refinement phase must not evaluate the 2025 final holdout."
        )

    prediction_path = _resolve(
        project_root,
        config["paths"]["validation_predictions"],
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {prediction_path}"
        )

    predictions = pd.read_parquet(prediction_path)

    required = {
        "fold",
        "fsa",
        "horizon",
        "actual",
        "probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(
            f"Validation predictions are missing required columns: {missing}"
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

    thresholds = threshold_grid(
        config["threshold_analysis"]["min_threshold"],
        config["threshold_analysis"]["max_threshold"],
        config["threshold_analysis"]["step"],
    )

    global_sweep = threshold_sweep(
        predictions,
        thresholds,
    )
    fold_sweep = threshold_sweep(
        predictions,
        thresholds,
        ["fold"],
    )

    fold_summary = fold_threshold_summary(
        fold_sweep
    )

    selected, feasible_ranked = select_operational_threshold(
        fold_summary,
        minimum_recall_each_fold=float(
            config["threshold_analysis"]["minimum_recall_each_fold"]
        ),
        minimum_precision_each_fold=float(
            config["threshold_analysis"]["minimum_precision_each_fold"]
        ),
    )

    selected_threshold = float(selected["threshold"])

    selected_evaluation = evaluate_selected_threshold(
        predictions,
        selected_threshold,
    )

    local_sensitivity = local_threshold_sensitivity(
        fold_summary,
        selected_threshold,
        float(config["threshold_analysis"]["local_window"]),
    )

    stress = m9w_stress_sweep(
        predictions,
        thresholds,
        fold=config["stress_case"]["fold"],
        fsa=config["stress_case"]["fsa"],
    )

    selected_stress = stress.loc[
        stress["threshold"].eq(selected_threshold)
    ].copy()

    threshold_evidence = {
        "selected_threshold": selected_threshold,
        "selection_rule": (
            "Thresholds must satisfy configured minimum Recall and Precision "
            "in every development fold; among feasible thresholds maximize "
            "worst-fold F1, then worst-fold Recall, mean F1, mean Precision."
        ),
        "minimum_recall_each_fold": float(
            config["threshold_analysis"]["minimum_recall_each_fold"]
        ),
        "minimum_precision_each_fold": float(
            config["threshold_analysis"]["minimum_precision_each_fold"]
        ),
        "selected_summary": {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value in selected.to_dict().items()
        },
    }

    manifest = build_freeze_manifest(
        project_root=project_root,
        config=config,
        selected_threshold=selected_threshold,
        threshold_evidence=threshold_evidence,
    )

    manifest_path = save_freeze_manifest(
        manifest,
        outputs_dir / "XGBC_FINAL_FREEZE_MANIFEST.json",
    )

    _save(
        global_sweep,
        reports_dir / "XGBC_FR_02_global_threshold_sweep.csv",
    )
    _save(
        fold_sweep,
        reports_dir / "XGBC_FR_03_fold_threshold_sweep.csv",
    )
    _save(
        fold_summary,
        reports_dir / "XGBC_FR_03_fold_threshold_summary.csv",
    )
    _save(
        feasible_ranked,
        reports_dir / "XGBC_FR_04_feasible_threshold_ranking.csv",
    )
    _save(
        local_sensitivity,
        reports_dir / "XGBC_FR_04_local_threshold_sensitivity.csv",
    )
    _save(
        stress,
        reports_dir / "XGBC_FR_05_m9w_2023_threshold_sweep.csv",
    )
    _save(
        selected_stress,
        reports_dir / "XGBC_FR_05_m9w_2023_selected_threshold.csv",
    )

    for name, frame in selected_evaluation.items():
        _save(
            frame,
            reports_dir / f"XGBC_FR_05_selected_{name}_metrics.csv",
        )

    (
        outputs_dir / "selected_operational_threshold.json"
    ).write_text(
        json.dumps(
            threshold_evidence,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("XGBOOST PEAK-RISK FINAL REFINEMENT COMPLETED")
    print(f"Selected operational threshold: {selected_threshold:.2f}")
    print("Algorithm / features / hyperparameters: unchanged")
    print(f"Freeze manifest: {manifest_path}")
    print("2025 final holdout evaluated: NO")
    print("=" * 72)

    return {
        "predictions": predictions,
        "global_sweep": global_sweep,
        "fold_sweep": fold_sweep,
        "fold_summary": fold_summary,
        "feasible_thresholds": feasible_ranked,
        "selected_threshold": selected_threshold,
        "selected_evaluation": selected_evaluation,
        "local_sensitivity": local_sensitivity,
        "m9w_stress": stress,
        "m9w_selected": selected_stress,
        "freeze_manifest": manifest,
    }
