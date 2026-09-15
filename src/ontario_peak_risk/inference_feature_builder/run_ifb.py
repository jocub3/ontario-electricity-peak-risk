"""Complete operational IFB workflow with performance profiling."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .builder import (
    build_feature_matrix,
    feature_contract_report,
    load_model_contracts,
    model_feature_frames,
)
from .integrated import OperationalIntegratedPipeline
from .io import (
    latest_weather_forecast_for_origin,
    load_ifb_config,
    load_operational_updates,
    load_project_history,
    observed_history_for_origin,
    resolve_path,
)
from .validation import (
    demand_history_requirements,
    domain_check,
    domain_reference,
    validate_prediction_output,
    validate_weather_grid,
)


def run_ifb(
    config_path: str | Path,
    *,
    forecast_origin: str | pd.Timestamp,
    fsas: list[str] | None = None,
    execute_models: bool = True,
    decision_threshold: float | None = None,
    model_load_workers: int = 4,
    use_model_cache: bool = True,
) -> dict[str, object]:
    """
    Run operational IFB inference.

    `decision_threshold=None` preserves the official frozen threshold from
    XGBoost metadata. A numeric override changes only the alert decision rule;
    it does not retrain the classifier or alter Peak-Risk scores.
    """
    timings: dict[str, float] = {}
    total_started = perf_counter()

    started = perf_counter()
    config, project_root = load_ifb_config(config_path)
    origin = pd.Timestamp(forecast_origin)
    fsas = fsas or list(config["inference"]["fsas"])
    timings["config"] = perf_counter() - started

    reports_dir = resolve_path(project_root, config["paths"]["reports_dir"])
    outputs_dir = resolve_path(project_root, config["paths"]["outputs_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    project_history = load_project_history(config, project_root)
    operational = load_operational_updates(config, project_root)
    timings["data_loading"] = perf_counter() - started

    started = perf_counter()
    observed_history, duplicates = observed_history_for_origin(
        project_history,
        operational["demand"],
        operational["weather_history"],
        origin,
        config,
    )
    forecast_weather = latest_weather_forecast_for_origin(
        operational["weather_forecasts"], origin, fsas, config
    )
    timings["operational_input_preparation"] = perf_counter() - started

    started = perf_counter()
    demand_coverage, missing_demand = demand_history_requirements(
        observed_history, origin, fsas, config
    )
    weather_coverage, missing_weather = validate_weather_grid(
        forecast_weather, origin, fsas, config
    )
    timings["input_validation"] = perf_counter() - started

    operational["inventory"].to_csv(
        reports_dir / "IFB_01_operational_file_inventory.csv", index=False
    )
    duplicates.to_csv(
        reports_dir / "IFB_01_duplicate_keys.csv", index=False
    )
    demand_coverage.to_csv(
        reports_dir / "IFB_02_demand_history_coverage.csv", index=False
    )
    missing_demand.to_csv(
        reports_dir / "IFB_02_missing_demand_timestamps.csv", index=False
    )
    weather_coverage.to_csv(
        reports_dir / "IFB_05_weather_forecast_coverage.csv", index=False
    )
    missing_weather.to_csv(
        reports_dir / "IFB_05_missing_weather_rows.csv", index=False
    )

    if (
        config["inference"]["fail_on_missing_demand_history"]
        and demand_coverage["status"].eq("FAIL").any()
    ):
        raise ValueError(
            "Insufficient recent observed demand. Review IFB_02 reports."
        )

    if (
        config["inference"]["require_future_weather_grid"]
        and weather_coverage["status"].eq("FAIL").any()
    ):
        raise ValueError(
            "Incomplete h0...h24 weather forecast. Review IFB_05 reports."
        )

    started = perf_counter()
    generated = build_feature_matrix(
        observed_history,
        forecast_weather,
        origin,
        fsas,
        config=config,
    )
    timings["feature_construction"] = perf_counter() - started

    artifacts_dir = resolve_path(
        project_root, config["paths"]["artifacts_dir"]
    )
    contracts = load_model_contracts(artifacts_dir)

    started = perf_counter()
    contract = pd.concat(
        [
            feature_contract_report(
                generated,
                contracts["rf"],
                model_name="RandomForestRegressor",
            ),
            feature_contract_report(
                generated,
                contracts["xgb"],
                model_name="XGBoostClassifier",
            ),
        ],
        ignore_index=True,
    )
    timings["feature_contract_validation"] = perf_counter() - started

    contract.to_csv(
        reports_dir / "IFB_07_feature_contract.csv", index=False
    )

    if (
        config["inference"]["fail_on_feature_contract_mismatch"]
        and contract["status"].eq("FAIL").any()
    ):
        raise ValueError(
            "IFB feature contract validation failed. Review IFB_07 report."
        )

    # Domain reference still uses the original project history because it is
    # intended to represent the model-development domain.
    reference = domain_reference(project_history, config)
    recent = observed_history.copy()
    domain_report = pd.concat(
        [
            domain_check(
                recent,
                reference,
                context="recent_observed_history",
            ),
            domain_check(
                forecast_weather,
                reference,
                context="weather_forecast",
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    reference.to_csv(
        reports_dir / "IFB_07_domain_reference.csv", index=False
    )
    domain_report.to_csv(
        reports_dir / "IFB_07_domain_warnings.csv", index=False
    )

    generated.to_parquet(
        outputs_dir / "model_ready_features.parquet", index=False
    )
    generated.to_csv(
        outputs_dir / "model_ready_features.csv", index=False
    )

    started = perf_counter()
    feature_frames = model_feature_frames(generated, contracts)
    timings["model_frame_construction"] = perf_counter() - started

    prediction = None
    prediction_validation = pd.DataFrame()
    pipeline = None

    if execute_models:
        started = perf_counter()
        pipeline = OperationalIntegratedPipeline(
            artifacts_dir,
            decision_threshold=decision_threshold,
            load_workers=model_load_workers,
            use_cache=use_model_cache,
        )
        timings["model_loading"] = perf_counter() - started

        started = perf_counter()
        prediction = pipeline.predict_from_feature_frames(feature_frames)
        timings["model_inference"] = perf_counter() - started

        prediction_validation = validate_prediction_output(
            prediction,
            fsas,
            int(config["inference"]["horizons"]),
            pipeline.threshold,
        )

        if prediction_validation["status"].eq("FAIL").any():
            raise ValueError(
                "Integrated prediction output validation failed."
            )

        prediction.to_parquet(
            outputs_dir / "integrated_operational_prediction.parquet",
            index=False,
        )
        prediction.to_csv(
            outputs_dir / "integrated_operational_prediction.csv",
            index=False,
        )
        prediction_validation.to_csv(
            reports_dir / "IFB_09_prediction_validation.csv",
            index=False,
        )

    timings["total"] = perf_counter() - total_started
    timing_table = pd.DataFrame(
        [
            {"stage": key, "seconds": value}
            for key, value in timings.items()
        ]
    )
    timing_table.to_csv(
        reports_dir / "IFB_09_runtime_profile.csv",
        index=False,
    )

    manifest = {
        "forecast_origin": str(origin),
        "fsas": fsas,
        "horizons": int(config["inference"]["horizons"]),
        "demand_history_status": (
            "PASS" if demand_coverage["status"].eq("PASS").all()
            else "FAIL"
        ),
        "weather_grid_status": (
            "PASS" if weather_coverage["status"].eq("PASS").all()
            else "FAIL"
        ),
        "feature_contract_status": (
            "PASS"
            if not contract["status"].eq("FAIL").any()
            else "FAIL"
        ),
        "domain_warning_count": int(len(domain_report)),
        "prediction_generated": prediction is not None,
        "official_threshold": (
            pipeline.official_threshold if pipeline is not None else None
        ),
        "decision_threshold_used": (
            pipeline.threshold if pipeline is not None else None
        ),
        "threshold_override_used": (
            pipeline is not None
            and pipeline.threshold != pipeline.official_threshold
        ),
        "model_loaded_from_cache": (
            pipeline.loaded_from_cache if pipeline is not None else None
        ),
        "runtime_seconds": timings,
    }

    (outputs_dir / "IFB_RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "observed_history": observed_history,
        "forecast_weather": forecast_weather,
        "file_inventory": operational["inventory"],
        "demand_coverage": demand_coverage,
        "missing_demand": missing_demand,
        "weather_coverage": weather_coverage,
        "missing_weather": missing_weather,
        "generated_features": generated,
        "contract_report": contract,
        "domain_report": domain_report,
        "feature_frames": feature_frames,
        "prediction": prediction,
        "prediction_validation": prediction_validation,
        "runtime_profile": timing_table,
        "pipeline": pipeline,
        "manifest": manifest,
        # Full demand updates are returned separately so a historical/post-
        # holdout simulation can compare predictions with observed future
        # demand without allowing those values into inference features.
        "operational_demand_updates": operational["demand"],
    }
