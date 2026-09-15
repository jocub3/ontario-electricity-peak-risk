"""Build lightweight public-demo artifacts from validated project components."""

from __future__ import annotations

from pathlib import Path
import gc
import json
import pandas as pd
import yaml

from .common import resolve


def build_prediction_demo(
    cfg: dict,
    project_root: Path,
    demo_origins: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute the already-validated IFB pipeline for selected real origins.

    This function is intended for OFFLINE preparation. The public Streamlit
    deployment does not import or load the model artifacts.
    """
    from src.ontario_peak_risk.inference_feature_builder.run_ifb import run_ifb

    origins = pd.to_datetime(demo_origins["forecast_origin"]).tolist()
    fsas = list(cfg["application"]["fsas"])
    ifb_config = resolve(project_root, cfg["paths"]["ifb_config"])

    outputs = []
    for i, origin in enumerate(origins, start=1):
        print(f"[{i}/{len(origins)}] Generating predictions for {origin} ...")
        run = run_ifb(
            ifb_config,
            forecast_origin=origin,
            fsas=fsas,
            execute_models=True,
            decision_threshold=None,
            model_load_workers=1,
            use_model_cache=True,
        )
        prediction = run["prediction"]
        if prediction is None:
            raise RuntimeError(f"No prediction returned for {origin}.")
        outputs.append(prediction.copy())

        # Keep each loop conservative on memory.
        del run, prediction
        gc.collect()

    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["forecast_origin", "fsa", "horizon"])
        .reset_index(drop=True)
    )


def build_weather_scenario_demo(
    cfg: dict,
    project_root: Path,
    demo_origins: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the five frozen Model-v1 temperature scenarios for demo origins.

    The logic reuses the validated Weather Sensitivity modules. It perturbs only
    forecast-origin temperature and does not retrain either model.
    """
    from src.ontario_peak_risk.inference_feature_builder.run_ifb import run_ifb
    from src.ontario_peak_risk.weather_sensitivity.scenarios import (
        build_scenario_frames,
    )
    from src.ontario_peak_risk.weather_sensitivity.validation import (
        temperature_reference,
        add_domain_status,
    )
    from src.ontario_peak_risk.weather_sensitivity.inference import run_sensitivity
    from src.ontario_peak_risk.weather_sensitivity.comparison import (
        add_baseline_deltas,
    )

    wsa_config_path = resolve(project_root, cfg["paths"]["weather_sensitivity_config"])
    with wsa_config_path.open("r", encoding="utf-8") as handle:
        wsa = yaml.safe_load(handle)

    ifb_config_path = resolve(project_root, cfg["paths"]["ifb_config"])
    with ifb_config_path.open("r", encoding="utf-8") as handle:
        ifb_cfg = yaml.safe_load(handle)

    source_temp = ifb_cfg["columns"]["temperature"]
    model_temp = wsa["analysis"]["temperature_feature"]
    feature_data = pd.read_parquet(
        resolve(project_root, wsa["paths"]["feature_dataset"])
    )
    if source_temp not in feature_data.columns:
        raise KeyError(
            f"Historical temperature column '{source_temp}' is absent from "
            f"{wsa['paths']['feature_dataset']}."
        )

    ref = temperature_reference(
        feature_data,
        feature=source_temp,
        q=tuple(wsa["analysis"]["caution_quantiles"]),
    )

    metadata = {}
    for task in ["rf", "xgb"]:
        path = (
            project_root
            / wsa["paths"]["artifacts_dir"]
            / wsa["models"][f"{task}_subdir"]
            / "metadata.json"
        )
        metadata[task] = json.loads(path.read_text(encoding="utf-8"))

    outputs = []
    for origin in pd.to_datetime(demo_origins["forecast_origin"]):
        print(f"Generating weather scenarios for {origin} ...")
        ifb_run = run_ifb(
            ifb_config_path,
            forecast_origin=origin,
            fsas=list(cfg["application"]["fsas"]),
            execute_models=False,
        )
        frames = ifb_run["feature_frames"]

        scenario_frames = build_scenario_frames(
            frames,
            wsa["analysis"]["scenarios_c"],
            model_temp,
        )
        for task in ["rf", "xgb"]:
            for horizon in range(1, int(wsa["analysis"]["horizons"]) + 1):
                scenario_frames[task][horizon] = add_domain_status(
                    scenario_frames[task][horizon],
                    ref,
                    model_temp,
                )

        raw = run_sensitivity(scenario_frames, metadata, wsa, project_root)
        result = add_baseline_deltas(raw)
        outputs.append(result)

        del ifb_run, frames, scenario_frames, raw, result
        gc.collect()

    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values(
            ["forecast_origin", "fsa", "temperature_delta_c", "horizon"]
        )
        .reset_index(drop=True)
    )
