"""Efficient rolling-origin SARIMAX candidate model.

Why this implementation exists
------------------------------
The original direct multi-horizon implementation fits a separate SARIMAX
model for every FSA, fold, and horizon. That produces hundreds of expensive
maximum-likelihood fits.

This revised implementation preserves the original code and uses a different
computational strategy:

* one SARIMAX fit per FSA and validation fold;
* the fitted state is updated with newly observed hourly demand without refit;
* at every validation origin, a 24-hour forecast is generated;
* progress, ETA, convergence, fit runtime, and checkpoints are reported;
* the 2025 holdout remains untouched.

The previous tuning result can be reused through ``best_specification.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.ontario_peak_risk.modeling.common import (
    load_feature_dataset,
    load_forecast_targets,
    load_modeling_config,
    resolve_project_path,
    save_csv,
)
from src.ontario_peak_risk.modeling.metrics import forecasting_metrics
from src.ontario_peak_risk.modeling.splits import (
    build_validation_folds,
    slice_forecast_origin_fold,
)

from .data_rolling import (
    build_calendar_exog,
    build_future_exog,
    prepare_observed_series,
)


def _load_branch_config(
    config_path: str | Path,
) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config, path.parent.parent


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _load_specification(
    branch_config: dict,
    project_root: Path,
) -> dict:
    cfg = branch_config["model"]

    if cfg.get("specification"):
        return dict(cfg["specification"])

    spec_path_value = cfg.get("best_specification_path")
    if not spec_path_value:
        raise ValueError(
            "Provide model.specification or model.best_specification_path."
        )

    spec_path = resolve_project_path(
        project_root,
        spec_path_value,
    )
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Best SARIMAX specification not found: {spec_path}"
        )

    return json.loads(
        spec_path.read_text(encoding="utf-8")
    )


def _fit_one_series(
    train_frame: pd.DataFrame,
    specification: dict,
    fit_cfg: dict,
) -> tuple[object, dict]:
    """Fit one SARIMAX model and return convergence diagnostics."""
    train_times = pd.DatetimeIndex(train_frame["timestamp"])
    train_y = train_frame["total_consumption_kwh"].astype(float).to_numpy()
    train_exog = build_calendar_exog(train_times)

    # The original selected specification uses trend="c".
    # In rolling-origin mode statsmodels reconstructs the state-space model
    # when `extend()` receives newly observed data. Because an hourly update
    # can contain only one row, every exogenous column is technically constant
    # within that tiny update and statsmodels rejects trend="c" together with
    # "constant" exog.
    #
    # We preserve the same intercept concept by including an explicit
    # `intercept = 1.0` column in build_calendar_exog() and setting trend="n".
    # This is algebraically equivalent to using a constant trend, but it is
    # robust to one-row rolling updates.
    requested_trend = specification.get("trend", "c")
    model_trend = "n" if requested_trend == "c" else requested_trend

    model = SARIMAX(
        endog=train_y,
        exog=train_exog.to_numpy(),
        order=tuple(specification["order"]),
        seasonal_order=tuple(specification["seasonal_order"]),
        trend=model_trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    started = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        fitted = model.fit(
            disp=False,
            maxiter=int(fit_cfg.get("maxiter", 75)),
            method=str(fit_cfg.get("method", "lbfgs")),
        )

    fit_seconds = time.perf_counter() - started

    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]

    mle = getattr(fitted, "mle_retvals", {}) or {}
    converged = bool(mle.get("converged", not convergence_warnings))
    iterations = mle.get("iterations")
    if iterations is None:
        iterations = mle.get("iter")

    diagnostics = {
        "requested_trend": requested_trend,
        "implemented_trend": model_trend,
        "explicit_intercept_in_exog": requested_trend == "c",
        "converged": converged,
        "fit_seconds": fit_seconds,
        "iterations": iterations,
        "convergence_warning_count": len(convergence_warnings),
        "aic": float(fitted.aic) if np.isfinite(fitted.aic) else np.nan,
        "bic": float(fitted.bic) if np.isfinite(fitted.bic) else np.nan,
    }

    return fitted, diagnostics


def _task_checkpoint_path(
    checkpoint_dir: Path,
    fold_name: str,
    fsa: str,
) -> Path:
    return checkpoint_dir / f"{fold_name}__{fsa}.parquet"


def _task_diagnostics_path(
    checkpoint_dir: Path,
    fold_name: str,
    fsa: str,
) -> Path:
    return checkpoint_dir / f"{fold_name}__{fsa}__fit.json"


def _run_fold_fsa(
    *,
    fold,
    validation_targets: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    fsa: str,
    specification: dict,
    cfg: dict,
    checkpoint_dir: Path,
    resume: bool,
    origin_limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit once, update state sequentially, and forecast 24 hours per origin."""
    checkpoint = _task_checkpoint_path(
        checkpoint_dir,
        fold.name,
        fsa,
    )
    diagnostics_path = _task_diagnostics_path(
        checkpoint_dir,
        fold.name,
        fsa,
    )

    if resume and checkpoint.exists() and diagnostics_path.exists():
        print(
            f"  [resume] Loading completed task: {fold.name} / {fsa}",
            flush=True,
        )
        saved = pd.read_parquet(checkpoint)
        diagnostics = json.loads(
            diagnostics_path.read_text(encoding="utf-8")
        )
        diagnostics["resumed_from_checkpoint"] = True
        return saved, diagnostics

    observed = prepare_observed_series(
        feature_dataset,
        fsa,
    )

    validation_fsa = (
        validation_targets.loc[
            validation_targets["fsa"].eq(fsa)
        ]
        .sort_values("forecast_origin", kind="stable")
        .reset_index(drop=True)
    )

    if validation_fsa.empty:
        raise ValueError(
            f"No validation targets found for {fold.name} / {fsa}."
        )

    if origin_limit is not None:
        validation_fsa = validation_fsa.head(int(origin_limit)).copy()

    first_origin = pd.Timestamp(
        validation_fsa["forecast_origin"].min()
    )

    # Training uses all observations strictly before the first validation origin.
    # These observations are historically available when the first forecast is made.
    train_frame = observed.loc[
        observed["timestamp"] < first_origin
    ].copy()

    minimum_training_hours = int(
        cfg["execution"].get(
            "minimum_training_hours",
            24 * 90,
        )
    )

    if len(train_frame) < minimum_training_hours:
        raise ValueError(
            f"{fold.name} / {fsa}: only {len(train_frame):,} training hours; "
            f"minimum required is {minimum_training_hours:,}."
        )

    max_training_hours = cfg["execution"].get(
        "max_training_hours"
    )
    if max_training_hours:
        train_frame = train_frame.tail(
            int(max_training_hours)
        ).copy()

    print(
        f"  Fit data: {len(train_frame):,} hourly observations "
        f"({train_frame['timestamp'].min()} -> "
        f"{train_frame['timestamp'].max()})",
        flush=True,
    )

    fitted, fit_diag = _fit_one_series(
        train_frame,
        specification,
        cfg["fit"],
    )

    print(
        "  Fit completed | "
        f"time={_format_seconds(fit_diag['fit_seconds'])} | "
        f"converged={fit_diag['converged']} | "
        f"iterations={fit_diag['iterations']} | "
        f"warnings={fit_diag['convergence_warning_count']}",
        flush=True,
    )

    # Keep the state at the final training observation.
    current_result = fitted
    state_time = pd.Timestamp(
        train_frame["timestamp"].max()
    )

    observed_indexed = observed.set_index("timestamp")

    total_origins = len(validation_fsa)
    progress_every = int(
        cfg["execution"].get(
            "progress_every_origins",
            250,
        )
    )

    horizon_count = int(cfg["horizons"])
    target_prefix = str(cfg["target_prefix"])

    prediction_rows: list[dict] = []
    forecast_started = time.perf_counter()

    for origin_number, row in validation_fsa.iterrows():
        origin = pd.Timestamp(row["forecast_origin"])

        # Update the state with all newly observed data up to and including origin.
        update_frame = observed.loc[
            (observed["timestamp"] > state_time)
            & (observed["timestamp"] <= origin)
        ].copy()

        if not update_frame.empty:
            update_y = update_frame[
                "total_consumption_kwh"
            ].astype(float).to_numpy()

            update_exog = build_calendar_exog(
                update_frame["timestamp"]
            ).to_numpy()

            current_result = current_result.extend(
                endog=update_y,
                exog=update_exog,
            )

            state_time = pd.Timestamp(
                update_frame["timestamp"].max()
            )

        if state_time < origin:
            raise ValueError(
                f"{fold.name} / {fsa}: observed demand does not reach "
                f"forecast origin {origin}. Last observation is {state_time}."
            )

        future_exog = build_future_exog(
            origin,
            horizon=horizon_count,
        )

        forecast = current_result.get_forecast(
            steps=horizon_count,
            exog=future_exog.to_numpy(),
        )

        predicted = np.asarray(
            forecast.predicted_mean,
            dtype=float,
        )

        for horizon in range(1, horizon_count + 1):
            target_column = (
                f"{target_prefix}{horizon:02d}"
            )
            prediction_rows.append(
                {
                    "fold": fold.name,
                    "fsa": fsa,
                    "forecast_origin": origin,
                    "horizon": horizon,
                    "actual": float(row[target_column]),
                    "predicted": float(
                        predicted[horizon - 1]
                    ),
                }
            )

        completed = origin_number + 1

        if (
            completed == 1
            or completed % progress_every == 0
            or completed == total_origins
        ):
            elapsed = time.perf_counter() - forecast_started
            rate = completed / elapsed if elapsed > 0 else np.nan
            remaining = total_origins - completed
            eta = remaining / rate if rate > 0 else np.nan

            print(
                "  Forecast progress | "
                f"{completed:,}/{total_origins:,} origins "
                f"({completed / total_origins:.1%}) | "
                f"elapsed={_format_seconds(elapsed)} | "
                f"task ETA={_format_seconds(eta)}",
                flush=True,
            )

    result = pd.DataFrame(prediction_rows)
    result["residual"] = (
        result["actual"]
        - result["predicted"]
    )

    task_seconds = (
        fit_diag["fit_seconds"]
        + (time.perf_counter() - forecast_started)
    )

    diagnostics = {
        "fold": fold.name,
        "fsa": fsa,
        "training_observations": int(len(train_frame)),
        "validation_origins": int(total_origins),
        "forecast_rows": int(len(result)),
        "specification": specification,
        **fit_diag,
        "task_seconds": task_seconds,
        "resumed_from_checkpoint": False,
    }

    checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result.to_parquet(
        checkpoint,
        index=False,
    )
    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"  Task checkpoint saved: {checkpoint.name}",
        flush=True,
    )

    return result, diagnostics


def _build_metrics(
    predictions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    fold_rows = []
    fsa_rows = []
    horizon_rows = []

    for fold, group in predictions.groupby(
        "fold",
        observed=True,
    ):
        fold_rows.append(
            {
                "fold": fold,
                **forecasting_metrics(
                    group["actual"],
                    group["predicted"],
                ),
            }
        )

    for (fold, fsa), group in predictions.groupby(
        ["fold", "fsa"],
        observed=True,
    ):
        fsa_rows.append(
            {
                "fold": fold,
                "fsa": fsa,
                **forecasting_metrics(
                    group["actual"],
                    group["predicted"],
                ),
            }
        )

    for (fold, horizon), group in predictions.groupby(
        ["fold", "horizon"],
        observed=True,
    ):
        horizon_rows.append(
            {
                "fold": fold,
                "horizon": int(horizon),
                **forecasting_metrics(
                    group["actual"],
                    group["predicted"],
                ),
            }
        )

    global_metrics = pd.DataFrame(
        [
            {
                "model": "sarimax_rolling",
                **forecasting_metrics(
                    predictions["actual"],
                    predictions["predicted"],
                ),
            }
        ]
    )

    return {
        "global_metrics": global_metrics,
        "fold_metrics": pd.DataFrame(fold_rows),
        "fsa_metrics": pd.DataFrame(fsa_rows),
        "horizon_metrics": pd.DataFrame(horizon_rows),
    }


def run_sarimax_rolling(
    config_path: str | Path = "configs/sarimax_rolling.yaml",
    *,
    mode: str = "full",
    resume: bool = True,
) -> dict[str, object]:
    """
    Run the revised SARIMAX model.

    Parameters
    ----------
    mode:
        ``"smoke"`` runs only the configured FSA/fold and a small number
        of origins. Use this first to verify convergence and estimate runtime.
        ``"full"`` runs all development folds and FSAs.
    resume:
        Reuse completed FSA/fold checkpoints.
    """
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'.")

    branch_config, project_root = _load_branch_config(
        config_path
    )

    foundation_path = resolve_project_path(
        project_root,
        branch_config["paths"]["modeling_foundation_config"],
    )
    modeling_config, _ = load_modeling_config(
        foundation_path
    )

    feature_dataset = load_feature_dataset(
        modeling_config,
        project_root,
    )
    forecast_targets = load_forecast_targets(
        modeling_config,
        project_root,
    )

    feature_dataset = feature_dataset.copy()
    feature_dataset["timestamp"] = pd.to_datetime(
        feature_dataset["timestamp"]
    )
    forecast_targets = forecast_targets.copy()
    forecast_targets["forecast_origin"] = pd.to_datetime(
        forecast_targets["forecast_origin"]
    )

    cfg = branch_config["model"]

    if bool(cfg.get("evaluate_final_holdout", False)):
        raise ValueError(
            "Revised SARIMAX must not evaluate the final 2025 holdout."
        )

    specification = _load_specification(
        branch_config,
        project_root,
    )

    paths = {
        key: resolve_project_path(
            project_root,
            value,
        )
        for key, value in branch_config["paths"].items()
        if key.endswith("_dir")
    }

    for path in paths.values():
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    checkpoint_dir = paths["outputs_dir"] / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds = build_validation_folds(
        modeling_config
    )

    smoke_cfg = cfg.get("smoke_test", {})

    if mode == "smoke":
        requested_fold = str(
            smoke_cfg.get(
                "fold",
                folds[0].name,
            )
        )
        folds = [
            fold
            for fold in folds
            if fold.name == requested_fold
        ]
        if not folds:
            raise ValueError(
                f"Smoke-test fold not found: {requested_fold}"
            )

    task_descriptors = []

    for fold in folds:
        _, validation_targets = slice_forecast_origin_fold(
            forecast_targets,
            fold,
            time_column="forecast_origin",
            max_horizon_hours=int(cfg["horizons"]),
        )

        fsas = sorted(
            validation_targets["fsa"]
            .dropna()
            .unique()
        )

        if mode == "smoke":
            requested_fsa = str(
                smoke_cfg.get(
                    "fsa",
                    fsas[0],
                )
            )
            fsas = [
                fsa
                for fsa in fsas
                if fsa == requested_fsa
            ]
            if not fsas:
                raise ValueError(
                    f"Smoke-test FSA not found: {requested_fsa}"
                )

        for fsa in fsas:
            task_descriptors.append(
                (fold, validation_targets, fsa)
            )

    total_tasks = len(task_descriptors)

    print("=" * 72, flush=True)
    print("SARIMAX ROLLING-ORIGIN REVISION", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Specification: {specification}", flush=True)
    print(
        "Strategy: one fit per FSA/fold + rolling state updates",
        flush=True,
    )
    print(f"Tasks to process: {total_tasks}", flush=True)
    print("Final 2025 holdout evaluated: NO", flush=True)
    print("=" * 72, flush=True)

    if mode == "smoke":
        print(
            "Smoke test does not create the official final metrics files.",
            flush=True,
        )

    run_started = time.perf_counter()
    completed_task_times: list[float] = []
    prediction_parts = []
    diagnostics_rows = []

    for task_number, (
        fold,
        validation_targets,
        fsa,
    ) in enumerate(task_descriptors, start=1):

        print(
            f"\n[Task {task_number}/{total_tasks}] "
            f"{fold.name} / {fsa}",
            flush=True,
        )

        origin_limit = None
        if mode == "smoke":
            origin_limit = int(
                smoke_cfg.get(
                    "origins",
                    48,
                )
            )

        task_started = time.perf_counter()

        predictions, diagnostics = _run_fold_fsa(
            fold=fold,
            validation_targets=validation_targets,
            feature_dataset=feature_dataset,
            fsa=fsa,
            specification=specification,
            cfg=cfg,
            checkpoint_dir=(
                checkpoint_dir
                if mode == "full"
                else paths["outputs_dir"] / "smoke_checkpoints"
            ),
            resume=(resume and mode == "full"),
            origin_limit=origin_limit,
        )

        prediction_parts.append(
            predictions
        )
        diagnostics_rows.append(
            diagnostics
        )

        task_elapsed = time.perf_counter() - task_started
        completed_task_times.append(task_elapsed)

        average_task = float(
            np.mean(completed_task_times)
        )
        remaining_tasks = total_tasks - task_number
        overall_eta = average_task * remaining_tasks

        print(
            f"[Task {task_number}/{total_tasks}] completed | "
            f"task time={_format_seconds(task_elapsed)} | "
            f"estimated remaining run time={_format_seconds(overall_eta)}",
            flush=True,
        )

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    diagnostics_frame = pd.DataFrame(
        diagnostics_rows
    )

    if mode == "smoke":
        smoke_metrics = _build_metrics(
            predictions
        )
        total_seconds = time.perf_counter() - run_started

        print("\n" + "=" * 72, flush=True)
        print("SMOKE TEST COMPLETED", flush=True)
        print(
            f"Elapsed: {_format_seconds(total_seconds)}",
            flush=True,
        )
        print(
            "Review convergence and runtime before running mode='full'.",
            flush=True,
        )
        print("=" * 72, flush=True)

        return {
            **smoke_metrics,
            "fit_diagnostics": diagnostics_frame,
            "predictions": predictions,
            "specification": specification,
            "mode": mode,
        }

    metrics = _build_metrics(
        predictions
    )

    save_csv(
        metrics["global_metrics"],
        paths["reports_dir"] / "SARIMAX_ROLLING_06_global_metrics.csv",
    )
    save_csv(
        metrics["fold_metrics"],
        paths["reports_dir"] / "SARIMAX_ROLLING_06_fold_metrics.csv",
    )
    save_csv(
        metrics["fsa_metrics"],
        paths["reports_dir"] / "SARIMAX_ROLLING_06_fsa_metrics.csv",
    )
    save_csv(
        metrics["horizon_metrics"],
        paths["reports_dir"] / "SARIMAX_ROLLING_06_horizon_metrics.csv",
    )
    save_csv(
        diagnostics_frame,
        paths["reports_dir"] / "SARIMAX_ROLLING_fit_diagnostics.csv",
    )

    predictions.to_parquet(
        paths["outputs_dir"] / "sarimax_rolling_validation_predictions.parquet",
        index=False,
    )

    (
        paths["outputs_dir"] / "specification_used.json"
    ).write_text(
        json.dumps(
            specification,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_seconds = time.perf_counter() - run_started

    print("\n" + "=" * 72, flush=True)
    print("SARIMAX ROLLING COMPLETED SUCCESSFULLY", flush=True)
    print(f"Total runtime: {_format_seconds(total_seconds)}", flush=True)
    print(f"Specification: {specification}", flush=True)
    print("Final 2025 holdout evaluated: NO", flush=True)
    print("=" * 72, flush=True)

    return {
        **metrics,
        "fit_diagnostics": diagnostics_frame,
        "predictions": predictions,
        "specification": specification,
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run efficient rolling-origin SARIMAX."
    )
    parser.add_argument(
        "--config",
        default="configs/sarimax_rolling.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )
    args = parser.parse_args()

    run_sarimax_rolling(
        args.config,
        mode=args.mode,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
