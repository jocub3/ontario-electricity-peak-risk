"""Robust loaders for standardized model-comparison outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


@dataclass
class ModelReports:
    key: str
    display_name: str
    family: str
    task: str
    report_dir: Path
    global_metrics: pd.DataFrame | None = None
    fold_metrics: pd.DataFrame | None = None
    fsa_metrics: pd.DataFrame | None = None
    horizon_metrics: pd.DataFrame | None = None
    threshold_analysis: pd.DataFrame | None = None
    tuning_details: pd.DataFrame | None = None
    tuning_summary: pd.DataFrame | None = None
    runtime_diagnostics: pd.DataFrame | None = None


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, path.parent.parent


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _csv_candidates(
    report_dir: Path,
    prefixes: Iterable[str],
    suffix_keywords: Iterable[str],
) -> list[Path]:
    """Find the most likely CSV without assuming one exact model prefix."""
    if not report_dir.exists():
        return []

    files = list(report_dir.glob("*.csv"))
    if not files:
        files = list(report_dir.rglob("*.csv"))

    prefixes_lower = [p.lower() for p in prefixes]
    keywords_lower = [k.lower() for k in suffix_keywords]

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        prefix_score = int(any(prefix in name for prefix in prefixes_lower))
        keyword_score = sum(keyword in name for keyword in keywords_lower)
        return (keyword_score, prefix_score, name)

    matched = [
        path
        for path in files
        if all(keyword in path.name.lower() for keyword in keywords_lower)
    ]

    return sorted(
        matched,
        key=score,
        reverse=True,
    )


def _find_report_dir(
    project_root: Path,
    configured_dir: str,
    model_key: str,
    prefixes: list[str],
    task: str,
) -> Path:
    configured = resolve_project_path(project_root, configured_dir)
    if configured.exists():
        return configured

    task_root = (
        project_root
        / "reports"
        / "modeling"
        / ("forecasting" if task == "forecasting" else "peak_risk")
    )
    if not task_root.exists():
        return configured

    candidates = [
        p for p in task_root.iterdir()
        if p.is_dir()
    ]

    tokens = [model_key.lower(), *[p.lower() for p in prefixes]]

    ranked = sorted(
        candidates,
        key=lambda p: sum(token in p.name.lower() for token in tokens),
        reverse=True,
    )

    if ranked and sum(token in ranked[0].name.lower() for token in tokens) > 0:
        return ranked[0]

    return configured


def _read_optional(
    report_dir: Path,
    prefixes: list[str],
    keywords: list[str],
) -> pd.DataFrame | None:
    candidates = _csv_candidates(
        report_dir,
        prefixes,
        keywords,
    )
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def _runtime_diagnostics(
    report_dir: Path,
    prefixes: list[str],
) -> pd.DataFrame | None:
    files = list(report_dir.glob("*.csv"))
    candidates = [
        path
        for path in files
        if any(
            token in path.name.lower()
            for token in [
                "runtime",
                "fit_diagnostic",
                "training_time",
                "timing",
            ]
        )
    ]
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def load_model_reports(
    config: dict,
    project_root: Path,
    task: str,
) -> dict[str, ModelReports]:
    models_cfg = config[task]["models"]
    loaded: dict[str, ModelReports] = {}

    for key, model_cfg in models_cfg.items():
        prefixes = list(model_cfg.get("prefixes", []))

        report_dir = _find_report_dir(
            project_root,
            model_cfg["report_dir"],
            key,
            prefixes,
            task,
        )

        reports = ModelReports(
            key=key,
            display_name=model_cfg["display_name"],
            family=model_cfg["family"],
            task=task,
            report_dir=report_dir,
        )

        reports.global_metrics = _read_optional(
            report_dir, prefixes, ["global", "metric"]
        )
        reports.fold_metrics = _read_optional(
            report_dir, prefixes, ["fold", "metric"]
        )
        reports.fsa_metrics = _read_optional(
            report_dir, prefixes, ["fsa", "metric"]
        )
        reports.horizon_metrics = _read_optional(
            report_dir, prefixes, ["horizon", "metric"]
        )
        reports.threshold_analysis = _read_optional(
            report_dir, prefixes, ["threshold", "analysis"]
        )
        reports.tuning_details = _read_optional(
            report_dir, prefixes, ["tuning", "detail"]
        )
        reports.tuning_summary = _read_optional(
            report_dir, prefixes, ["tuning", "summary"]
        )
        reports.runtime_diagnostics = _runtime_diagnostics(
            report_dir, prefixes
        )

        loaded[key] = reports

    return loaded


def validate_loaded_reports(
    reports: dict[str, ModelReports],
) -> pd.DataFrame:
    rows = []

    for key, model in reports.items():
        rows.append(
            {
                "model_key": key,
                "model": model.display_name,
                "task": model.task,
                "report_dir": str(model.report_dir),
                "global_metrics": model.global_metrics is not None,
                "fold_metrics": model.fold_metrics is not None,
                "fsa_metrics": model.fsa_metrics is not None,
                "horizon_metrics": model.horizon_metrics is not None,
                "threshold_analysis": model.threshold_analysis is not None,
                "tuning_details": model.tuning_details is not None,
                "runtime_diagnostics": model.runtime_diagnostics is not None,
            }
        )

    return pd.DataFrame(rows)
