"""Shared utilities for the Decision-Support Application phase."""

from __future__ import annotations

from pathlib import Path
import json
import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root by searching for both `src` and `configs`."""
    current = Path(start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "src").exists() and (current / "configs").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not locate project root containing both `src/` and `configs/`."
    )


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    """Load DSA configuration and resolve the project root."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        project_root = find_project_root()
        config_path = project_root / config_path
    else:
        project_root = find_project_root(config_path.parent)

    if not config_path.exists():
        raise FileNotFoundError(f"Decision-support config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    return cfg, project_root


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_phase_dirs(cfg: dict, project_root: Path) -> dict[str, Path]:
    """Create only DSA-owned output/document folders."""
    keys = ["app_data_dir", "outputs_dir", "reports_dir", "docs_dir"]
    result = {}
    for key in keys:
        path = resolve(project_root, cfg["paths"][key])
        path.mkdir(parents=True, exist_ok=True)
        result[key] = path
    return result


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
