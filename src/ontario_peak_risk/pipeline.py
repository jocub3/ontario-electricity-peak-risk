"""High-level project pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


def run_pipeline(config_path: str | Path, stage: str = "all") -> None:
    #"""Run one stage or the complete pipeline.
    #
    #The implementation is intentionally minimal. Each module should later
    #expose functions that can be called from this orchestration layer.
    #"""
    #config = load_config(config_path)

    #print(f"Project: {config['project']['name']}")
    #print(f"Requested stage: {stage}")
    #print("Pipeline skeleton initialized successfully.")


    """Initialize the project and display the notebook-based execution workflow."""

    config = load_config(config_path)

    print(f"Project: {config['project']['name']}")
    print(f"Requested stage: {stage}")
    print()
    print("Project workflow initialized successfully.")
    print()
    print(
        "To reproduce the analytical pipeline, execute the project notebooks "
        "step by step in numerical order, starting from 01 and continuing "
        "through the required modeling, evaluation, inference, and "
        "decision-support phases."
    )
    print()
    print(
        "See notebooks/README.md and the root README.md for the recommended "
        "execution order."
    )
    print()


    
