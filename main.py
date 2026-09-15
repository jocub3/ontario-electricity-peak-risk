"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.ontario_peak_risk.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ontario electricity demand forecasting and peak-risk "
            "decision-support pipeline."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "forecast", "peak-risk", "scenarios", "all"],
        default="all",
        help="Pipeline stage context to initialize.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit decision-support dashboard.",
    )

    return parser


def launch_dashboard() -> None:
    """Launch the Streamlit decision-support dashboard."""

    dashboard_path = Path(__file__).resolve().parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        raise FileNotFoundError(
            f"Dashboard application not found: {dashboard_path}"
        )

    print()
    print("Launching Ontario Electricity Peak-Risk dashboard...")
    print(f"Dashboard entry point: {dashboard_path}")
    print()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
        ],
        check=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(config_path=args.config, stage=args.stage)

    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
