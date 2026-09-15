"""Cached data access for the lightweight Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import yaml


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError("Could not locate the project root containing src/ and configs/.")


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    path = ROOT / "configs" / "decision_support_app.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading validated demo artifacts...")
def load_repository():
    from src.ontario_peak_risk.decision_support.backend import DemoRepository
    return DemoRepository(load_config(), ROOT)


@st.cache_data(show_spinner=False)
def load_optional_csv(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_optional_markdown(relative_path: str) -> str:
    path = ROOT / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""
