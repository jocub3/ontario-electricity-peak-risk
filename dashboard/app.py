"""Ontario Electricity Demand & Peak-Risk Decision-Support Application."""
from __future__ import annotations

import streamlit as st
from dsa_app.data import load_repository
from dsa_app.state import render_sidebar_controls
from dsa_app.pages import selection, executive, forecast_risk, weather_sensitivity, model_insights, project_info

st.set_page_config(page_title="Ontario Electricity Peak-Risk", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Ontario Electricity Demand & Peak-Risk Decision Support")
st.caption("24-hour electricity-demand forecasting • Peak-Risk monitoring • temperature sensitivity • model interpretation")

try:
    repo = load_repository()
except Exception as exc:
    st.error("The validated public-demo artifacts could not be loaded.")
    st.exception(exc)
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["Forecast Setup", "Executive Summary", "Forecast & Peak-Risk", "Weather Sensitivity", "Model Insights", "Project Information"],
)
origin, fsas = render_sidebar_controls(repo)
data = repo.prediction_slice(origin, fsas)

st.sidebar.divider()
st.sidebar.caption(f"Active origin: {origin:%Y-%m-%d %H:%M}")
st.sidebar.caption("Active FSAs: " + ", ".join(fsas))

pages = {
    "Forecast Setup": selection.render,
    "Executive Summary": executive.render,
    "Forecast & Peak-Risk": forecast_risk.render,
    "Weather Sensitivity": weather_sensitivity.render,
    "Model Insights": model_insights.render,
    "Project Information": project_info.render,
}
pages[page](repo, origin, fsas, data)
