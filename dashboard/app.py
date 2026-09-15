"""Ontario Electricity Demand & Peak-Risk Decision-Support Application."""
from __future__ import annotations

import streamlit as st
from dsa_app.data import load_repository
from dsa_app.state import active_analysis, render_analysis_controls
from dsa_app.pages import executive, forecast_risk, weather_sensitivity, model_insights, project_info
from dsa_app.theme import app_css

st.set_page_config(page_title="Ontario Electricity Peak-Risk", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown(app_css(), unsafe_allow_html=True)

try:
    repo = load_repository()
except Exception as exc:
    st.error("The validated public-demo artifacts could not be loaded.")
    st.exception(exc)
    st.stop()

def _context():
    origin, fsas = active_analysis(repo)
    return origin, fsas, repo.prediction_slice(origin, fsas)


def _decision_overview():
    executive.render(repo, *_context())


def _forecast_risk():
    forecast_risk.render(repo, *_context())


def _weather_scenarios():
    weather_sensitivity.render(repo, *_context())


def _model_methodology():
    model_insights.render(repo, *_context())


def _about_project():
    project_info.render(repo, *_context())


st.sidebar.markdown("## ⚡ Ontario Peak-Risk")
st.sidebar.caption("Decision Support System")

navigation = st.navigation(
    {
        "DECISION SUPPORT": [
            st.Page(_decision_overview, title="Decision Overview", icon="📊", default=True),
            st.Page(_forecast_risk, title="Forecast & Peak Risk", icon="📈"),
            st.Page(_weather_scenarios, title="Weather Scenarios", icon="🌡️"),
        ],
        "TECHNICAL INFORMATION": [
            st.Page(_model_methodology, title="Model & Methodology", icon="🧠"),
            st.Page(_about_project, title="About the Project", icon="ℹ️"),
        ],
    },
    position="sidebar",
)

st.title("⚡ Ontario Electricity Demand & Peak-Risk Decision Support")
st.caption("24-hour demand forecasting · Peak-Risk monitoring · temperature sensitivity · model interpretation")

if navigation.title in {"Decision Overview", "Forecast & Peak Risk", "Weather Scenarios"}:
    render_analysis_controls(repo)

st.sidebar.divider()
st.sidebar.caption("Public demo")
st.sidebar.caption(
    "Using this application requires recent demand history and a weather forecast "
    "to be preloaded and validated before model execution."
)
navigation.run()
