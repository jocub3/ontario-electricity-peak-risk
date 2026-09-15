"""Analysis state and the explicit Run Analysis interaction."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def _format_origin(value) -> str:
    return pd.Timestamp(value).strftime("%b %d, %Y - %H:%M")


def initialize_analysis_state(repo) -> None:
    origins = [pd.Timestamp(x) for x in repo.available_origins]
    if not origins:
        st.error("No validated forecast origins are available.")
        st.stop()

    defaults = {
        "analysis_origin": origins[-1],
        "analysis_fsas": list(repo.fsas),
        "pending_origin": origins[-1],
        "pending_fsas": list(repo.fsas),
        "analysis_updated_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def active_analysis(repo) -> tuple[pd.Timestamp, list[str]]:
    initialize_analysis_state(repo)
    return pd.Timestamp(st.session_state.analysis_origin), list(st.session_state.analysis_fsas)


def render_analysis_controls(repo) -> tuple[pd.Timestamp, list[str]]:
    """Render top-level controls and commit changes only on form submission."""
    initialize_analysis_state(repo)
    origins = [pd.Timestamp(x) for x in repo.available_origins]

    st.markdown('<div class="section-kicker">ANALYSIS CONFIGURATION</div>', unsafe_allow_html=True)
    with st.form("analysis_controls", border=True):
        c1, c2, c3 = st.columns([1.45, 2.35, 1.05], vertical_alignment="bottom")
        with c1:
            pending_origin = st.selectbox(
                "Forecast origin",
                options=origins,
                format_func=_format_origin,
                key="pending_origin",
                help="The public demo exposes validated, precomputed forecast origins only.",
            )
        with c2:
            pending_fsas = st.multiselect(
                "FSAs included in the analysis",
                options=repo.fsas,
                key="pending_fsas",
                help="Every KPI, chart and table updates to the selected FSA set.",
            )
        with c3:
            submitted = st.form_submit_button(
                "Run Analysis", type="primary", width="stretch", icon="▶"
            )

    if submitted:
        if not pending_fsas:
            st.error("Select at least one FSA before running the analysis.")
        else:
            with st.spinner("Updating analysis..."):
                st.session_state.analysis_origin = pd.Timestamp(pending_origin)
                st.session_state.analysis_fsas = list(pending_fsas)
                st.session_state.analysis_updated_at = datetime.now(ZoneInfo("America/Toronto"))
            st.toast("Analysis updated successfully.", icon="✅")

    origin, fsas = active_analysis(repo)
    updated = st.session_state.analysis_updated_at
    updated_text = updated.strftime("%b %d, %Y at %H:%M %Z") if updated else "initial validated selection"
    plural = "s" if len(fsas) != 1 else ""
    st.markdown(
        '<div class="analysis-context">'
        f'<strong>Active results:</strong> {_format_origin(origin)} '
        f'({repo.cfg["application"]["timezone"]}) &nbsp;·&nbsp; '
        f'<strong>{len(fsas)} FSA{plural}:</strong> {", ".join(fsas)} '
        f'&nbsp;·&nbsp; <strong>Updated:</strong> {updated_text}'
        '</div>',
        unsafe_allow_html=True,
    )
    return origin, fsas
