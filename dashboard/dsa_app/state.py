"""Selection state: controls can change without updating the analysis until Run is pressed."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def _format_origin(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def initialize_analysis_state(repo) -> None:
    origins = [pd.Timestamp(x) for x in repo.available_origins]
    if not origins:
        st.error("No precomputed forecast origins are available.")
        st.stop()
    if "analysis_origin" not in st.session_state:
        st.session_state.analysis_origin = origins[-1]
    if "analysis_fsas" not in st.session_state:
        st.session_state.analysis_fsas = list(repo.fsas)


def render_sidebar_controls(repo) -> tuple[pd.Timestamp, list[str]]:
    initialize_analysis_state(repo)
    origins = [pd.Timestamp(x) for x in repo.available_origins]
    current_origin = pd.Timestamp(st.session_state.analysis_origin)
    default_index = origins.index(current_origin) if current_origin in origins else len(origins) - 1

    st.sidebar.markdown("### Analysis controls")
    pending_origin = st.sidebar.selectbox(
        "Forecast date / origin",
        options=origins,
        index=default_index,
        format_func=_format_origin,
        help="Only validated precomputed forecast origins are selectable in the public demo.",
    )
    pending_fsas = st.sidebar.multiselect(
        "FSAs",
        options=repo.fsas,
        default=st.session_state.analysis_fsas,
        help="Select one or more FSAs to include in charts and tables.",
    )

    changed = (
        pd.Timestamp(pending_origin) != pd.Timestamp(st.session_state.analysis_origin)
        or set(pending_fsas) != set(st.session_state.analysis_fsas)
    )
    if changed:
        st.sidebar.caption("Selection changed. Press **Run analysis** to refresh all views.")

    run = st.sidebar.button("▶ Run analysis", type="primary", use_container_width=True)
    if run:
        if not pending_fsas:
            st.sidebar.error("Select at least one FSA before running the analysis.")
        else:
            st.session_state.analysis_origin = pd.Timestamp(pending_origin)
            st.session_state.analysis_fsas = list(pending_fsas)
            st.sidebar.success("Analysis refreshed.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Local operational use requires recent demand history and a weather forecast to be preloaded and validated before model execution. "
        "This public demo reads validated precomputed results only."
    )

    return pd.Timestamp(st.session_state.analysis_origin), list(st.session_state.analysis_fsas)
