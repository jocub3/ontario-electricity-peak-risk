from __future__ import annotations
import pandas as pd
import streamlit as st
from ..charts import scenario_forecast_lines, scenario_delta_heatmap, scenario_risk_delta_heatmap, baseline_scenario_demand_lines, baseline_scenario_risk_lines, scenario_alert_changes, all_scenarios_summary_chart
from ..theme import page_header


def render(repo, origin, fsas, data):
    st.markdown(
        page_header(
            "Weather Scenarios",
            "Compare model sensitivity to controlled forecast-origin temperature changes.",
        ),
        unsafe_allow_html=True,
    )
    if repo.scenarios.empty:
        st.warning("No precomputed weather-sensitivity artifact is available.")
        return

    base = repo.scenarios[repo.scenarios["forecast_origin"].eq(pd.Timestamp(origin)) & repo.scenarios["fsa"].isin(fsas)].copy()
    if base.empty:
        st.warning("No scenario results exist for the selected forecast origin.")
        return

    available = base[["scenario","temperature_delta_c"]].drop_duplicates().sort_values("temperature_delta_c")
    labels = available["scenario"].astype(str).tolist()
    non_baseline = available[available["temperature_delta_c"].ne(0)]
    default_label = non_baseline.iloc[-1]["scenario"] if not non_baseline.empty else labels[0]
    default = labels.index(str(default_label))

    overview = base.groupby(["scenario", "temperature_delta_c"], as_index=False).agg(
        mean_absolute_demand_change_kwh=("forecast_delta_kwh", lambda x: x.abs().mean()),
        maximum_absolute_demand_change_kwh=("forecast_delta_kwh", lambda x: x.abs().max()),
        fsa_hour_decisions_changed=("alert_changed", lambda x: x.astype(bool).sum()),
    ).sort_values("temperature_delta_c")
    st.subheader("Scenario overview")
    st.dataframe(overview, width="stretch", hide_index=True)

    c1, c2 = st.columns([2,1])
    with c1:
        scenario = st.select_slider("Temperature scenario", options=labels, value=labels[default])
    selected_fsa = c2.selectbox("Detailed FSA", options=fsas, index=0)

    selected = repo.scenario_slice(origin, scenario, fsas)
    delta_c = float(selected["temperature_delta_c"].iloc[0])
    st.warning(
        f"Selected change: {delta_c:+.1f} °C at the forecast origin only. "
        "This is a noncausal model-sensitivity test, not a complete 24-hour weather forecast or a causal estimate."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Δ demand", f"{selected['forecast_delta_kwh'].mean():+,.1f} kWh")
    c2.metric("Max |Δ demand|", f"{selected['forecast_delta_kwh'].abs().max():,.1f} kWh")
    c3.metric("Mean Δ Peak-Risk", f"{selected['peak_risk_delta'].mean():+.3f}")
    c4.metric("Alert decisions changed", int(selected["alert_changed"].astype(bool).sum()))

    tab1, tab2, tab3 = st.tabs(["Baseline Comparison", "Scenario Heatmaps", "All-Scenario Detail"])
    with tab1:
        st.plotly_chart(baseline_scenario_demand_lines(selected, selected_fsa), width="stretch")
        st.plotly_chart(baseline_scenario_risk_lines(selected, selected_fsa, repo.threshold), width="stretch")
        st.plotly_chart(scenario_alert_changes(selected), width="stretch")
    with tab2:
        st.plotly_chart(scenario_forecast_lines(selected), width="stretch")
        st.caption("Red diamonds identify new Peak-Risk alerts produced by the selected scenario compared with Baseline.")
        st.plotly_chart(scenario_delta_heatmap(selected), width="stretch")
        st.plotly_chart(scenario_risk_delta_heatmap(selected), width="stretch")
    with tab3:
        st.plotly_chart(all_scenarios_summary_chart(base), width="stretch")
        comparison = base.groupby(["scenario","temperature_delta_c","fsa"], as_index=False).agg(
            baseline_max_demand=("baseline_forecast_kwh","max"),
            scenario_max_demand=("forecast_consumption_kwh","max"),
            mean_demand_change=("forecast_delta_kwh","mean"),
            max_abs_demand_change=("forecast_delta_kwh", lambda x: x.abs().max()),
            mean_peak_risk_change=("peak_risk_delta","mean"),
            max_abs_peak_risk_change=("peak_risk_delta", lambda x: x.abs().max()),
            baseline_alerts=("baseline_peak_alert","sum"),
            scenario_alerts=("peak_alert","sum"),
            alert_changes=("alert_changed", lambda x: x.astype(bool).sum()),
        ).sort_values(["temperature_delta_c","fsa"])
        st.dataframe(comparison, width="stretch", hide_index=True, height=480)
