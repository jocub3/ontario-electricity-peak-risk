from __future__ import annotations
import pandas as pd
import streamlit as st
from ..charts import scenario_forecast_lines, scenario_delta_heatmap, scenario_risk_delta_heatmap, baseline_scenario_demand_lines, baseline_scenario_risk_lines, scenario_alert_changes, all_scenarios_summary_chart


def render(repo, origin, fsas, data):
    st.header("Weather Sensitivity & Scenario Comparison")
    if repo.scenarios.empty:
        st.warning("No precomputed weather-sensitivity artifact is available.")
        return

    base = repo.scenarios[repo.scenarios["forecast_origin"].eq(pd.Timestamp(origin)) & repo.scenarios["fsa"].isin(fsas)].copy()
    if base.empty:
        st.warning("No scenario results exist for the selected forecast origin.")
        return

    available = base[["scenario","temperature_delta_c"]].drop_duplicates().sort_values("temperature_delta_c")
    labels = available["scenario"].astype(str).tolist()
    default = labels.index("Baseline") if "Baseline" in labels else 0

    c1, c2 = st.columns([2,1])
    with c1:
        scenario = st.select_slider("Temperature scenario", options=labels, value=labels[default])
    selected_fsa = c2.selectbox("Detailed FSA", options=fsas, index=0)

    selected = repo.scenario_slice(origin, scenario, fsas)
    delta_c = float(selected["temperature_delta_c"].iloc[0])
    st.caption(
        f"Selected scenario: {delta_c:+.1f} °C at the forecast origin. Model-v1 sensitivity perturbs forecast-origin temperature only; results are noncausal sensitivity, not a future-weather causal effect."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Δ demand", f"{selected['forecast_delta_kwh'].mean():+,.1f} kWh")
    c2.metric("Max |Δ demand|", f"{selected['forecast_delta_kwh'].abs().max():,.1f} kWh")
    c3.metric("Mean Δ Peak-Risk", f"{selected['peak_risk_delta'].mean():+.3f}")
    c4.metric("Alert decisions changed", int(selected["alert_changed"].astype(bool).sum()))

    tab1, tab2, tab3 = st.tabs(["Scenario Heatmaps", "Baseline Comparison", "Scenario Summary"])
    with tab1:
        st.plotly_chart(scenario_forecast_lines(selected), use_container_width=True)
        st.plotly_chart(scenario_delta_heatmap(selected), use_container_width=True)
        st.plotly_chart(scenario_risk_delta_heatmap(selected), use_container_width=True)
    with tab2:
        st.plotly_chart(baseline_scenario_demand_lines(selected, selected_fsa), use_container_width=True)
        st.plotly_chart(baseline_scenario_risk_lines(selected, selected_fsa, repo.threshold), use_container_width=True)
        st.plotly_chart(scenario_alert_changes(selected), use_container_width=True)
    with tab3:
        st.plotly_chart(all_scenarios_summary_chart(base), use_container_width=True)
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
        st.dataframe(comparison, use_container_width=True, hide_index=True, height=480)
