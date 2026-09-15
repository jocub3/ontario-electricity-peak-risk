from __future__ import annotations
import streamlit as st
from ..charts import forecast_lines, ranking_max_risk, ranking_max_demand, alert_hours_bar


def render(repo, origin, fsas, data):
    st.header("Executive Summary")
    metrics = repo.executive_metrics(data)
    if not metrics:
        st.warning("No prediction data for the selected analysis.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max expected demand", f"{metrics['expected_maximum_demand_kwh']:,.0f} kWh")
    c2.metric("Max-demand FSA", metrics["maximum_demand_fsa"])
    c3.metric("Highest Peak-Risk", f"{metrics['highest_risk_score']:.3f}")
    c4.metric("Highest-risk FSA", metrics["highest_risk_fsa"])
    c5.metric("Alert hours", metrics["peak_risk_alerts"])
    st.caption(f"Highest-risk target hour: {metrics['highest_risk_hour']:%Y-%m-%d %H:%M}")

    st.plotly_chart(forecast_lines(data), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(ranking_max_risk(data), use_container_width=True)
    with c2:
        st.plotly_chart(ranking_max_demand(data), use_container_width=True)
    st.plotly_chart(alert_hours_bar(data), use_container_width=True)

    summary = data.groupby("fsa").agg(max_demand_kwh=("forecast_consumption_kwh","max"), max_peak_risk=("peak_risk_score","max"), alert_hours=("peak_alert","sum")).reset_index().sort_values("max_peak_risk", ascending=False)
    st.subheader("FSA decision summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
