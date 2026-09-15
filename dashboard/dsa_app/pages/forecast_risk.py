from __future__ import annotations
import streamlit as st
from ..charts import demand_heatmap, risk_heatmap, combined_decision_heatmap, forecast_lines, risk_timeline, ranking_max_risk, ranking_max_demand, alert_hours_bar


def render(repo, origin, fsas, data):
    st.header("Forecast & Peak-Risk Monitor")
    tab1, tab2, tab3, tab4 = st.tabs(["Heatmaps", "24-hour Forecast", "Peak-Risk", "Rankings & Table"])
    with tab1:
        st.plotly_chart(demand_heatmap(data), use_container_width=True)
        st.plotly_chart(risk_heatmap(data, repo.threshold), use_container_width=True)
        st.plotly_chart(combined_decision_heatmap(data, repo.threshold), use_container_width=True)
        st.caption("In the combined heatmap, color represents Peak-Risk, cell labels show forecast demand, and ◆ identifies an alert decision.")
    with tab2:
        st.plotly_chart(forecast_lines(data), use_container_width=True)
        st.caption("Normal forecast points use the line marker; diamond markers identify hours classified as Peak-Risk.")
    with tab3:
        st.plotly_chart(risk_timeline(data, repo.threshold), use_container_width=True)
        st.caption(f"Official frozen Peak-Risk threshold: {repo.threshold:.2f}.")
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ranking_max_risk(data), use_container_width=True)
        with c2:
            st.plotly_chart(ranking_max_demand(data), use_container_width=True)
        st.plotly_chart(alert_hours_bar(data), use_container_width=True)
        table = data[["fsa","target_timestamp","horizon","forecast_consumption_kwh","peak_risk_score","peak_alert"]].sort_values(["peak_alert","peak_risk_score"], ascending=[False,False])
        st.dataframe(table, use_container_width=True, hide_index=True, height=430)
