from __future__ import annotations
import streamlit as st
from ..charts import demand_heatmap, risk_heatmap, demand_decision_heatmap, risk_timeline, ranking_max_risk, ranking_max_demand, alert_hours_bar
from ..theme import page_header


def render(repo, origin, fsas, data):
    st.markdown(
        page_header(
            "Forecast & Peak Risk",
            "Detailed 24-hour demand forecasts and Peak-Risk scores for the active selection.",
        ),
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["Decision Heatmap", "Peak-Risk Scores", "FSA Detail"])
    with tab1:
        st.plotly_chart(demand_decision_heatmap(data, repo.threshold), width="stretch")
        st.caption("Color represents forecast demand from white to red; cell labels show demand, and ◆ identifies a Peak-Risk alert decision.")
        with st.expander("Show separate demand and Peak-Risk heatmaps"):
            st.plotly_chart(demand_heatmap(data), width="stretch")
            st.plotly_chart(risk_heatmap(data, repo.threshold), width="stretch")
    with tab2:
        st.plotly_chart(risk_timeline(data, repo.threshold), width="stretch")
        st.caption(f"Peak-Risk is presented as a model score. The official frozen alert threshold is {repo.threshold:.2f}.")
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ranking_max_risk(data), width="stretch")
        with c2:
            st.plotly_chart(ranking_max_demand(data), width="stretch")
        st.plotly_chart(alert_hours_bar(data), width="stretch")
        table = data[["fsa","target_timestamp","horizon","forecast_consumption_kwh","peak_risk_score","peak_alert"]].sort_values(["peak_alert","peak_risk_score"], ascending=[False,False])
        table = table.rename(columns={
            "fsa": "FSA",
            "target_timestamp": "Target timestamp",
            "horizon": "Horizon (h+)",
            "forecast_consumption_kwh": "Forecast demand (kWh)",
            "peak_risk_score": "Peak-Risk score",
            "peak_alert": "Alert",
        })
        st.dataframe(table, width="stretch", hide_index=True, height=430)
        st.download_button(
            "Download active results (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_peak_risk_{origin:%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )
