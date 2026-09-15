from __future__ import annotations
import streamlit as st
from ..charts import combined_forecast_line, forecast_lines
from ..theme import page_header


def render(repo, origin, fsas, data):
    st.markdown(
        page_header(
            "Decision Overview",
            "What is expected, when it may occur, and where attention is required during the next 24 hours.",
        ),
        unsafe_allow_html=True,
    )
    metrics = repo.executive_metrics(data)
    
    c1, c2, c3, c4, c5, c6  = st.columns(6)
    c1.metric(
        # "Maximum combined selected-FSA demand",
        "Total Daily Demand",
        f"{metrics['maximum_combined_demand_kwh']:,.0f} kWh",
        help="Sum of forecast demand across the selected FSAs at each target hour; this is not total Ontario demand.",
    )
    c2.metric(
        # "Maximum FSA-level demand",
        "Maximum FSA demand",
        f"{metrics['maximum_fsa_demand_kwh']:,.0f} kWh",
        help=f"Highest individual forecast among selected FSAs: {metrics['maximum_demand_fsa']}.",
    )
    c3.metric(
        "Unique alert hours",
        metrics["unique_alert_hours"],
        help="Distinct target hours with at least one selected FSA in alert.",
    )
    c4.metric(
        "FSA-hour alerts",
        metrics["fsa_hour_alerts"],
        help="Total alert combinations across selected FSAs and the 24 forecast hours.",
    )

    # c5, c6 = st.columns(2)
    c5.metric(
        "Max Demand FSA",
        metrics["maximum_demand_fsa"],
        help="Selected FSA containing the maximum individual FSA-hour demand forecast.",
    )
    c6.metric(
        "Highest-risk FSA",
        metrics["highest_risk_fsa"],
        help=f"Selected FSA with the highest Peak-Risk score ({metrics['highest_risk_score']:.3f}).",
    )

    st.caption(
        f"Combined-demand maximum: {metrics['maximum_combined_demand_hour']:%b %d, %Y %H:%M} · "
        f"Maximum FSA-level demand: {metrics['maximum_demand_fsa']} at "
        f"{metrics['maximum_fsa_demand_hour']:%b %d, %Y %H:%M} · "
        f"Official Peak-Risk threshold: {repo.threshold:.2f}"
    )

    st.plotly_chart(forecast_lines(data), width="stretch")
    st.caption("Red diamond markers identify FSA-hour forecasts classified as Peak-Risk alerts.")

    if not metrics:
        st.warning("No prediction data for the selected analysis.")
        return
    if metrics["fsa_hour_alerts"]:
        st.markdown(
            '<div class="risk-alert">'
            '<div class="risk-alert-title">Peak-Risk alert detected</div>'
            '<div>'
            f'{metrics["fsa_hour_alerts"]} FSA-hour alert(s) occur across '
            f'{metrics["unique_alert_hours"]} unique target hour(s). The highest score is '
            f'{metrics["highest_risk_score"]:.3f} for {metrics["highest_risk_fsa"]} at '
            f'{metrics["highest_risk_hour"]:%b %d, %Y %H:%M}.</div>'
            '<div class="risk-alert-actions"><strong>Suggested actions</strong>'
            '<ul>'
            '<li>Coordinate operational actions for the flagged FSAs and target hours.</li>'
            '<li>Review capacity, staffing and demand-response readiness before the highest-risk period.</li>'
            '<li>Validate the latest demand and weather inputs and monitor the forecast for material changes.</li>'
            '</ul></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-normal"><strong>No Peak-Risk alerts detected.</strong> '
            'All selected FSA-hour scores remain below the official decision threshold.</div>',
            unsafe_allow_html=True,
        )


    st.plotly_chart(combined_forecast_line(data), width="stretch")
    st.caption("Red diamond markers identify target hours with at least one selected FSA in Peak-Risk alert.")

    summary = data.groupby("fsa").agg(
        maximum_fsa_demand_kwh=("forecast_consumption_kwh", "max"),
        maximum_peak_risk_score=("peak_risk_score", "max"),
        fsa_hour_alerts=("peak_alert", "sum"),
    ).reset_index().sort_values("maximum_peak_risk_score", ascending=False)
    st.subheader("Selected-FSA decision summary")
    st.dataframe(summary, width="stretch", hide_index=True)
