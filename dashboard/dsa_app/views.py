"""Streamlit view functions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .charts import (
    demand_heatmap,
    risk_heatmap,
    forecast_lines,
    risk_timeline,
    scenario_forecast_lines,
    scenario_delta_heatmap,
)
from .data import (
    ROOT,
    load_config,
    load_optional_csv,
    load_optional_markdown,
)


def render_input_panel(repo):
    st.header("1. Forecast Selection & Input Validation")
    st.info(
        "Public Demo Mode uses precomputed outputs from the validated local "
        "pipeline. No 13 GB model bundle is loaded by this web application."
    )

    origins = repo.available_origins
    if not origins:
        st.error("No precomputed forecast origins are available.")
        st.stop()

    labels = {
        pd.Timestamp(x).strftime("%Y-%m-%d %H:%M"): pd.Timestamp(x)
        for x in origins
    }
    selected_label = st.selectbox("Forecast origin", list(labels))
    selected_origin = labels[selected_label]

    selected_fsas = st.multiselect(
        "FSAs",
        repo.fsas,
        default=repo.fsas,
    )
    if not selected_fsas:
        st.warning("Select at least one FSA.")
        st.stop()

    st.caption(
        "Operational implementation: recent observed demand and weather forecast "
        "files are loaded from `data/operational/` and validated by the IFB before "
        "model execution. This public deployment uses their precomputed results."
    )
    return selected_origin, selected_fsas


def render_executive(repo, data):
    st.header("2. Executive Summary")
    metrics = repo.executive_metrics(data)
    if not metrics:
        st.warning("No prediction data for the selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Expected Maximum Demand",
        f"{metrics['expected_maximum_demand_kwh']:,.0f} kWh",
    )
    c2.metric(
        "Highest-Risk FSA",
        metrics["highest_risk_fsa"],
    )
    c3.metric(
        "Highest-Risk Hour",
        metrics["highest_risk_hour"].strftime("%Y-%m-%d %H:%M"),
    )
    c4.metric(
        "Peak-Risk Alerts",
        f"{metrics['peak_risk_alerts']}",
    )

    st.plotly_chart(forecast_lines(data), use_container_width=True)

    summary = (
        data.groupby("fsa")
        .agg(
            max_demand_kwh=("forecast_consumption_kwh", "max"),
            max_peak_risk=("peak_risk_score", "max"),
            alert_hours=("peak_alert", "sum"),
        )
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_forecast_risk(repo, data):
    st.header("3. Forecast & Peak-Risk Monitor")
    tab1, tab2, tab3 = st.tabs(
        ["Heatmaps", "24-hour Forecast", "Peak-Risk Monitor"]
    )

    with tab1:
        st.plotly_chart(demand_heatmap(data), use_container_width=True)
        st.plotly_chart(
            risk_heatmap(data, repo.threshold),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(forecast_lines(data), use_container_width=True)
        table = data[
            [
                "fsa",
                "target_timestamp",
                "horizon",
                "forecast_consumption_kwh",
                "peak_risk_score",
                "peak_alert",
            ]
        ].copy()
        st.dataframe(table, use_container_width=True, hide_index=True)

    with tab3:
        st.plotly_chart(
            risk_timeline(data, repo.threshold),
            use_container_width=True,
        )
        risk = data.sort_values("peak_risk_score", ascending=False).copy()
        st.dataframe(
            risk[
                [
                    "fsa",
                    "target_timestamp",
                    "forecast_consumption_kwh",
                    "peak_risk_score",
                    "peak_alert",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Official Peak-Risk threshold: {repo.threshold:.2f}. "
            "The threshold is a decision rule applied to the frozen XGBoost score."
        )


def render_scenarios(repo, origin, fsas):
    st.header("4. Weather Sensitivity & Scenario Comparison")
    if repo.scenarios.empty:
        st.warning(
            "No precomputed scenario artifact is available. Run the DSA demo "
            "scenario builder first."
        )
        return

    available = (
        repo.scenarios.loc[
            repo.scenarios["forecast_origin"].eq(pd.Timestamp(origin)),
            ["scenario", "temperature_delta_c"],
        ]
        .drop_duplicates()
        .sort_values("temperature_delta_c")
    )
    if available.empty:
        st.warning("No scenario results exist for this forecast origin.")
        return

    scenario = st.select_slider(
        "Temperature scenario",
        options=available["scenario"].astype(str).tolist(),
        value=(
            "Baseline"
            if "Baseline" in available["scenario"].astype(str).tolist()
            else available["scenario"].astype(str).iloc[0]
        ),
    )

    data = repo.scenario_slice(origin, scenario, fsas)
    st.caption(
        "Model-v1 sensitivity: only forecast-origin temperature is perturbed. "
        "The results are not causal and are not target-hour future-weather forecasts."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Δ Demand", f"{data['forecast_delta_kwh'].mean():,.1f} kWh")
    c2.metric("Mean Δ Peak-Risk", f"{data['peak_risk_delta'].mean():.3f}")
    c3.metric("Alert Decisions Changed", int(data["alert_changed"].astype(bool).sum()))

    st.plotly_chart(scenario_forecast_lines(data), use_container_width=True)
    st.plotly_chart(scenario_delta_heatmap(data), use_container_width=True)

    comparison = (
        data.groupby("fsa")
        .agg(
            base_max_demand=("baseline_forecast_kwh", "max"),
            scenario_max_demand=("forecast_consumption_kwh", "max"),
            mean_demand_change=("forecast_delta_kwh", "mean"),
            base_alerts=("baseline_peak_alert", "sum"),
            scenario_alerts=("peak_alert", "sum"),
        )
        .reset_index()
    )
    st.subheader("Scenario Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_model_insights(repo):
    st.header("5. Model Insights")
    cfg = load_config()
    tab1, tab2, tab3 = st.tabs(
        ["Explainability", "Model Performance", "Model Comparison"]
    )

    with tab1:
        global_imp = load_optional_csv(cfg["paths"]["explainability_global"])
        family_imp = load_optional_csv(cfg["paths"]["explainability_family"])

        if not global_imp.empty:
            st.subheader("Global feature importance")
            top = (
                global_imp.sort_values(
                    ["task", "mean_importance_pct"],
                    ascending=[True, False],
                )
                .groupby("task", observed=True)
                .head(10)
            )
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.info("Global explainability CSV is not available.")

        if not family_imp.empty:
            st.subheader("Feature-family importance")
            st.dataframe(family_imp, use_container_width=True, hide_index=True)

        st.caption(
            "Feature importance and SHAP describe fitted-model behavior; "
            "they do not establish causal effects."
        )

    with tab2:
        st.markdown(
            """
**Final protected 2025 holdout**

**Random Forest forecasting**
- MAE: 540.52 kWh
- RMSE: 927.03 kWh
- MAPE: 5.42%
- WAPE: 5.59%
- Bias: -53.18 kWh

**XGBoost Peak-Risk**
- Threshold: 0.06
- Precision: 0.341
- Recall: 0.930
- F1: 0.499
- Balanced Accuracy: 0.898
- PR-AUC: 0.712
- ROC-AUC: 0.964
- Brier Score: 0.0428
"""
        )
        st.caption(
            "These values are the frozen project's protected 2025 holdout results; "
            "the dashboard does not retune models."
        )

    with tab3:
        st.markdown(
            """
The project compared multiple forecasting and Peak-Risk candidates before
freezing the final system.

**Final production pair**
- Forecasting: Random Forest Regressor
- Peak-Risk: XGBoost Classifier
- Operational Peak-Risk threshold: 0.06

Model-selection evidence remains in the project's modeling/model-comparison
outputs and documentation. The public application consumes the final frozen
system rather than reopening model selection.
"""
        )


def render_project_info(repo):
    st.header("6. Project & Deployment Information")
    st.markdown(
        """
### Architecture

**Local / academic pipeline**

Operational demand + weather → IFB validation → 24 RF models + 24 XGBoost
models → forecast / Peak-Risk / scenario outputs.

**Public portfolio application**

Precomputed Parquet/JSON → lightweight Streamlit application.

The public application intentionally does **not** deploy the approximately
13 GB Random Forest artifact bundle. This keeps deployment lightweight while
preserving an interactive demonstration of the validated decision-support
outputs.
"""
    )
    st.markdown(
        """
### Interpretation boundaries

- Forecasts cover h+1 through h+24.
- Peak-Risk is a probability/score plus the official 0.06 decision rule.
- Weather scenarios perturb only forecast-origin temperature in Model v1.
- Scenario analysis is model sensitivity, not causal weather inference.
- Out-of-domain inputs should be accompanied by reliability warnings.
"""
    )
