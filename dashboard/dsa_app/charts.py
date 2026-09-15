"""Reusable Plotly figures. No Streamlit calls live in this module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .theme import COLORS, FSA_COLORS, style_figure


DEMAND_HEATMAP_SCALE = [
    [0.00, "#FFFFFF"],
    [0.30, "#FEE2E2"],
    [0.60, "#FCA5A5"],
    [0.82, "#EF4444"],
    [1.00, "#991B1B"],
]


def _ordered(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(["fsa", "horizon"]).copy()


def demand_heatmap(data: pd.DataFrame):
    pivot = data.pivot(index="fsa", columns="target_timestamp", values="forecast_consumption_kwh")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=DEMAND_HEATMAP_SCALE,
        labels={"x": "Target date and time", "y": "FSA", "color": "Demand (kWh)"},
        title="Forecast demand by FSA and target hour",
    )
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=420)


def risk_heatmap(data: pd.DataFrame, threshold: float):
    pivot = data.pivot(index="fsa", columns="target_timestamp", values="peak_risk_score")
    fig = px.imshow(
        pivot,
        aspect="auto",
        zmin=0,
        zmax=1,
        color_continuous_scale=[[0, "#F4F8FC"], [threshold, "#F2A900"], [1, "#D64545"]],
        labels={"x": "Target date and time", "y": "FSA", "color": "Peak-Risk score"},
        title=f"Peak-Risk score by FSA and target hour — alert threshold {threshold:.2f}",
    )
    alerts = data[data["peak_risk_score"].ge(threshold)]
    for _, row in alerts.iterrows():
        fig.add_annotation(x=row["target_timestamp"], y=row["fsa"], text="◆", showarrow=False, font=dict(size=11))
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=420)


def combined_decision_heatmap(data: pd.DataFrame, threshold: float):
    risk = data.pivot(index="fsa", columns="target_timestamp", values="peak_risk_score")
    demand = data.pivot(index="fsa", columns="target_timestamp", values="forecast_consumption_kwh")
    alert = data.pivot(index="fsa", columns="target_timestamp", values="peak_alert").astype(bool)
    text = np.empty(risk.shape, dtype=object)
    for i in range(risk.shape[0]):
        for j in range(risk.shape[1]):
            mark = " ◆" if alert.iloc[i, j] else ""
            text[i, j] = f"{demand.iloc[i, j]:,.0f} kWh{mark}"
    fig = go.Figure(go.Heatmap(
        z=risk.values,
        x=risk.columns,
        y=risk.index,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#F4F8FC"], [threshold, "#F2A900"], [1, "#D64545"]],
        text=text,
        texttemplate="%{text}",
        hovertemplate="FSA=%{y}<br>Target=%{x|%b %d, %Y %H:%M}<br>Peak-Risk score=%{z:.3f}<br>%{text}<extra></extra>",
        colorbar=dict(title="Peak-Risk score"),
    ))
    fig.update_layout(
        title=f"Decision heatmap — color = Peak-Risk score, label = demand, ◆ = alert (≥ {threshold:.2f})",
        xaxis_title="Target date and time",
        yaxis_title="FSA",
        height=430,
    )
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=440)


def demand_decision_heatmap(data: pd.DataFrame, threshold: float):
    """Decision heatmap colored by demand without changing the risk heatmap."""
    demand = data.pivot(index="fsa", columns="target_timestamp", values="forecast_consumption_kwh")
    risk = data.pivot(index="fsa", columns="target_timestamp", values="peak_risk_score")
    alert = data.pivot(index="fsa", columns="target_timestamp", values="peak_alert").astype(bool)
    text = np.empty(demand.shape, dtype=object)
    customdata = np.empty((*demand.shape, 2), dtype=object)

    for i in range(demand.shape[0]):
        for j in range(demand.shape[1]):
            mark = " ◆" if alert.iloc[i, j] else ""
            text[i, j] = f"{demand.iloc[i, j]:,.0f} kWh{mark}"
            customdata[i, j, 0] = float(risk.iloc[i, j])
            customdata[i, j, 1] = "Alert" if alert.iloc[i, j] else "No alert"

    fig = go.Figure(go.Heatmap(
        z=demand.values,
        x=demand.columns,
        y=demand.index,
        colorscale=DEMAND_HEATMAP_SCALE,
        text=text,
        texttemplate="%{text}",
        customdata=customdata,
        hovertemplate=(
            "FSA=%{y}<br>Target=%{x|%b %d, %Y %H:%M}<br>"
            "Demand=%{z:,.0f} kWh<br>Peak-Risk score=%{customdata[0]:.3f}<br>"
            "Decision=%{customdata[1]}<extra></extra>"
        ),
        colorbar=dict(title="Demand (kWh)"),
    ))
    fig.update_layout(
        title=f"Demand decision heatmap — ◆ = Peak-Risk alert (score ≥ {threshold:.2f})",
        xaxis_title="Target date and time",
        yaxis_title="FSA",
    )
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=440)


def forecast_lines(data: pd.DataFrame):
    data = _ordered(data)
    fig = px.line(
        data,
        x="target_timestamp",
        y="forecast_consumption_kwh",
        color="fsa",
        markers=True,
        color_discrete_sequence=FSA_COLORS,
        labels={"target_timestamp": "Target date and time", "forecast_consumption_kwh": "Forecast demand (kWh)", "fsa": "FSA"},
        title="24-hour demand forecast by selected FSA",
    )
    alerts = data[data["peak_alert"].astype(bool)]
    if not alerts.empty:
        fig.add_trace(go.Scatter(
            x=alerts["target_timestamp"],
            y=alerts["forecast_consumption_kwh"],
            mode="markers",
            marker=dict(symbol="diamond", size=11, color=COLORS["alert"], line=dict(color="white", width=1)),
            name="Peak-Risk alert",
            customdata=alerts[["fsa", "target_timestamp", "peak_risk_score"]],
            hovertemplate="FSA=%{customdata[0]}<br>Target=%{x|%b %d, %Y %H:%M}<br>Demand=%{y:,.0f} kWh<br>Peak-Risk score=%{customdata[2]:.3f}<extra></extra>",
        ))
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=460)


def combined_forecast_line(data: pd.DataFrame):
    """Total forecast demand for the currently selected FSA set."""
    combined = data.groupby("target_timestamp", as_index=False).agg(
        combined_demand_kwh=("forecast_consumption_kwh", "sum"),
        has_peak_risk_alert=("peak_alert", lambda x: x.astype(bool).any()),
    )
    fig = px.area(
        combined,
        x="target_timestamp",
        y="combined_demand_kwh",
        markers=True,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"target_timestamp": "Target date and time", "combined_demand_kwh": "Combined demand (kWh)"},
        title="Combined demand forecast for selected FSAs",
    )
    alert_hours = combined[combined["has_peak_risk_alert"]]
    if not alert_hours.empty:
        fig.add_trace(go.Scatter(
            x=alert_hours["target_timestamp"],
            y=alert_hours["combined_demand_kwh"],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color=COLORS["alert"], line=dict(color="white", width=1)),
            name="Peak-Risk alert hour",
            hovertemplate=(
                "Alert hour=%{x|%b %d, %Y %H:%M}<br>"
                "Combined demand=%{y:,.0f} kWh<extra></extra>"
            ),
        ))
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=420)


def risk_timeline(data: pd.DataFrame, threshold: float):
    data = _ordered(data)
    fig = px.line(
        data,
        x="target_timestamp",
        y="peak_risk_score",
        color="fsa",
        markers=True,
        color_discrete_sequence=FSA_COLORS,
        labels={"target_timestamp": "Target date and time", "peak_risk_score": "Peak-Risk score", "fsa": "FSA"},
        title="Peak-Risk score by target hour",
    )
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    fig.update_yaxes(range=[0, max(1.0, float(data["peak_risk_score"].max()) * 1.05)])
    return style_figure(fig, height=450)


def ranking_max_risk(data: pd.DataFrame):
    ranked = data.groupby("fsa", as_index=False)["peak_risk_score"].max().sort_values("peak_risk_score")
    fig = px.bar(ranked, x="peak_risk_score", y="fsa", orientation="h", color_discrete_sequence=[COLORS["warning"]], title="FSA ranking — maximum Peak-Risk score", labels={"peak_risk_score":"Maximum Peak-Risk score", "fsa":"FSA"})
    return style_figure(fig, height=390)


def ranking_max_demand(data: pd.DataFrame):
    ranked = data.groupby("fsa", as_index=False)["forecast_consumption_kwh"].max().sort_values("forecast_consumption_kwh")
    fig = px.bar(ranked, x="forecast_consumption_kwh", y="fsa", orientation="h", color_discrete_sequence=[COLORS["primary"]], title="FSA ranking — maximum FSA-level demand", labels={"forecast_consumption_kwh":"Maximum FSA-level demand (kWh)", "fsa":"FSA"})
    return style_figure(fig, height=390)


def alert_hours_bar(data: pd.DataFrame):
    counts = data.assign(_alert=data["peak_alert"].astype(bool).astype(int)).groupby("fsa", as_index=False)["_alert"].sum().sort_values("_alert")
    fig = px.bar(counts, x="_alert", y="fsa", orientation="h", color_discrete_sequence=[COLORS["alert"]], title="FSA-hour alerts by FSA", labels={"_alert":"FSA-hour alerts in next 24 hours", "fsa":"FSA"})
    return style_figure(fig, height=390)


def scenario_forecast_lines(data: pd.DataFrame):
    fig = px.line(data, x="target_timestamp", y="forecast_consumption_kwh", color="fsa", markers=True, color_discrete_sequence=FSA_COLORS, title="Selected temperature-scenario demand forecast", labels={"target_timestamp":"Target date and time","forecast_consumption_kwh":"Demand (kWh)","fsa":"FSA"})
    new_alerts = data[
        data["peak_alert"].astype(bool)
        & ~data["baseline_peak_alert"].astype(bool)
    ]
    if not new_alerts.empty:
        fig.add_trace(go.Scatter(
            x=new_alerts["target_timestamp"],
            y=new_alerts["forecast_consumption_kwh"],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color=COLORS["alert"], line=dict(color="white", width=1)),
            name="New alert vs Baseline",
            customdata=new_alerts[["fsa", "peak_risk_score"]],
            hovertemplate=(
                "FSA=%{customdata[0]}<br>Target=%{x|%b %d, %Y %H:%M}<br>"
                "Scenario demand=%{y:,.0f} kWh<br>Peak-Risk score=%{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        ))
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=440)


def scenario_delta_heatmap(data: pd.DataFrame):
    pivot = data.pivot(index="fsa", columns="target_timestamp", values="forecast_delta_kwh")
    bound = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1e-9)
    fig = px.imshow(pivot, aspect="auto", zmin=-bound, zmax=bound, color_continuous_scale="RdBu_r", labels={"x":"Target date and time","y":"FSA","color":"Δ Demand (kWh)"}, title="Demand change vs Baseline")
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=410)


def scenario_risk_delta_heatmap(data: pd.DataFrame):
    pivot = data.pivot(index="fsa", columns="target_timestamp", values="peak_risk_delta")
    bound = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1e-9)
    fig = px.imshow(pivot, aspect="auto", zmin=-bound, zmax=bound, color_continuous_scale="RdBu_r", labels={"x":"Target date and time","y":"FSA","color":"Δ Peak-Risk score"}, title="Peak-Risk score change vs Baseline")
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=410)


def baseline_scenario_demand_lines(data: pd.DataFrame, fsa: str):
    d = data[data["fsa"].eq(fsa)].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["target_timestamp"], y=d["baseline_forecast_kwh"], mode="lines+markers", name="Baseline", line=dict(color=COLORS["muted"])))
    fig.add_trace(go.Scatter(x=d["target_timestamp"], y=d["forecast_consumption_kwh"], mode="lines+markers", name=str(d["scenario"].iloc[0]), line=dict(color=COLORS["warning"])))
    fig.update_layout(title=f"Demand sensitivity — {fsa}: Baseline vs selected scenario", xaxis_title="Target date and time", yaxis_title="Demand (kWh)")
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    return style_figure(fig, height=420)


def baseline_scenario_risk_lines(data: pd.DataFrame, fsa: str, threshold: float):
    d = data[data["fsa"].eq(fsa)].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["target_timestamp"], y=d["baseline_peak_risk_score"], mode="lines+markers", name="Baseline", line=dict(color=COLORS["muted"])))
    fig.add_trace(go.Scatter(x=d["target_timestamp"], y=d["peak_risk_score"], mode="lines+markers", name=str(d["scenario"].iloc[0]), line=dict(color=COLORS["warning"])))
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_layout(title=f"Peak-Risk sensitivity — {fsa}: Baseline vs selected scenario", xaxis_title="Target date and time", yaxis_title="Peak-Risk score")
    fig.update_xaxes(tickformat="%b %d\n%H:%M", dtick=3 * 60 * 60 * 1000)
    fig.update_yaxes(range=[0, 1])
    return style_figure(fig, height=420)


def scenario_alert_changes(data: pd.DataFrame):
    d = data.assign(_changed=data["alert_changed"].astype(bool).astype(int)).groupby("fsa", as_index=False)["_changed"].sum().sort_values("_changed")
    fig = px.bar(d, x="_changed", y="fsa", orientation="h", color_discrete_sequence=[COLORS["alert"]], title="Alert-decision changes vs Baseline", labels={"_changed":"Changed FSA-hour decisions", "fsa":"FSA"})
    return style_figure(fig, height=370)


def all_scenarios_summary_chart(all_data: pd.DataFrame):
    s = all_data.groupby(["scenario", "temperature_delta_c"], as_index=False).agg(mean_abs_demand_change=("forecast_delta_kwh", lambda x: x.abs().mean()), alert_changes=("alert_changed", lambda x: x.astype(bool).sum())).sort_values("temperature_delta_c")
    fig = px.bar(s, x="scenario", y="mean_abs_demand_change", color_discrete_sequence=[COLORS["primary"]], title="Scenario comparison — mean absolute demand change", labels={"scenario":"Temperature scenario", "mean_abs_demand_change":"Mean |Δ demand| (kWh)"})
    return style_figure(fig, height=400)
