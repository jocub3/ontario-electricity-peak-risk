"""Reusable Plotly figures. No Streamlit calls live in this module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _ordered(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values(["fsa", "horizon"]).copy()


def demand_heatmap(data: pd.DataFrame):
    pivot = data.pivot(index="fsa", columns="horizon", values="forecast_consumption_kwh")
    fig = px.imshow(
        pivot,
        aspect="auto",
        labels={"x": "Forecast horizon (h+)", "y": "FSA", "color": "Demand (kWh)"},
        title="Forecast electricity demand",
    )
    fig.update_xaxes(dtick=1)
    return fig


def risk_heatmap(data: pd.DataFrame, threshold: float):
    pivot = data.pivot(index="fsa", columns="horizon", values="peak_risk_score")
    fig = px.imshow(
        pivot,
        aspect="auto",
        zmin=0,
        zmax=1,
        labels={"x": "Forecast horizon (h+)", "y": "FSA", "color": "Peak-Risk score"},
        title=f"Peak-Risk probability / score — threshold {threshold:.2f}",
    )
    alerts = data[data["peak_risk_score"].ge(threshold)]
    for _, row in alerts.iterrows():
        fig.add_annotation(x=row["horizon"], y=row["fsa"], text="◆", showarrow=False, font=dict(size=11))
    fig.update_xaxes(dtick=1)
    return fig


def combined_decision_heatmap(data: pd.DataFrame, threshold: float):
    risk = data.pivot(index="fsa", columns="horizon", values="peak_risk_score")
    demand = data.pivot(index="fsa", columns="horizon", values="forecast_consumption_kwh")
    alert = data.pivot(index="fsa", columns="horizon", values="peak_alert").astype(bool)
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
        text=text,
        texttemplate="%{text}",
        hovertemplate="FSA=%{y}<br>h+%{x}<br>Risk=%{z:.3f}<br>%{text}<extra></extra>",
        colorbar=dict(title="Peak-Risk"),
    ))
    fig.update_layout(
        title=f"Combined decision heatmap — color = Peak-Risk, label = demand, ◆ = alert (≥ {threshold:.2f})",
        xaxis_title="Forecast horizon (h+)",
        yaxis_title="FSA",
        height=430,
    )
    fig.update_xaxes(dtick=1)
    return fig


def forecast_lines(data: pd.DataFrame):
    data = _ordered(data)
    fig = px.line(
        data,
        x="horizon",
        y="forecast_consumption_kwh",
        color="fsa",
        markers=True,
        labels={"horizon": "Forecast horizon (h+)", "forecast_consumption_kwh": "Forecast demand (kWh)", "fsa": "FSA"},
        title="24-hour electricity-demand forecast — all selected FSAs",
    )
    alerts = data[data["peak_alert"].astype(bool)]
    if not alerts.empty:
        fig.add_trace(go.Scatter(
            x=alerts["horizon"],
            y=alerts["forecast_consumption_kwh"],
            mode="markers",
            marker=dict(symbol="diamond", size=11),
            name="Peak-Risk alert",
            customdata=alerts[["fsa", "target_timestamp", "peak_risk_score"]],
            hovertemplate="FSA=%{customdata[0]}<br>h+%{x}<br>Demand=%{y:,.0f} kWh<br>Risk=%{customdata[2]:.3f}<extra></extra>",
        ))
    fig.update_xaxes(dtick=1)
    return fig


def risk_timeline(data: pd.DataFrame, threshold: float):
    data = _ordered(data)
    fig = px.line(
        data,
        x="horizon",
        y="peak_risk_score",
        color="fsa",
        markers=True,
        labels={"horizon": "Forecast horizon (h+)", "peak_risk_score": "Peak-Risk score", "fsa": "FSA"},
        title="Peak-Risk score by forecast hour",
    )
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, max(1.0, float(data["peak_risk_score"].max()) * 1.05)])
    return fig


def ranking_max_risk(data: pd.DataFrame):
    ranked = data.groupby("fsa", as_index=False)["peak_risk_score"].max().sort_values("peak_risk_score")
    return px.bar(ranked, x="peak_risk_score", y="fsa", orientation="h", title="FSA ranking — maximum Peak-Risk", labels={"peak_risk_score":"Maximum Peak-Risk", "fsa":"FSA"})


def ranking_max_demand(data: pd.DataFrame):
    ranked = data.groupby("fsa", as_index=False)["forecast_consumption_kwh"].max().sort_values("forecast_consumption_kwh")
    return px.bar(ranked, x="forecast_consumption_kwh", y="fsa", orientation="h", title="FSA ranking — maximum expected demand", labels={"forecast_consumption_kwh":"Maximum demand (kWh)", "fsa":"FSA"})


def alert_hours_bar(data: pd.DataFrame):
    counts = data.assign(_alert=data["peak_alert"].astype(bool).astype(int)).groupby("fsa", as_index=False)["_alert"].sum().sort_values("_alert")
    return px.bar(counts, x="_alert", y="fsa", orientation="h", title="Peak-Risk alert hours by FSA", labels={"_alert":"Alert hours in next 24h", "fsa":"FSA"})


def scenario_forecast_lines(data: pd.DataFrame):
    fig = px.line(data, x="horizon", y="forecast_consumption_kwh", color="fsa", markers=True, title="Selected temperature-scenario demand forecast", labels={"horizon":"Forecast horizon (h+)","forecast_consumption_kwh":"Demand (kWh)","fsa":"FSA"})
    fig.update_xaxes(dtick=1)
    return fig


def scenario_delta_heatmap(data: pd.DataFrame):
    pivot = data.pivot(index="fsa", columns="horizon", values="forecast_delta_kwh")
    bound = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1e-9)
    fig = px.imshow(pivot, aspect="auto", zmin=-bound, zmax=bound, color_continuous_scale="RdBu_r", labels={"x":"Forecast horizon (h+)","y":"FSA","color":"Δ Demand (kWh)"}, title="Demand change vs Baseline")
    fig.update_xaxes(dtick=1)
    return fig


def scenario_risk_delta_heatmap(data: pd.DataFrame):
    pivot = data.pivot(index="fsa", columns="horizon", values="peak_risk_delta")
    bound = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())), 1e-9)
    fig = px.imshow(pivot, aspect="auto", zmin=-bound, zmax=bound, color_continuous_scale="RdBu_r", labels={"x":"Forecast horizon (h+)","y":"FSA","color":"Δ Peak-Risk"}, title="Peak-Risk change vs Baseline")
    fig.update_xaxes(dtick=1)
    return fig


def baseline_scenario_demand_lines(data: pd.DataFrame, fsa: str):
    d = data[data["fsa"].eq(fsa)].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["horizon"], y=d["baseline_forecast_kwh"], mode="lines+markers", name="Baseline"))
    fig.add_trace(go.Scatter(x=d["horizon"], y=d["forecast_consumption_kwh"], mode="lines+markers", name=str(d["scenario"].iloc[0])))
    fig.update_layout(title=f"Demand sensitivity — {fsa}: Baseline vs selected scenario", xaxis_title="Forecast horizon (h+)", yaxis_title="Demand (kWh)")
    fig.update_xaxes(dtick=1)
    return fig


def baseline_scenario_risk_lines(data: pd.DataFrame, fsa: str, threshold: float):
    d = data[data["fsa"].eq(fsa)].sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["horizon"], y=d["baseline_peak_risk_score"], mode="lines+markers", name="Baseline"))
    fig.add_trace(go.Scatter(x=d["horizon"], y=d["peak_risk_score"], mode="lines+markers", name=str(d["scenario"].iloc[0])))
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_layout(title=f"Peak-Risk sensitivity — {fsa}: Baseline vs selected scenario", xaxis_title="Forecast horizon (h+)", yaxis_title="Peak-Risk score")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 1])
    return fig


def scenario_alert_changes(data: pd.DataFrame):
    d = data.assign(_changed=data["alert_changed"].astype(bool).astype(int)).groupby("fsa", as_index=False)["_changed"].sum().sort_values("_changed")
    return px.bar(d, x="_changed", y="fsa", orientation="h", title="Alert-decision changes vs Baseline", labels={"_changed":"Changed hourly decisions", "fsa":"FSA"})


def all_scenarios_summary_chart(all_data: pd.DataFrame):
    s = all_data.groupby(["scenario", "temperature_delta_c"], as_index=False).agg(mean_abs_demand_change=("forecast_delta_kwh", lambda x: x.abs().mean()), alert_changes=("alert_changed", lambda x: x.astype(bool).sum())).sort_values("temperature_delta_c")
    return px.bar(s, x="scenario", y="mean_abs_demand_change", title="Scenario comparison — mean absolute demand change", labels={"scenario":"Temperature scenario", "mean_abs_demand_change":"Mean |Δ demand| (kWh)"})
