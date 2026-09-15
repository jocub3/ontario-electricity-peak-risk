from __future__ import annotations
import plotly.express as px
import streamlit as st
from ..data import load_config, load_optional_csv
from ..theme import COLORS, page_header, style_figure


def render(repo, origin, fsas, data):
    st.markdown(
        page_header(
            "Model & Methodology",
            "Technical evidence supporting the forecasting and Peak-Risk decision system.",
        ),
        unsafe_allow_html=True,
    )
    cfg = load_config()
    tab1, tab2, tab3 = st.tabs(["Explainability", "Final Performance", "Frozen System"])
    with tab1:
        global_imp = load_optional_csv(cfg["paths"]["explainability_global"])
        family_imp = load_optional_csv(cfg["paths"]["explainability_family"])
        if not global_imp.empty and {"task","mean_importance_pct"}.issubset(global_imp.columns):
            st.subheader("Top global features")
            top = global_imp.sort_values(["task","mean_importance_pct"], ascending=[True,False]).groupby("task", observed=True).head(10)
            for task, task_data in top.groupby("task", observed=True):
                chart_data = task_data.sort_values("mean_importance_pct")
                fig = px.bar(
                    chart_data,
                    x="mean_importance_pct",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=[COLORS["primary"] if task == "rf" else COLORS["warning"]],
                    labels={"mean_importance_pct": "Mean importance (%)", "feature": "Feature"},
                    title=f"{task.upper()} — top global features",
                )
                st.plotly_chart(style_figure(fig, height=420), width="stretch")
            with st.expander("View feature-importance data"):
                st.dataframe(top, width="stretch", hide_index=True)
        else:
            st.info("Global explainability artifact is unavailable in this deployment package.")
        if not family_imp.empty:
            with st.expander("View feature-family importance by horizon"):
                st.dataframe(family_imp, width="stretch", hide_index=True, height=360)
        st.caption("Feature importance and SHAP explain fitted-model behavior; they do not establish causality.")
    with tab2:
        c1, c2 = st.columns(2)
        c1.markdown("""**Random Forest — protected 2025 holdout**\n- MAE: **540.52 kWh**\n- RMSE: **927.03 kWh**\n- MAPE: **5.42%**\n- WAPE: **5.59%**\n- Bias: **-53.18 kWh**""")
        c2.markdown("""**XGBoost Peak-Risk — threshold 0.06**\n- Precision: **0.341**\n- Recall: **0.930**\n- F1: **0.499**\n- Balanced Accuracy: **0.898**\n- PR-AUC: **0.712**\n- ROC-AUC: **0.964**\n- Brier: **0.0428**""")
        st.info(
            "The frozen 0.06 decision threshold produces high recall (0.930) and lower precision (0.341) on the reported evaluation. "
            "Operationally, this operating point identifies most observed peaks but also produces more false alerts."
        )
    with tab3:
        st.markdown("""
**Frozen production pair**
- 24 direct Random Forest regressors for h+1...h+24 demand forecasting.
- 24 XGBoost classifiers for h+1...h+24 Peak-Risk scoring.
- XGBoost does **not** consume the RF forecast as an input.
- Official Peak-Risk decision threshold: **0.06**.
- The public Streamlit deployment uses validated precomputed outputs to provide lightweight, interactive access to the system results.
""")
