from __future__ import annotations
import streamlit as st
from ..data import load_config, load_optional_csv


def render(repo, origin, fsas, data):
    st.header("Model Insights")
    cfg = load_config()
    tab1, tab2, tab3 = st.tabs(["Explainability", "Final Performance", "Frozen System"])
    with tab1:
        global_imp = load_optional_csv(cfg["paths"]["explainability_global"])
        family_imp = load_optional_csv(cfg["paths"]["explainability_family"])
        if not global_imp.empty and {"task","mean_importance_pct"}.issubset(global_imp.columns):
            st.subheader("Top global features")
            top = global_imp.sort_values(["task","mean_importance_pct"], ascending=[True,False]).groupby("task", observed=True).head(10)
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.info("Global explainability artifact is unavailable in this deployment package.")
        if not family_imp.empty:
            st.subheader("Feature-family importance by horizon")
            st.dataframe(family_imp, use_container_width=True, hide_index=True, height=360)
        st.caption("Feature importance and SHAP explain fitted-model behavior; they do not establish causality.")
    with tab2:
        c1, c2 = st.columns(2)
        c1.markdown("""**Random Forest — protected 2025 holdout**\n- MAE: **540.52 kWh**\n- RMSE: **927.03 kWh**\n- MAPE: **5.42%**\n- WAPE: **5.59%**\n- Bias: **-53.18 kWh**""")
        c2.markdown("""**XGBoost Peak-Risk — threshold 0.06**\n- Precision: **0.341**\n- Recall: **0.930**\n- F1: **0.499**\n- Balanced Accuracy: **0.898**\n- PR-AUC: **0.712**\n- ROC-AUC: **0.964**\n- Brier: **0.0428**""")
    with tab3:
        st.markdown("""
**Frozen production pair**
- 24 direct Random Forest regressors for h+1...h+24 demand forecasting.
- 24 XGBoost classifiers for h+1...h+24 Peak-Risk scoring.
- XGBoost does **not** consume the RF forecast as an input.
- Official Peak-Risk decision threshold: **0.06**.
- Public Streamlit deployment reads precomputed validated outputs and does not load the ~13 GB RF artifact bundle.
""")
