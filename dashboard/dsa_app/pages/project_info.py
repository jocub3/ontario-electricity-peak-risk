from __future__ import annotations
import streamlit as st


def render(repo, origin, fsas, data):
    st.header("Project & Deployment Information")
    st.markdown("""
### Architecture
**Local / academic pipeline**  
Operational demand + weather → IFB validation → frozen RF/XGBoost inference → forecasts, Peak-Risk and scenario outputs.

**Public portfolio application**  
Validated precomputed Parquet/JSON → Streamlit UI.

### Deployment boundary
The public application intentionally does **not** load the approximately 13 GB Random Forest model bundle. This keeps the web deployment lightweight while preserving interactive filtering, scenario selection, charts and tables.

### Interpretation boundaries
- Forecasts cover direct h+1 through h+24 predictions.
- Peak-Risk uses the frozen XGBoost score and official **0.06** decision rule.
- Model-v1 weather sensitivity perturbs only **forecast-origin temperature**.
- Weather-scenario results represent model sensitivity, **not causal effects**.
- Only validated/precomputed forecast origins are selectable in the public demo.
""")
    with st.expander("What would be required for a live local run?"):
        st.markdown("""
1. Preload at least 168 contiguous hours of recent observed demand for every requested FSA.  
2. Preload the required weather forecast/grid.  
3. Run IFB validation.  
4. Execute the frozen 24 RF + 24 XGBoost inference pipeline.  
5. Optionally generate temperature-sensitivity scenarios.  
6. Write the same application data contract consumed by this Streamlit UI.
""")
