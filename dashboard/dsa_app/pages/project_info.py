from __future__ import annotations
import streamlit as st
from ..theme import page_header


def render(repo, origin, fsas, data):
    st.markdown(
        page_header(
            "About the Project",
            "Scope, architecture, deployment design and interpretation boundaries.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("""
### Purpose
This Capstone application supports review of 24-hour electricity-demand forecasts and Peak-Risk alerts for six configured Ontario FSAs. Results always reflect the FSAs selected by the user and must not be interpreted as total Ontario demand.

### Architecture
**Local / academic pipeline**  
Operational demand + weather → IFB validation → frozen RF/XGBoost inference → forecasts, Peak-Risk and scenario outputs.

**Public portfolio application**  
Validated precomputed Parquet/JSON → Streamlit UI.

### Deployment boundary
The public application uses validated precomputed outputs. This keeps the deployment lightweight while preserving interactive filtering, scenario selection, charts and tables.

### Interpretation boundaries
- Forecasts cover direct h+1 through h+24 predictions.
- Peak-Risk uses the frozen XGBoost score and official **0.06** decision rule.
- The application weather-sensitivity analysis perturbs only **forecast-origin temperature**.
- Weather-scenario results represent model sensitivity, **not causal effects**.
- Only validated/precomputed forecast origins are selectable in the public demo.
""")
    with st.expander("Operational input contract"):
        st.markdown("""
Before running the **local operational pipeline**, the following inputs must be preloaded:

- **Recent electricity demand:** at least 168 contiguous observed hours for each selected FSA.
- **Weather forecast / weather grid:** the forecast-origin and required future weather inputs used by the validated IFB contract.
- **Forecast origin/date:** selected only after both input sources pass validation.

The public deployment exposes only already-validated forecast origins.
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
