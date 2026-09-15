# Ontario Electricity Demand Forecasting and Peak-Risk Decision Support System

Capstone project for **DAMO 699 – Capstone Project**, Master of Data Analytics, University of Niagara Falls Canada.

**Live Dashboard:** [Open Streamlit Application](https://ontario-electricity-peak-risk-decision-support-application-on.streamlit.app/)


## Project team

- Juan Avila
- Jorge Cuenca
- Gonzalo Garcia
- Tessy Ramirez

**Faculty advisor:** Dr. Bilal El Toufaili

---

## Project overview

This project develops a reproducible analytics and decision-support system for short-term electricity demand forecasting and peak-demand risk assessment across six Forward Sortation Areas (FSAs) in the Greater Toronto Area.

The system:

1. Forecasts hourly electricity demand for the next 24 hours, for the 6 FSAs.
2. Estimates the probability that each forecasted hour will be classified as a peak-demand risk for each forecast horizon.
3. Evaluates the influence of weather conditions through temperature-sensitivity scenarios.
4. Provides model interpretation and operational decision-support information.
4. Presents forecast, peak-risk, weather-sensitivity, and model-insight results through an interactive Streamlit dashboard.

The final solution uses a shared analytical pipeline applied to two representative regions of the Greater Toronto Area, each defined by its nearest ECCC weather station:

- **Downtown** — FSA `M5S`, `M5R`, `M6G`, weather from the Toronto City station.
- **Airport-West** — FSA `L4T`, `M9W`, `M9R`, weather from the Toronto Pearson (INTL A) station.

Weather information is obtained from two Environment and Climate Change Canada (ECCC) stations representing the Toronto City and Toronto Pearson areas.

The analytical methodology is designed as a common pipeline that can be extended to additional FSAs when equivalent electricity-demand and weather data are available.

## Live Demo

The interactive decision-support dashboard is publicly available through Streamlit:

**[Open the Ontario Electricity Peak-Risk Decision Support Application](https://ontario-electricity-peak-risk-decision-support-application-on.streamlit.app/)**

The application provides access to:

- 24-hour electricity-demand forecasts;
- peak-risk probabilities and alert indicators;
- FSA-level comparisons;
- weather-sensitivity scenarios;
- model insights and methodology information.

The public deployment uses validated precomputed results stored in `data/app/`, allowing users to explore the decision-support outputs without retraining the forecasting and classification models.

---

## Main data sources

The project combines three main sources of information:

- **Electricity demand:** hourly electricity consumption by Forward Sortation Area (FSA). Data are obtained from Independent Electricity System Operator (IESO).
- **Weather:** hourly weather observations from the Toronto City and Toronto Pearson (INTL A) stations. Observations gathering from Environment and Climate Change Canada (ECCC).
- **Calendar information:** temporal variables including hour, weekday, weekend, month, season, quarter, holidays, and daylight-saving-time information.

Additional engineered variables include lagged demand, rolling statistics, weather-derived features, interactions, and other predictors required by the forecasting and peak-risk models.

See:

- `data/README.md` for the data-folder structure and data usage.
- `docs/data_dictionary.md` for variable definitions.
- `docs/` for phase-specific methodological and implementation documentation.

Raw source datasets are not stored in the GitHub public repository.

---

## Repository Structure

The repository is organized by analytical phase and application responsibility:

```text
ontario-electricity-peak-risk/
│
├── .github/                    # Pull-request and GitHub configuration
├── artifacts/                  # Final model metadata and supporting artifacts
├── configs/                    # YAML configuration files by analytical phase
│
├── dashboard/                  # Streamlit decision-support application
│   ├── app.py
│   └── dsa_app/                # Dashboard components, views, state and pages
│
├── data/
│   ├── app/                    # Validated data required by the dashboard
│   ├── external/               # External-data placeholder
│   ├── interim/                # Intermediate local data
│   ├── operational/            # Operational inputs and templates
│   ├── processed/              # Processed analytical datasets
│   ├── raw/                    # Raw-data placeholder
│   └── sample/                 # Sample-data placeholder
│
├── docs/                       # Technical and phase-specific documentation
│
├── notebooks/
│   ├── 01–08                   # Core data, EDA, features, targets and foundation
│   ├── modeling/               # Forecasting, peak-risk and model comparison
│   ├── final_holdout_2025/     # Final holdout evaluation
│   ├── final_training/         # Final model training and validation
│   ├── explainability/         # Model explainability
│   ├── weather_sensitivity/    # Weather-scenario analysis
│   ├── inference_feature_builder/
│   ├── integrated_pipeline/
│   └── decision_support/       # Decision-support application development
│
├── reports/                    # Analytical reports and summarized results
├── scripts/                    # Dashboard launch and validation utilities
│
├── src/
│   └── ontario_peak_risk/      # Reusable Python implementation by phase
│
├── tests/                      # Automated validation tests
│
├── main.py                     # Project command-line entry point
├── requirements.txt            # Project Python dependencies
└── README.md
```

Generated outputs, model checkpoints, Python cache files, and serialized `.joblib` model files are excluded from version control through `.gitignore`.

---

## Analytical Workflow

The project follows a notebook-driven analytical workflow.

The main phases are:

1. **Data Dictionary**
2. **Data Quality**
3. **Data Preparation**
4. **Data Cleaning**
5. **Exploratory Data Analysis**
6. **Feature Engineering**
7. **Target Definition**
8. **Modeling Foundation**
9. **Forecasting Model Development**
10. **Peak-Risk Model Development**
11. **Model Comparison and Final Refinement**
12. **Final Holdout Evaluation**
13. **Final Model Training**
14. **Explainability**
15. **Weather Sensitivity Analysis**
16. **Inference Feature Construction**
17. **Integrated Prediction Pipeline**
18. **Decision-Support Application**

The analytical pipeline is reproduced by executing the notebooks **step by step in their defined numerical order**, starting with the root notebooks (`01`, `02`, `03`, etc.) and then continuing through the specialized phase directories.

Within each specialized phase, notebooks should also be executed in numerical order.

Phase documentation under `docs/` provides additional information about methodology, assumptions, outputs, and validation.

> **Important:** `main.py` does not automatically execute the complete notebook-based analytical workflow. It provides the project entry point, initializes the project configuration, displays workflow guidance, and can launch the decision-support dashboard.

---

## Modeling Approach

### Electricity Demand Forecasting

Multiple forecasting approaches were developed and evaluated, including:

- Seasonal Naive baseline
- SARIMAX
- Random Forest Regressor
- LightGBM Regressor
- XGBoost Regressor

The forecasting workflow generates predictions for the next **24 hourly horizons**.

Following model comparison and refinement, **Random Forest** is used as the final forecasting model for the operational decision-support workflow.

### Peak-Risk Classification

Multiple classification approaches were evaluated, including:

- Logistic Regression
- HistGradientBoosting
- Random Forest Classifier
- LightGBM Classifier
- XGBoost Classifier

Peak-risk classification is performed across the same 24-hour forecast horizon.

Following model comparison, threshold analysis, and final refinement, **XGBoost** is used as the final peak-risk classifier.

### Model Evaluation

The project includes:

- time-aware train/validation/test separation;
- common evaluation metrics;
- model stability analysis;
- model comparison;
- operational threshold analysis;
- final holdout evaluation;
- final model training and artifact validation.

### Explainability and Weather Sensitivity

The project also includes:

- global feature importance;
- SHAP-based interpretation;
- local prediction explanations;
- feature-family analysis;
- horizon and FSA comparisons;
- weather-influence analysis;
- temperature-sensitivity scenarios;
- operational interpretation of forecast and peak-risk results.

---

## Decision-Support Dashboard

The final analytical outputs are presented through an interactive **Streamlit decision-support application**.

A public deployment is available through the [Live Demo](#live-demo).

The dashboard provides views for:

- executive overview;
- 24-hour electricity-demand forecasts;
- peak-risk probabilities and alerts;
- FSA-level comparisons;
- model insights;
- weather-sensitivity scenarios;
- project and methodological information.

The public dashboard uses validated precomputed application data stored under:

```text
data/app/
```

This allows the deployed application to present the validated analytical results without requiring model retraining during each dashboard session.

---

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/jocub3/ontario-electricity-peak-risk.git
cd ontario-electricity-peak-risk
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For dashboard-specific dependencies, if required:

```bash
pip install -r dashboard/requirements_dashboard.txt
```

### 4. Verify the project

```bash
python main.py --help
pytest
```

---

## Running the Project

The analytical development workflow and the interactive dashboard are intentionally separated.

The notebooks reproduce the analytical pipeline, while the dashboard consumes validated application-ready results.

### Option 1 — Run from `main.py` (Recommended)

From the repository root:

```bash
python main.py
```

This initializes the project configuration and displays guidance for reproducing the notebook-based analytical workflow.

To launch the dashboard through the project entry point:

```bash
python main.py --dashboard
```

This starts the Streamlit decision-support application using:

```text
dashboard/app.py
```

This is the recommended way to launch the application from Python because `main.py` acts as the central project entry point.

---

### Option 2 — Run from Windows Command Prompt

A Windows batch launcher is provided under:

```text
scripts/run_dashboard_windows.bat
```

From the repository root, run:

```cmd
scripts\run_dashboard_windows.bat
```

The script automatically changes to the project root and launches the Streamlit application.

A PowerShell launcher is also available:

```powershell
.\scripts\run_dashboard_powershell.ps1
```

---

### Option 3 — Run the Dashboard Directly

The Streamlit application can also be started without using `main.py` or the helper scripts.

From the repository root:

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will start the local application and provide the local browser address in the terminal.

---

## Reproducing the Analytical Pipeline

The complete analytical workflow is notebook-driven and should be executed sequentially.

Start with the root notebooks:

```text
01_data_dictionary.ipynb
02_data_quality.ipynb
03_data_preparation.ipynb
04_data_cleaning.ipynb
05_*  Exploratory Data Analysis
06_*  Feature Engineering
07_*  Target Definition
08_*  Modeling Foundation
```

Then continue through the specialized directories for:

```text
modeling/
final_holdout_2025/
final_training/
explainability/
weather_sensitivity/
inference_feature_builder/
integrated_pipeline/
decision_support/
```

Within each phase, follow the numerical prefixes of the notebooks.

Configuration files for the corresponding analytical phases are available under:

```text
configs/
```

Reusable Python implementations are maintained under:

```text
src/ontario_peak_risk/
```

Detailed methodological documentation and phase summaries are available under:

```text
docs/
```

---

## Data and Version-Control Policy

The repository separates source code and reproducible project assets from large or generated runtime artifacts.

### Versioned

The repository includes:

- source code;
- notebooks;
- configuration files;
- technical documentation;
- processed project datasets under `data/processed/`;
- validated dashboard data under `data/app/`;
- dashboard source code;
- analytical reports selected for project documentation.

### Not Versioned

The following are excluded through `.gitignore`:

- raw datasets under `data/raw/`;
- intermediate datasets under `data/interim/`;
- model-training checkpoints;
- serialized `.joblib` models;
- Python cache files;
- Jupyter checkpoints;
- logs and temporary files;
- local environment variables and secrets.


---


## Reproducibility rules


To maintain reproducibility and consistency:

- use relative project paths;
- keep configuration parameters under `configs/`;
- maintain reusable implementation code under `src/`;
- keep analytical workflow and validation notebooks under `notebooks/`;
- maintain processed analytical datasets under `data/processed/`;
- keep dashboard-ready data under `data/app/`;
- document analytical decisions and phase results under `docs/`;
- use time-aware validation for forecasting and peak-risk modeling;
- preserve model parameters, thresholds, metadata, and methodological documentation required to interpret final results;
- use Pull Requests for reviewed repository changes.

---

## Documentation

Detailed documentation is available under `docs/`, including:

- data dictionary and data-quality documentation;
- data preparation and cleaning;
- exploratory data analysis;
- feature engineering;
- target definition;
- modeling foundation;
- forecasting and peak-risk modeling;
- model comparison and selection;
- final holdout evaluation;
- final training;
- explainability;
- weather-sensitivity analysis;
- inference feature construction;
- integrated prediction pipeline;
- decision-support application architecture.

Phase-specific documentation should be consulted together with the corresponding notebooks and configuration files.

