# Ontario Electricity Demand Forecasting and Peak-Risk Decision Support System

Capstone project for **DAMO 699 – Capstone Project**, Master of Data Analytics, University of Niagara Falls Canada.

## Project team

- Juan Avila
- Jorge Cuenca
- Gonzalo Garcia
- Tessy Ramirez

**Faculty advisor:** Dr. Bilal El Toufaili

## Project overview

This project develops a reproducible decision-support analytics prototype that:

1. Forecasts Ontario electricity demand for each of the next 24 hours.
2. Estimates the probability that each forecasted hour will be classified as a peak-demand risk.
3. Evaluates forecast sensitivity under alternative temperature scenarios.
4. Presents the results through an interactive dashboard.

The final solution will use a shared analytical pipeline for selected representative areas and will be designed so that the same methodology can later be applied to additional regions.

## Main data sources

- Independent Electricity System Operator (IESO): hourly electricity consumption.
- Environment and Climate Change Canada (ECCC): hourly weather observations.
- Calendar variables: hour, weekday, weekend, season, and statutory holidays.

Raw datasets are not stored in GitHub. See `data/README.md` for the expected local folder structure.

## Repository structure

```text
ontario-electricity-peak-risk/
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── configs/
├── dashboard/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── external/
│   └── sample/
├── docs/
├── models/
├── notebooks/
│   ├── 01_data_exploration/
│   ├── 02_forecasting/
│   ├── 03_peak_risk/
│   └── 04_evaluation/
├── outputs/
├── reports/
│   └── figures/
├── src/
│   └── ontario_peak_risk/
│       ├── data/
│       ├── features/
│       ├── forecasting/
│       ├── peak_risk/
│       ├── evaluation/
│       ├── scenarios/
│       ├── visualization/
│       └── utils/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```


## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/jocub3/ontario-electricity-peak-risk.git
cd ontario-electricity-peak-risk
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the project

```bash
python main.py --help
pytest
```