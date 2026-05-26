# U.S. State Crime Rate Analysis
### BUAD 5722: Big Data & Cloud Analytics — William & Mary MSBA

**Team 13:** Latisha Khorana · Turner Mathieux · Violet Zhao · Bryce Marin

---

## Business Question

> **What socioeconomic and geographic factors best predict state-level crime rates in the US?**

1. Which states have the highest crime rates relative to their socioeconomic profile?
2. Does poverty, unemployment, or income best predict crime rate?
3. Where are crime rates higher or lower than expected — and why?

---

## Project Overview

Crime rates vary dramatically across US states, but the reasons are poorly understood by policymakers. High-crime states aren't always high-poverty states, and low-income communities are often underserved by both data and resources.

This project joins **FBI NIBRS 2022** incident data (11.2M records) with **US Census ACS** socioeconomic variables at the state level, then applies machine learning and LLM techniques to surface patterns, predict crime rates, and generate actionable narratives.

| Stat | Value |
|------|-------|
| Criminal incidents (2022) | 11.2M |
| Annual cost of crime to US economy | ~$1.2T |
| US population covered by NIBRS | 85% |

---

## Repository Structure

```
U.S._State_Crime_Rate_Analysis/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── state_crime_census_2022.csv       # Final merged dataset (51 states × 19 cols)
│
├── 01_data_pipeline.ipynb            # ETL: NIBRS ZIP extraction + Census API join
├── 02_crime_data_loading.ipynb       # NIBRS column mapping and chunk loading
├── 03_regression_clustering.ipynb    # H2O Random Forest + K-Means clustering
│
├── eda.py                            # Correlation analysis + scatter/heatmap plots
├── choropleth_app.py                 # Interactive Dash choropleth map
└── llm_state_analysis.py            # Groq API + LLaMA 3.3 70B state narratives
```

---

## Tech Stack

| Tool | Role |
|------|------|
| Python | Core language |
| Pandas | Data wrangling — chunk loading, aggregation |
| FBI NIBRS 2022 | Crime dataset — 11.2M incident records (ICPSR Study 38925) |
| US Census ACS API | 9 socioeconomic variables across all states |
| Google Colab | Notebook environment (GPU-enabled) |
| Google Drive | Shared team storage |
| scikit-learn | Preprocessing and evaluation |
| H2O.ai | Random Forest regression + K-Means clustering |
| Plotly / Dash | Interactive choropleth map |
| Groq API + LLaMA 3.3 70B | Automated per-state narrative generation |

---

## Pipeline

### 1. Data Pipeline (`01_data_pipeline.ipynb`)

Joins FBI NIBRS 2022 incident data with US Census ACS socioeconomic variables.

**Steps:**
1. Extract NIBRS ZIP (ICPSR Study 38925) from Google Drive
2. Load 11.2M rows in 500k-row chunks via Pandas iterator
3. Rename ICPSR V-codes to readable column names
4. Aggregate from agency level → state level
5. Pull 9 ACS variables via Census API for all 51 states
6. Merge on 2-digit state FIPS code
7. Compute `crime_rate_per_100k` as modeling target

**Output:** `state_crime_census_2022.csv` — 51 states · 19 columns

**Data Sources:**
- FBI NIBRS 2022: [ICPSR Study 38925](https://www.icpsr.umich.edu/web/NACJD/studies/38925)
- US Census ACS 5-Year 2022: [api.census.gov](https://api.census.gov)
- Join Key: 2-digit state FIPS code

---

### 2. Exploratory Data Analysis (`eda.py`)

**Key findings:**

- `clearance_rate` and `num_agencies` are both 1 for all states (duplicate columns)
- Several large states had **partial NIBRS reporting** in 2022 (NY, CA, FL, NJ, PA, AK), causing understated crime rates — flagged with orange circles on the map
- **DC removed as structural outlier** — 100% urban city-state, crime rate non-comparable to 50 states

**Cleaning impact (N=51 → N=44):**

| Metric | All States (N=51) | Clean States (N=44) | Change |
|--------|------------------|---------------------|--------|
| Crime Rate (per 100k) | 3,896.8 | 4,108.4 | +5.4% |
| Crime Skewness | 0.506 | 0.307 | -39.3% |
| Median Income | $74,805 | $73,079 | -2.3% |
| Poverty Rate | 12.4% | 12.4% | Stable |
| Avg. Population | 6.49M | 5.15M | -20.6% |

**Correlation with crime rate (Pearson & Spearman):**

| Feature | Pearson r | Spearman r | Significant? |
|---------|-----------|------------|--------------|
| White (Non-Hispanic) % | -0.219 | -0.326 | ✅ Spearman ** |
| Median Household Income | -0.143 | -0.146 | — |
| Bachelor's Degree or Higher % | -0.128 | -0.119 | — |
| Housing Vacancy Rate % | -0.106 | -0.032 | — |
| Unemployment Rate % | 0.106 | 0.053 | — |
| Total Population | 0.145 | 0.134 | — |
| Poverty Rate % | 0.180 | 0.215 | — |
| Hispanic / Latino % | 0.285 | 0.246 | ✅ Pearson * |

---

### 3. Choropleth Map (`choropleth_app.py`)

Interactive Dash app displaying 2022 US crime and socioeconomic data by state.

**Layers available:**
- Crime Rate (/100k)
- K-Means Cluster assignments
- Median Household Income
- Poverty Rate
- Unemployment Rate
- Housing Vacancy Rate

Flagged states (partial NIBRS reporting) shown with **orange circles**. DC shown with **black circle**.

**To run locally:**
```bash
pip install -r requirements.txt
python choropleth_app.py
# Open http://127.0.0.1:8050
```

---

### 4. Regression & Clustering (`03_regression_clustering.ipynb`)

**Objective:** Identify which socioeconomic factors predict crime rates, and where resources are misaligned.

#### 4a. Random Forest Regression (H2O)

**Features (X):** Unemployment rate, Bachelor's degree %, Median household income,
Poverty rate, Total population, Housing vacancy rate, White %, Black %, Hispanic/Latino %

**Target (Y):** `crime_rate_per_100k` (log-transformed)

**Pipeline:**
```
Load CSV
  → Remove DC (structural outlier)
    → Log-transform target
      → H2OFrame + 80/20 Train/Test Split
        → Random Forest (ntrees=50, max_depth=3, nfolds=5)
          → Feature Importance + Test Evaluation
```

**Model Performance:**

| Metric | Value |
|--------|-------|
| CV RMSE (log space) | 0.433 |
| CV R² | -0.097 |
| Test RMSE | 0.533 |
| **Test R²** | **0.188** |

> **Note on R²:** Negative CV R² reflects the fundamental constraint of n=50 state-level
> observations — each CV fold trains on ~40 rows. Test R² = 0.188 on the held-out set
> confirms the model retains meaningful predictive signal. Feature Importance findings
> are valid and interpretable independent of regression accuracy.

**Feature Importance Results:**

| Rank | Feature | Importance |
|------|---------|-----------|
| #1 | Total Population | 29.6% |
| #2 | Housing Vacancy Rate | 15.2% |
| #3 | Unemployment Rate | 14.1% |
| #4 | Hispanic/Latino % | 13.1% |
| #5 | Poverty Rate | 9.2% |
| #9 | White % (lowest) | 4.2% |

**Key insight:** Race composition variables rank consistently at the bottom.
Economic structure — not race — drives crime rates.

#### 4b. K-Means Clustering (H2O, k=3)

DC excluded before clustering. k=3 chosen over k=4 because k=4 produced an
unstable 2-state cluster dominated by DC.

| Cluster | States | Poverty Rate | Crime /100k | Profile |
|---------|--------|-------------|------------|---------|
| 0 — Mainstream (Red) | ~33 states | 7–13% | 2,800–5,700 | Midwest/Northeast/West baseline |
| 1 — Large-State Paradox (Blue) | CA, FL, NY, TX, PA | 12–14% | 700–2,000 ↓ | High population + urban resources suppress crime |
| 2 — High-Risk Southern (Green) ⚠️ | NM, AR, TN, AL, LA, MS… | 14–19% | 2,600–6,000 ↑ | High poverty + high crime; most severe misalignment |

---

### 5. LLM Narrative Generation (`llm_state_analysis.py`)

Automatically generates plain-English crime analysis summaries for all 50 states
using **LLaMA 3.3 70B** via Groq API.

**How it works:**
1. Load `state_crime_census_2022.csv` — one row per state
2. Format each row as a structured JSON prompt
3. Call Groq API → LLaMA returns a full written analysis per state
4. Export to Markdown report with national stats table + per-state summaries

**Output:** 51 state analyses · 32 pages · generated in under 2 minutes

**Setup:**
```bash
# Get a free API key at https://console.groq.com
export GROQ_API_KEY="your_key_here"
python llm_state_analysis.py
```

---

## Key Findings

1. **Economic structure, not race, predicts crime.** Race composition variables ranked last in feature importance across all model versions.

2. **Total population is the strongest predictor (29.6%).** Large states (CA, FL, NY, TX) show systematically lower crime rates — confirmed by both regression and clustering ("large-state paradox").

3. **Housing vacancy (15.2%) and unemployment (14.1%)** are the next strongest predictors, acting as proxies for community decline and economic pressure.

4. **Southern states (Cluster 2) show the most severe misalignment** — high poverty + high crime across NM, AR, TN, AL, LA, MS and others.

5. **DC is a structural outlier** — 100% urban density makes its crime rate (8,906/100k) inherently non-comparable to states and was excluded from modeling.

---

## Limitations

- State-level aggregation (n=50) limits regression accuracy; county-level data would significantly improve model performance
- NIBRS participation was partial for several large states in 2022 (NY, CA, FL, NJ, PA, AK), which may affect crime rate comparability
- Feature Importance reflects association within this dataset, not causal relationships

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/violet730323/U.S._State_Crime_Rate_Analysis.git
cd U.S._State_Crime_Rate_Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys as environment variables (never hardcode)
export GROQ_API_KEY="your_groq_key"
export CENSUS_API_KEY="your_census_key"

# 4. Run notebooks in order (Google Colab recommended)
# 01_data_pipeline.ipynb → 02_crime_data_loading.ipynb → 03_regression_clustering.ipynb

# 5. Run scripts
python eda.py                 # Generate correlation charts
python choropleth_app.py      # Launch interactive map at localhost:8050
python llm_state_analysis.py  # Generate per-state LLM report
```

**Environment:** Google Colab (recommended) or local Python 3.9+

---

## Data

The processed dataset `state_crime_census_2022.csv` is included in this repository.

Raw FBI NIBRS data (11.2M rows, ~737MB) is available from:
[ICPSR Study 38925](https://www.icpsr.umich.edu/web/NACJD/studies/38925)

---

*BUAD 5722 Big Data & Cloud Analytics · William & Mary MSBA · Spring 2026*
