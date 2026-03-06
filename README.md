<div align="center">

<br/>

```
████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗
╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝
   ██║   ██║██╔████╔██║█████╗      ███████╗█████╗  ██████╔╝██║█████╗  ███████╗
   ██║   ██║██║╚██╔╝██║██╔══╝      ╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║
   ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║
   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝

          F O R E C A S T I N G   &   I N V E N T O R Y   I N T E L L I G E N C E
```

<br/>

**Prophet-powered multi-store demand forecasting with a full-stack architecture —**
**interactive Streamlit dashboard · REST API · automated training pipeline**

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-1a1a2e?style=flat-square&logo=python&logoColor=c9a84c)
![Prophet](https://img.shields.io/badge/Prophet-1.1-1a1a2e?style=flat-square&logo=meta&logoColor=c9a84c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-1a1a2e?style=flat-square&logo=streamlit&logoColor=c9a84c)
![FastAPI](https://img.shields.io/badge/FastAPI-0.x-1a1a2e?style=flat-square&logo=fastapi&logoColor=c9a84c)
![License](https://img.shields.io/badge/License-MIT-1a1a2e?style=flat-square&logoColor=c9a84c)

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Workflow](#2-architecture--workflow)
3. [Repository Structure](#3-repository-structure)
4. [Dataset](#4-dataset)
5. [Feature Engineering](#5-feature-engineering)
6. [Modelling — Facebook Prophet](#6-modelling--facebook-prophet)
7. [Training Pipeline](#7-training-pipeline)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [REST API — FastAPI](#9-rest-api--fastapi)
10. [Evaluation](#10-evaluation)
11. [Installation & Setup](#11-installation--setup)
12. [Usage Guide](#12-usage-guide)
13. [Model Inventory](#13-model-inventory)
14. [Key Results & Insights](#14-key-results--insights)
15. [Tech Stack](#15-tech-stack)

---

## 1. Project Overview

This project delivers an **end-to-end time-series demand forecasting system** for a multi-store, multi-product retail warehouse environment. It trains an individual **Facebook Prophet** model per store–product pair (100 models total), persists them as serialised artefacts, and exposes predictions through two consumer interfaces: a polished **Streamlit web dashboard** and a production-grade **FastAPI REST service**.

### Core capabilities

| Capability | Description |
|---|---|
| Per-pair forecasting | One dedicated Prophet model per store × product combination |
| Horizon flexibility | 7-day, 30-day, or arbitrary date lookups up to 4 years ahead |
| Trend intelligence | Automatic rise / drop / stable classification over any horizon |
| Inventory planning | Computes recommended stock levels with peak and daily averages |
| Portfolio scanning | Batch trend analysis across all 100 store-product pairs |
| Comparative analysis | Side-by-side forecast comparison for two products in the same store |
| Restock alerts | Threshold-based alerting for under-performing inventory pairs |
| REST API | Full FastAPI service mirroring all dashboard capabilities |

---

## 2. Architecture & Workflow

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    SYSTEM ARCHITECTURE — DEMAND FORECASTING                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  LAYER 0 — DATA INGESTION                                                   │
  │                                                                             │
  │   retail_warehouse_inventory_dataset.csv                                    │
  │         │                                                                   │
  │         ▼                                                                   │
  │   ┌─────────────┐    parse_dates=["Date"]    ┌──────────────────────────┐  │
  │   │ data_loader │ ─────────────────────────► │  pandas DataFrame        │  │
  │   │  .py        │                            │  73,100 rows × 22 cols   │  │
  │   └─────────────┘                            └──────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  LAYER 1 — PREPROCESSING & FEATURE ENGINEERING                             │
  │                                                                             │
  │   preprocessing.py :: prepare_time_series(df, store_id, product_id)        │
  │                                                                             │
  │   ┌──────────────────────────────────────────────────────────────────────┐ │
  │   │  Filter by Store ID + Product ID                                     │ │
  │   │         │                                                            │ │
  │   │         ▼                                                            │ │
  │   │  Group by Date → aggregate Units Sold (sum)                         │ │
  │   │         │                                                            │ │
  │   │         ▼                                                            │ │
  │   │  Resample to daily frequency ('D')                                   │ │
  │   │         │                                                            │ │
  │   │         ▼                                                            │ │
  │   │  Fill missing dates with 0 (asfreq)                                 │ │
  │   │         │                                                            │ │
  │   │         ▼                                                            │ │
  │   │  Rename: Date → ds,  Units Sold → y   (Prophet format)              │ │
  │   └──────────────────────────────────────────────────────────────────────┘ │
  │                                                                             │
  │   Notebook EDA also adds:  day · month · year · day_of_week · is_weekend   │
  │                            week_of_year · quarter                           │
  └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  LAYER 2 — MODEL TRAINING                                                   │
  │                                                                             │
  │   main.py  ──►  train.py :: train_models()                                 │
  │                                                                             │
  │   for each (Store, Product) pair — 5 × 20 = 100 combos                    │
  │                                                                             │
  │   ┌────────────────────────────────────────────────────────────────────┐   │
  │   │                                                                    │   │
  │   │   prepare_time_series()                                            │   │
  │   │          │                                                         │   │
  │   │          ▼                                                         │   │
  │   │   forecasting.py :: fit_prophet(ts)                                │   │
  │   │          │                                                         │   │
  │   │          │   Prophet(                                              │   │
  │   │          │     yearly_seasonality  = True,                        │   │
  │   │          │     weekly_seasonality  = True,                        │   │
  │   │          │     daily_seasonality   = False,                       │   │
  │   │          │   ).fit(ts)                                             │   │
  │   │          │                                                         │   │
  │   │          ▼                                                         │   │
  │   │   joblib.dump → "models/prophet_models/{S}_{P}_prophet.joblib"    │   │
  │   │                                                                    │   │
  │   └────────────────────────────────────────────────────────────────────┘   │
  │                                                                             │
  │   Output: 100 .joblib model files  (S001–S005 × P0001–P0020)               │
  └─────────────────────────────────────────────────────────────────────────────┘
                         │                              │
                         ▼                              ▼
  ┌────────────────────────────┐        ┌───────────────────────────────────────┐
  │  LAYER 3A — EVALUATION     │        │  LAYER 3B — INFERENCE SERVING         │
  │                            │        │                                       │
  │  evaluate.py               │        │  ┌──────────────────────────────────┐ │
  │                            │        │  │  app.py  — Streamlit Dashboard   │ │
  │  • Load each .joblib       │        │  │                                  │ │
  │  • make_future_dataframe   │        │  │  Tab 1  Demand Forecast          │ │
  │    (periods=30)            │        │  │  Tab 2  Trend Detection          │ │
  │  • model.predict(future)   │        │  │  Tab 3  Inventory Planning       │ │
  │  • model.plot(forecast)    │        │  │  Tab 4  Date Forecast            │ │
  │    → matplotlib figure     │        │  │  Tab 5  Portfolio Analysis       │ │
  │                            │        │  │  Tab 6  Product Comparison       │ │
  └────────────────────────────┘        │  │  Tab 7  Restock Alerts           │ │
                                        │  └──────────────────────────────────┘ │
                                        │                                       │
                                        │  ┌──────────────────────────────────┐ │
                                        │  │  server.py — FastAPI REST API    │ │
                                        │  │                                  │ │
                                        │  │  POST /predict-units/            │ │
                                        │  │  POST /detect-trend/             │ │
                                        │  │  POST /recommend-stock/          │ │
                                        │  │  POST /forecast-date/            │ │
                                        │  │  GET  /trend-analysis/           │ │
                                        │  └──────────────────────────────────┘ │
                                        └───────────────────────────────────────┘

  ─────────────────────────────────────────────────────────────────────────────
  INFERENCE FLOW (per request)

   User selects   ──►  load_model(store, product)       joblib.load(.joblib)
   Store + Product
        │
        ▼
   make_future_dataframe(periods=N)  ──►  model.predict(future)
        │                                        │
        │          ┌──────────────────────────────┤
        │          │  forecast DataFrame          │
        │          │  ds · yhat · yhat_lower ·    │
        │          │  yhat_upper · trend ·        │
        │          │  weekly · yearly             │
        │          └──────────────────────────────┘
        ▼
   Post-processing:
     • chart_df      → st.line_chart()
     • total / avg   → metric boxes
     • diff.mean()   → trend classification (rise / drop / stable)
     • yhat.sum()    → recommended inventory
  ─────────────────────────────────────────────────────────────────────────────
```

---

## 3. Repository Structure

```
Time-Series-Forecasting/
│
├── app.py                          Streamlit dashboard (7-tab UI)
├── server.py                       FastAPI REST service
├── main.py                         Training + evaluation entry point
├── requirements.txt                Python dependencies
│
├── data/
│   └── retail_warehouse_inventory_dataset.csv   73,100-row retail dataset
│
├── models/
│   └── prophet_models/
│       ├── S001_P0001_prophet.joblib
│       ├── S001_P0002_prophet.joblib
│       ├── ...                     (100 .joblib files — S001–S005 × P0001–P0020)
│       └── S005_P0020_prophet.joblib
│
├── notebooks/
│   └── EDA_and_Feature_Engineering.ipynb    Exploratory analysis notebook
│
└── src/
    ├── __init__.py
    ├── data_loader.py              load_data() — CSV → DataFrame
    ├── preprocessing.py            prepare_time_series() — filter/resample/format
    ├── forecasting.py              fit_prophet() — Prophet model constructor
    ├── train.py                    train_models() — batch training loop
    ├── evaluate.py                 evaluate_models() — forecast + plot loop
    └── utils.py                    set_seed() — reproducibility helper
```

---

## 4. Dataset

### Source file
`data/retail_warehouse_inventory_dataset.csv`

### Dimensions

| Attribute | Value |
|---|---|
| Total rows | 73,100 |
| Total columns | 22 |
| Stores | 5 (`S001` – `S005`) |
| Products | 20 (`P0001` – `P0020`) |
| Date range | 2022-01-01 → 2024-01-01 (2 years) |
| Granularity | Daily per store-product pair |
| Missing values | None (0 nulls across all columns) |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `Date` | datetime | Transaction date |
| `Store ID` | categorical | Store identifier (S001–S005) |
| `Product ID` | categorical | Product identifier (P0001–P0020) |
| `Category` | categorical | Product category |
| `Region` | categorical | Geographic region |
| `Inventory Level` | int | Current stock on hand |
| `Units Sold` | int | **Target variable** — daily units sold |
| `Units Ordered` | int | Replenishment order quantity |
| `Demand Forecast` | float | Pre-existing baseline forecast |
| `Price` | float | Unit selling price (USD) |
| `Discount` | int | Discount percentage applied |
| `Weather Condition` | categorical | Ambient weather on that day |
| `Holiday/Promotion` | int | Binary flag — 1 if holiday or promotion |
| `Competitor Pricing` | float | Competing store price (USD) |
| `Seasonality` | categorical | Seasonal label |
| `day` | int | Day of month (1–31) |
| `month` | int | Month number (1–12) |
| `year` | int | Year (2022–2024) |
| `day_of_week` | int | Weekday index (0=Monday, 6=Sunday) |
| `is_weekend` | int | Binary flag — 1 if Saturday or Sunday |
| `week_of_year` | int | ISO week number (1–52) |
| `quarter` | int | Calendar quarter (1–4) |

### Categorical Value Reference

```
┌─────────────────────┬──────────────────────────────────────────────────────┐
│  Column             │  Values                                              │
├─────────────────────┼──────────────────────────────────────────────────────┤
│  Category           │  Groceries · Toys · Electronics · Furniture ·        │
│                     │  Clothing                                             │
├─────────────────────┼──────────────────────────────────────────────────────┤
│  Region             │  North · South · East · West                         │
├─────────────────────┼──────────────────────────────────────────────────────┤
│  Weather Condition  │  Sunny · Rainy · Cloudy · Snowy                      │
├─────────────────────┼──────────────────────────────────────────────────────┤
│  Seasonality        │  Spring · Summer · Autumn · Winter                   │
└─────────────────────┴──────────────────────────────────────────────────────┘
```

### Descriptive Statistics

| Column | Min | Mean | Median | Max | Std Dev |
|---|---|---|---|---|---|
| Inventory Level | 50 | 274.5 | 273 | 500 | 129.9 |
| **Units Sold** | 0 | **136.5** | 107 | 499 | 108.9 |
| Units Ordered | 20 | 110.0 | 110 | 200 | 52.3 |
| Demand Forecast | -9.99 | 141.5 | 113 | 518.6 | 109.3 |
| Price (USD) | 10.00 | 55.14 | 55.05 | 100.00 | 26.0 |
| Discount (%) | 0 | 10.0 | 10 | 20 | 7.1 |
| Competitor Pricing (USD) | 5.03 | 55.15 | 55.01 | 104.94 | 26.2 |

### Dataset chart — Units Sold distribution

```
  Units Sold — approximate frequency distribution   (73,100 records)

  0  ┤
     │███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 50  ┤
     │████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
100  ┤
     │██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← mean ~137
150  ┤
     │████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
200  ┤
     │██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
250  ┤
     │███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
300  ┤
     │████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
350+ ┤
     │████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     └──────────────────────────────────────────────────
      Right-skewed distribution — high variance across pairs
```

### Dataset chart — Coverage by store and product

```
  Store × Product Grid  (each cell = daily observations over 2 years)

          P001  P002  P003  P004  P005  P006  P007  P008  P009  P010  ··  P020
  S001  [████][████][████][████][████][████][████][████][████][████]··[████]
  S002  [████][████][████][████][████][████][████][████][████][████]··[████]
  S003  [████][████][████][████][████][████][████][████][████][████]··[████]
  S004  [████][████][████][████][████][████][████][████][████][████]··[████]
  S005  [████][████][████][████][████][████][████][████][████][████]··[████]

  ████ = complete 2-year daily series (730 rows per cell), no gaps, no nulls
  Total cells: 100   Total rows: 73,100   Completeness: 100%
```

---

## 5. Feature Engineering

Feature engineering is performed in two places: the **preprocessing module** (runtime, for Prophet input) and the **EDA notebook** (analytical, columnar features appended to the raw dataset).

### Runtime preprocessing — `src/preprocessing.py`

```
Raw DataFrame
      │
      ├─ Filter rows where Store ID == store_id AND Product ID == product_id
      │
      ├─ groupby('Date')['Units Sold'].sum()
      │         Aggregates multiple rows on the same date into a single daily figure
      │
      ├─ .resample('D').sum()
      │         Ensures daily frequency; sums any remaining duplicates
      │
      ├─ .reset_index()
      │
      ├─ .set_index('ds').asfreq('D', fill_value=0).reset_index()
      │         Inserts any missing calendar days as 0-demand entries
      │         Guarantees a continuous, gap-free time series for Prophet
      │
      └─ Output: DataFrame with columns [ds, y]
                 ds = date (datetime64)
                 y  = units sold (int)
```

### Notebook analytical features — `EDA_and_Feature_Engineering.ipynb`

| Feature | Derivation | Value range | Purpose |
|---|---|---|---|
| `day` | `Date.dt.day` | 1–31 | Intra-month demand position |
| `month` | `Date.dt.month` | 1–12 | Monthly seasonality |
| `year` | `Date.dt.year` | 2022–2024 | Inter-year trend direction |
| `day_of_week` | `Date.dt.dayofweek` | 0–6 | Mon–Sun demand cycles |
| `is_weekend` | `day_of_week >= 5` | 0 or 1 | Weekend sales uplift flag |
| `week_of_year` | `Date.dt.isocalendar().week` | 1–52 | 52-week seasonal pattern |
| `quarter` | `Date.dt.quarter` | 1–4 | Quarterly business cycles |

### Seasonality map

```
  Quarter → Season mapping observed in dataset

  Q1  (Jan–Mar)  →  Winter  ·  Spring
  Q2  (Apr–Jun)  →  Spring  ·  Summer
  Q3  (Jul–Sep)  →  Summer  ·  Autumn
  Q4  (Oct–Dec)  →  Autumn  ·  Winter

  Weather distribution:  Sunny · Rainy · Cloudy · Snowy
  Promotion events:      ~50% of days flagged as Holiday/Promotion = 1
```

---

## 6. Modelling — Facebook Prophet

### Why Prophet?

Facebook Prophet is particularly well-suited to this problem because it:

- Handles **multiple seasonalities** natively (weekly + yearly) without manual lag engineering
- Is **robust to missing data and outliers** — critical for retail time series
- Provides **uncertainty intervals** (`yhat_lower`, `yhat_upper`) out of the box
- Scales cleanly to a **batch training loop** across 100 store-product pairs
- Produces interpretable **trend + seasonality decomposition** plots
- The additive model structure prevents negative-forecast artefacts seen in the raw dataset's baseline Demand Forecast column

### Model configuration — `src/forecasting.py`

```python
Prophet(
    yearly_seasonality = True,   # captures annual demand patterns (holiday spikes etc.)
    weekly_seasonality = True,   # captures Mon–Sun demand cycles
    daily_seasonality  = False,  # disabled — no sub-day signal at daily grain
)
```

### Prophet decomposition

```
  Observed signal  =  Trend  +  Weekly seasonality  +  Yearly seasonality  +  Noise

  ┌────────────┐     ┌─────────────────────────────────────────────────────────┐
  │  Trend     │     │  Long-run direction (growth / decline) over 2 years     │
  │  Component │     │  Captures gradual demand drift per store-product pair   │
  └────────────┘     └─────────────────────────────────────────────────────────┘

  ┌────────────┐     ┌─────────────────────────────────────────────────────────┐
  │  Weekly    │     │  Mon  Tue  Wed  Thu  Fri  Sat  Sun                      │
  │  Seasonal  │     │  ▂▂▂  ▃▃▃  ▅▅▅  ▄▄▄  ███  ██   ▂▂▂   (illustrative)   │
  └────────────┘     └─────────────────────────────────────────────────────────┘

  ┌────────────┐     ┌─────────────────────────────────────────────────────────┐
  │  Yearly    │     │  Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec │
  │  Seasonal  │     │  ▅▅   ▄▄   ▃▃   ▃▃   ▄▄   ▅▅   ██   ███  ██   ▅▅   ▄▄   ▅▅ │
  └────────────┘     └─────────────────────────────────────────────────────────┘

  ┌────────────┐     ┌─────────────────────────────────────────────────────────┐
  │  Forecast  │     │  yhat         — point estimate                          │
  │  Output    │     │  yhat_lower   — 80% confidence lower bound              │
  │            │     │  yhat_upper   — 80% confidence upper bound              │
  └────────────┘     └─────────────────────────────────────────────────────────┘
```

### Prophet output columns used in this project

| Column | Description | Where used |
|---|---|---|
| `ds` | Forecast date | Chart x-axis index |
| `yhat` | Point forecast | All predictions, metrics, line charts |
| `yhat_lower` | Lower confidence bound | Date Forecast tab — pessimistic bound |
| `yhat_upper` | Upper confidence bound | Date Forecast tab — optimistic bound |
| `trend` | Long-run trend component | Internal decomposition |
| `weekly` | Weekly seasonality component | Internal decomposition |
| `yearly` | Yearly seasonality component | Internal decomposition |

### Model storage

```
models/prophet_models/
├── {STORE_ID}_{PRODUCT_ID}_prophet.joblib
│
├── S001_P0001_prophet.joblib
├── S001_P0002_prophet.joblib
│   ...
├── S005_P0019_prophet.joblib
└── S005_P0020_prophet.joblib
```

100 serialised models, loaded on-demand via `joblib.load()`. No model is held in memory until explicitly requested.

---

## 7. Training Pipeline

### Entry point

```bash
python main.py
```

### Flow diagram

```
  main.py
    │
    ├─ os.makedirs("models/prophet_models", exist_ok=True)
    │
    ├──────────────────────────────────────────────────────────────────────────
    │  PHASE 1 — TRAINING   (src/train.py :: train_models)
    │
    │   load_data("data/retail_warehouse_inventory_dataset.csv")
    │         │
    │         ▼
    │   df[['Store ID','Product ID']].drop_duplicates()
    │         │
    │         ▼
    │   ┌───────────────────────────────────────────────────────────────┐
    │   │  for (store, product) in 100 unique pairs:                    │
    │   │                                                               │
    │   │    ts = prepare_time_series(df, store, product)               │
    │   │         → [ds, y] with daily frequency, no gaps               │
    │   │                                                               │
    │   │    model = fit_prophet(ts)                                    │
    │   │         → Prophet(yearly=T, weekly=T, daily=F).fit(ts)        │
    │   │                                                               │
    │   │    joblib.dump(model, f"models/.../{store}_{product}_prophet.joblib")
    │   │                                                               │
    │   │    print(f"Trained Prophet for {store}-{product}")            │
    │   └───────────────────────────────────────────────────────────────┘
    │
    ├──────────────────────────────────────────────────────────────────────────
    │  PHASE 2 — EVALUATION   (src/evaluate.py :: evaluate_models)
    │
    │   for fname in os.listdir("models/prophet_models"):
    │         │
    │         ▼
    │     joblib.load(model)
    │     make_future_dataframe(periods=30)
    │     model.predict(future)
    │     model.plot(forecast)   → matplotlib figure per pair
    │
    └──────────────────────────────────────────────────────────────────────────

  Output: 100 trained .joblib files  +  matplotlib evaluation charts
```

---

## 8. Streamlit Dashboard

### Launch

```bash
python -m streamlit run app.py
```

Navigate to `http://localhost:8501`

The dashboard is organised into **7 navigation tabs**:

---

### Tab 1 — Demand Forecast

**Business question:** How many units will be sold over the next 7 or 30 days?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS                                                                    │
  │  Store (select)  ·  Product (select)  ·  Horizon: 7 Days | 30 Days        │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  load_model → make_future_dataframe(N) → predict → tail(N)                │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUTS                                                                   │
  │                                                                            │
  │  Line chart — daily yhat  [gold gradient line over horizon]                │
  │                                                                            │
  │  ┌──────────────────┐  ┌──────────────────┐                               │
  │  │ Total Forecast   │  │ Daily Average    │                               │
  │  │   4,182 units    │  │   139.4 u/day    │                               │
  │  └──────────────────┘  └──────────────────┘                               │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 2 — Trend Detection

**Business question:** Is demand rising, falling, or stable?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS                                                                    │
  │  Store (select)  ·  Product (select)  ·  Horizon slider: 7–60 days        │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  forecast.tail(N)['yhat'].diff().mean()  →  change                        │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUTS                                                                   │
  │                                                                            │
  │  Line chart — demand trajectory (blue line)                                │
  │                                                                            │
  │  ┌────────────────────────────────────────────────────────────────────┐   │
  │  │  Demand is expected to RISE by an average of 2.41 units/day       │   │
  │  │  over the next 30 days.                                            │   │
  │  └────────────────────────────────────────────────────────────────────┘   │
  │  [ green banner for rise · amber banner for drop · blue for stable ]      │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 3 — Inventory Planning

**Business question:** How much stock should be ordered for next month?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS                                                                    │
  │  Store (select)  ·  Product (select)                                       │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  forecast.tail(30)['yhat']  →  sum · max · mean                           │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUTS                                                                   │
  │                                                                            │
  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
  │  │ Recommended    │  │ Peak Day       │  │ Avg Daily      │              │
  │  │ Stock (30d)    │  │ Demand         │  │ Demand         │              │
  │  │  4,182 units   │  │  198 units     │  │  139.4 u/day   │              │
  │  └────────────────┘  └────────────────┘  └────────────────┘              │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 4 — Single-Date Forecast

**Business question:** What are predicted sales on a specific future date?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS                                                                    │
  │  Store (select)  ·  Product (select)  ·  Date picker (today → +4 years)   │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  make_future_dataframe(1460) → predict → filter ds == target_date         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUTS                                                                   │
  │                                                                            │
  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
  │  │ Forecast       │  │ Lower Bound    │  │ Upper Bound    │              │
  │  │ 2025-06-15     │  │ (pessimistic)  │  │ (optimistic)   │              │
  │  │  157 units     │  │  112 units     │  │  203 units     │              │
  │  └────────────────┘  └────────────────┘  └────────────────┘              │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 5 — Portfolio Trend Analysis

**Business question:** Which store-product pairs are growing or declining most strongly?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS:  none (scans all 100 pairs automatically)                         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  iterate store_ids × product_ids → avg daily change → sort desc           │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUT                                                                    │
  │                                                                            │
  │  Store   │ Product  │ Avg Daily Change                                    │
  │  ────────┼──────────┼─────────────────                                   │
  │  S004    │ P0015    │ +3.87                                               │
  │  S001    │ P0002    │ +2.14                                               │
  │  S003    │ P0009    │ -0.43                                               │
  │  S005    │ P0011    │ -1.92                                               │
  │  ...     │ ...      │ ...                                                 │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 6 — Product Comparison

**Business question:** Which of two products has stronger demand in the same store?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS                                                                    │
  │  Store (select)  ·  Product A (select)  ·  Product B (select)             │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  run both models → concat yhat series → dual overlay chart                │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUT                                                                    │
  │                                                                            │
  │  Dual-series line chart (colour-coded per product, 30-day horizon)         │
  │                                                                            │
  │  ┌──────────────────────────┐  ┌──────────────────────────┐              │
  │  │  P0001 — Avg Daily       │  │  P0002 — Avg Daily       │              │
  │  │        142.3 units       │  │        98.7 units        │              │
  │  └──────────────────────────┘  └──────────────────────────┘              │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 7 — Restocking Alerts

**Business question:** Which store-product pairs need immediate restocking attention?

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUTS                                                                    │
  │  Demand threshold slider (0–100 units)                                     │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  PROCESSING                                                                │
  │  forecast.tail(7)['yhat'].mean() < threshold  →  flag pair                │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  OUTPUT                                                                    │
  │                                                                            │
  │  Alert banner:  "14 store-product pairs require restocking attention."     │
  │                                                                            │
  │  Store   │ Product  │ Avg Predicted Units (7d)                            │
  │  ────────┼──────────┼──────────────────────────                          │
  │  S002    │ P0007    │ 12.3  ← critical                                   │
  │  S003    │ P0014    │ 28.7                                                │
  │  ...     │ ...      │ ...                                                 │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. REST API — FastAPI

### Launch

```bash
uvicorn server:app --reload
```

Interactive docs: `http://127.0.0.1:8000/docs`

### Endpoint Map

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  FastAPI  —  Demand Forecasting REST Service                             │
  ├──────────────┬───────────────────────────┬────────────────────────────── │
  │  Method      │  Path                     │  Purpose                     │
  ├──────────────┼───────────────────────────┼────────────────────────────── │
  │  POST        │  /predict-units/          │  Day-by-day forecast          │
  │  POST        │  /detect-trend/           │  Rise / drop / stable         │
  │  POST        │  /recommend-stock/        │  Total inventory requirement  │
  │  POST        │  /forecast-date/          │  Single-date prediction       │
  │  GET         │  /trend-analysis/         │  Portfolio-wide trend scan    │
  └──────────────┴───────────────────────────┴────────────────────────────── │
```

### Request / Response examples

#### `POST /predict-units/`

```json
// Request
{
  "store_id":   "S001",
  "product_id": "P0003",
  "period":     30
}

// Response
[
  { "ds": "2024-01-02", "yhat": 143.7 },
  { "ds": "2024-01-03", "yhat": 138.2 },
  ...
]
```

#### `POST /detect-trend/`

```json
// Request
{ "store_id": "S002", "product_id": "P0007", "period": 14 }

// Response
{ "trend": "rise", "avg_change": 2.41 }
```

#### `POST /recommend-stock/`

```json
// Request
{ "store_id": "S003", "product_id": "P0012", "period": 30 }

// Response
{ "recommended_inventory": 4182 }
```

#### `POST /forecast-date/`

```json
// Request
{ "store_id": "S001", "product_id": "P0001", "forecast_date": "2025-06-15" }

// Response
{ "date": "2025-06-15", "predicted_units": 157.34 }
```

#### `GET /trend-analysis/`

```json
// Response (truncated — returns all 100 pairs sorted desc)
[
  { "store_id": "S004", "product_id": "P0015", "avg_change": 3.87 },
  { "store_id": "S001", "product_id": "P0002", "avg_change": 2.14 },
  ...
]
```

### Pydantic schemas

```python
class ForecastRequest(BaseModel):
    store_id:   str
    product_id: str
    period:     int

class DateForecastRequest(BaseModel):
    store_id:      str
    product_id:    str
    forecast_date: date

class CompareRequest(BaseModel):
    store_id:  str
    productA:  str
    productB:  str
```

---

## 10. Evaluation

`evaluate.py` provides a visual evaluation loop using Prophet's native plotting:

```python
model.plot(forecast)
```

This renders a matplotlib figure per model:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Forecast — S001-P0001                                                 │
  │                                                                        │
  │  400 ┤                                                          ·····  │
  │      │                                           ············         │
  │  300 ┤                       ·····················                    │
  │      │  ●●●●●●●● ●●●●●●●●●●●●  ← observed actuals (black dots)       │
  │  200 ┤─────────────────────────────────────────── ← yhat (blue line)  │
  │      │░░░░░░░░░░░░░░░░░░░░░░░░░ ← uncertainty band (shaded)           │
  │  100 ┤                                                                 │
  │      └───────────────────────────────────────────────────────────     │
  │        Jan 22       Jul 22       Jan 23       Jul 23       Jan 24     │
  └────────────────────────────────────────────────────────────────────────┘
```

`model.plot_components(forecast)` produces a decomposition chart:

| Panel | What it shows |
|---|---|
| Trend | Long-run growth or decline over the 2-year history |
| Weekly seasonality | Mon–Sun demand pattern |
| Yearly seasonality | Annual demand cycle (12-month pattern) |

---

## 11. Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Clone & install

```bash
git clone https://github.com/GeekSomesh/Time-Series-Forecasting.git
cd Time-Series-Forecasting
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.x | Interactive web dashboard |
| `prophet` | 1.1+ | Time-series forecasting engine |
| `pandas` | 1.x+ | Data manipulation |
| `joblib` | 1.x | Model serialisation / deserialisation |
| `matplotlib` | 3.x | Forecast visualisation in evaluation |
| `numpy` | 1.x | Numerical utilities |
| `fastapi` | 0.x | REST API framework |
| `pydantic` | 1.x / 2.x | API request/response validation |

---

## 12. Usage Guide

### Train models from scratch

```bash
python main.py
```

Trains and saves all 100 Prophet models to `models/prophet_models/`.

### Launch the Streamlit dashboard

```bash
python -m streamlit run app.py
# → http://localhost:8501
```

### Launch the REST API

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

### Run the EDA notebook

```bash
jupyter notebook notebooks/EDA_and_Feature_Engineering.ipynb
```

### Evaluate trained models

```bash
python -c "
from src.evaluate import evaluate_models
evaluate_models(
    'data/retail_warehouse_inventory_dataset.csv',
    'models/prophet_models'
)
"
```

---

## 13. Model Inventory

100 Prophet models are pre-trained and stored as `.joblib` artefacts.

```
  Model naming:  {STORE_ID}_{PRODUCT_ID}_prophet.joblib
  Example:       S003_P0014_prophet.joblib  →  Store 3, Product 14

  ┌────────┬─────────────────────────────────────────────┬───────┐
  │  Store │  Products                                   │ Count │
  ├────────┼─────────────────────────────────────────────┼───────┤
  │  S001  │  P0001 – P0020                             │  20   │
  │  S002  │  P0001 – P0020                             │  20   │
  │  S003  │  P0001 – P0020                             │  20   │
  │  S004  │  P0001 – P0020                             │  20   │
  │  S005  │  P0001 – P0020                             │  20   │
  ├────────┼─────────────────────────────────────────────┼───────┤
  │        │                                   TOTAL     │ 100   │
  └────────┴─────────────────────────────────────────────┴───────┘
```

All models are loaded on-demand. No model is held in memory until requested, keeping the system memory-efficient at any scale.

---

## 14. Key Results & Insights

### Dataset quality

- **Zero missing values** across all 73,100 records — no imputation required
- **Balanced coverage**: every store-product pair has continuous daily observations across the full 2-year window
- **Units Sold** ranges 0–499 per day per pair (mean 136.5, std 108.9) — high variance across pairs makes a single global model insufficient; per-pair modelling is essential

### Model design decisions

| Decision | Rationale |
|---|---|
| One model per store-product | Retail demand is highly specific to location and category; shared models cannot capture per-pair seasonality |
| Weekly seasonality ON | Clear Mon–Sun sales cycles exist in retail data; improves short-horizon accuracy |
| Daily seasonality OFF | At daily granularity there is no sub-day signal; enabling introduces noise |
| 4-year horizon (1,460 days) | Supports multi-year strategic planning scenarios without retraining |
| Joblib serialisation | Fast binary serialisation; load latency is negligible even for 100 models |

### Forecast reliability notes

```
  Short horizon  (7–14 days)   — narrow confidence bands, highest reliability
  Medium horizon (30 days)     — wider bands, good for inventory planning
  Long horizon   (90+ days)    — qualitative directional guidance only
  Max horizon    (4 years)     — strategic planning, treat bounds as scenarios
```

### Notable data observations

- The pre-existing `Demand Forecast` column in the source data has a minimum of -9.99, indicating the baseline forecast can produce negative values. Prophet's additive model is bounded more realistically.
- `Holiday/Promotion` is flagged on approximately 50% of all days, suggesting promotional periods are embedded evenly throughout the dataset.
- Price and Competitor Pricing have very similar distributions (mean ~55, range 5–105), indicating competitive pricing parity across the retail network.

---

## 15. Tech Stack

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   Language          Python 3.10+                                        │
  │                                                                         │
  │   Forecasting       Facebook Prophet  — additive time-series model      │
  │   Data              pandas            — ETL, resampling, aggregation     │
  │   Serialisation     joblib            — model persistence/loading        │
  │   Visualisation     Streamlit charts  — line charts, metric displays     │
  │   Evaluation plot   matplotlib        — Prophet component figures        │
  │   Numerics          numpy             — seed control, array operations   │
  │                                                                         │
  │   Dashboard         Streamlit 1.x     — multi-tab single-page app        │
  │   REST API          FastAPI           — async ASGI service               │
  │   Validation        Pydantic          — request/response schemas         │
  │   API Server        Uvicorn           — ASGI runner                     │
  │   Notebook          Jupyter + ipykernel — EDA & feature engineering      │
  │                                                                         │
  │   Dev Tools         git · pip · VS Code                                 │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

<div align="center">

Built with precision. Designed for scale.

</div>
