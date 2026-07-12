# 📊 End-to-End Sales Forecasting & Demand Intelligence System

> An enterprise-grade, interactive analytics platform that combines classical time-series models, machine learning, and business intelligence into a single Streamlit dashboard — built to help businesses forecast sales, detect anomalies, and segment product demand with confidence.

---

## 🧠 Project Summary

This project delivers a **complete, production-ready data science pipeline** for retail sales analytics. Starting from raw transactional data, it performs exploratory analysis, builds and compares three forecasting models (SARIMA, Prophet, XGBoost), detects sales anomalies using dual methods, segments products by demand profile using KMeans clustering, and presents all insights through a rich, interactive 4-page Streamlit dashboard.

**The system answers three critical business questions:**
1. 🔮 *What will my sales look like in the next 1–3 months?*
2. 🚨 *When did unusual spikes or drops in sales occur, and why?*
3. 🧩 *Which product categories are growing, stable, or declining?*

---

## 📋 Project Details

### 🗃️ Dataset
| Property | Value |
|---|---|
| **Source** | Superstore Sales Dataset (`train.csv`) |
| **Records** | ~10,000+ orders |
| **Time Period** | 2014 – 2017 |
| **Key Fields** | Order Date, Sales, Region, Category, Sub-Category, Ship Date |
| **Secondary Dataset** | `vgsales.csv` — Video Game Sales (used for cross-domain demand analysis) |

---

### 📌 Task Breakdown

#### Task 1 — Exploratory Data Analysis (`task1_eda.py`)
- Loads and cleans the raw transactional dataset
- Computes summary statistics: total revenue, order volume, average order value, shipping delay
- Visualises sales distribution by **Category**, **Region**, **Sub-Category**, and **Year**
- Identifies top-performing products and geographic revenue concentration
- Outputs: `charts/task1_exploration.png`

#### Task 2 — Time-Series Analysis (`task2_timeseries.py`)
- Resamples daily transactions into **monthly** and **weekly** aggregations
- Performs **ADF (Augmented Dickey-Fuller)** stationarity test
- Applies **seasonal decomposition** (trend + seasonality + residual) using STL
- Visualises rolling mean and rolling standard deviation
- Outputs: `charts/task2_*.png`

#### Task 3 — Sales Forecasting (`task3_forecasting.py`)
Trains and evaluates **three forecasting models** on a held-out 3-month test set:

| Model | Type | Configuration |
|---|---|---|
| **SARIMA** | Statistical | Order (1,1,1)(1,1,1,12) — captures monthly & annual seasonality |
| **Prophet** | Bayesian Structural | Yearly seasonality + changepoint detection |
| **XGBoost** | Machine Learning | Lag features (t-1, t-2, t-3), rolling mean, month, quarter, season |

**Evaluation Metrics:** MAE, RMSE, MAPE on test set  
**Winner selected** by lowest RMSE — typically Prophet, due to its native handling of seasonal patterns and trend shifts  
- Outputs: `charts/task3_*.png`, `model_metrics.json`

#### Task 4 — Segment-Level Forecasting (`task4_segment_forecast.py`)
- Forecasts sales independently for each **product Category** and **geographic Region**
- Identifies which segments are growing fastest vs. declining
- Outputs: `charts/task4_segment_forecasts.png`

#### Task 5 — Anomaly Detection (`task5_anomaly.py`)
Detects unusual sales weeks using **two complementary methods**:

| Method | How It Works |
|---|---|
| **Isolation Forest** | ML-based; isolates outliers by random feature splits (contamination=7%) |
| **Z-Score (Rolling)** | Statistical; flags weeks where \|z-score\| > 2.0 using an 8-week rolling window |

- Anomalies agreed upon by **both methods** are flagged as high-confidence
- Business context is auto-generated (e.g., Black Friday spikes, post-holiday dips)
- Outputs: `charts/task5_anomalies.png`, `anomalies.csv`

#### Task 6 — Customer & Product Segmentation (`task6_clustering.py`)
- Engineers features per Sub-Category: **total sales**, **YoY growth rate**, **demand volatility**, **avg order value**
- Runs **KMeans clustering (K=4)** determined via Elbow Method
- Reduces dimensions with **PCA** for 2D visualisation
- Assigns business labels to each cluster:

| Cluster Label | Inventory Strategy |
|---|---|
| 📈 High Volume, Growing | Increase buffer stock by 20–30% |
| ✅ High Volume, Stable | Maintain & optimise reorder points |
| ⚡ Low Volume, High Volatility | Safety stock + weekly monitoring |
| 📉 Low Volume, Declining | Reduce inventory, phase out |

- Outputs: `charts/task6_*.png`, `clusters.csv`

#### Task 7 — Interactive Dashboard (`app.py`)
A **4-page Streamlit dashboard** with dark theme, Plotly charts, and sidebar navigation:

| Page | Contents |
|---|---|
| 🏠 **Sales Overview** | KPI cards, filters by Region/Category/Year, revenue by year, category pie chart, monthly trend line, regional breakdown, top 10 sub-categories |
| 🔮 **Forecast Explorer** | Live Prophet model — choose Overall / Category / Region, adjust forecast horizon (1–3 months), see MAE/RMSE/next-month forecast, confidence bands |
| 🚨 **Anomaly Report** | Weekly anomaly chart (Isolation Forest vs Z-Score), anomaly table, business context explanations |
| 🧩 **Demand Segments** | PCA scatter plot of demand clusters, Elbow curve, stocking strategy per cluster, sub-category assignment table |

#### Task 8 — Automated Report Generation (`task8_report.py`)
- Programmatically generates a formatted **Word document** (`summary.docx`) using `python-docx`
- Includes all KPIs, model metrics, anomaly findings, and cluster insights with embedded charts

---

### 🏗️ Architecture

```
Raw CSV Data (train.csv)
        │
        ▼
┌──────────────────┐
│  Task 1: EDA     │ ──▶ Summary stats, distributions
└──────────────────┘
        │
        ▼
┌──────────────────┐
│  Task 2: TS      │ ──▶ Stationarity test, decomposition
│  Analysis        │
└──────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│         Task 3: Forecasting                │
│  SARIMA ──▶ Prophet ──▶ XGBoost           │
│         Model Comparison (MAE/RMSE/MAPE)   │
└────────────────────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Task 4     Task 5        Task 6
Segment    Anomaly       KMeans
Forecasts  Detection     Clustering
   │         │               │
   └────┬────┘               │
        ▼                    ▼
┌─────────────────────────────────┐
│    Task 7: Streamlit Dashboard  │ ◀── Real-time, interactive
└─────────────────────────────────┘
        │
        ▼
┌──────────────────┐
│  Task 8: Report  │ ──▶ summary.docx
└──────────────────┘
```

---

## 🚀 Deployment

### Option A — Streamlit Cloud (No-Code, Recommended)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository:** `Ravi-attada/End-to-End-Sales-Forecasting-and-Demand-Intelligence-System`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**

> ✅ `requirements.txt` is included — all dependencies install automatically.

### Option B — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Ravi-attada/End-to-End-Sales-Forecasting-and-Demand-Intelligence-System.git
cd End-to-End-Sales-Forecasting-and-Demand-Intelligence-System

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.59.1 | Interactive dashboard framework |
| `plotly` | 6.9.0 | Interactive charts & visualisations |
| `pandas` | 3.0.3 | Data manipulation & aggregation |
| `numpy` | 2.5.1 | Numerical computing |
| `prophet` | 1.3.0 | Bayesian time-series forecasting |
| `statsmodels` | 0.14.6 | SARIMA & statistical tests |
| `xgboost` | 3.3.0 | Gradient-boosted ML forecasting |
| `scikit-learn` | 1.9.0 | Clustering, anomaly detection, PCA |
| `seaborn` | 0.13.2 | Statistical visualisations |
| `matplotlib` | 3.11.0 | Static chart generation |
| `python-docx` | 1.2.0 | Automated Word report generation |

---

## 📁 Project Structure

```
├── app.py                      # 🎯 Main Streamlit dashboard (4 pages)
├── requirements.txt            # All Python dependencies
├── train.csv                   # Primary sales dataset
├── vgsales.csv                 # Video game sales dataset
├── anomalies.csv               # Pre-computed anomaly records
├── clusters.csv                # Pre-computed cluster assignments
├── summary.docx                # Auto-generated analytics report
├── task1_eda.py                # Task 1: Exploratory Data Analysis
├── task2_timeseries.py         # Task 2: Time-series decomposition
├── task3_forecasting.py        # Task 3: SARIMA + Prophet + XGBoost
├── task4_segment_forecast.py   # Task 4: Segment-level forecasting
├── task5_anomaly.py            # Task 5: Anomaly detection
├── task6_clustering.py         # Task 6: KMeans demand clustering
├── task8_report.py             # Task 8: Word document report generator
├── run_all.py                  # Run all tasks in sequence
└── charts/                     # Generated chart images
    ├── task1_exploration.png
    ├── task2_decomposition.png
    ├── task3_sarima.png
    ├── task3_prophet.png
    ├── task3_xgboost.png
    ├── task3_comparison.png
    ├── task4_segment_forecasts.png
    ├── task5_anomalies.png
    ├── task6_clusters.png
    └── task6_elbow.png
```

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` on Streamlit Cloud | Ensure `requirements.txt` is committed and pushed to `main` |
| `ModuleNotFoundError` locally | Run `pip install -r requirements.txt` |
| `streamlit: command not found` | Run `pip install streamlit` first |
| App error on Streamlit Cloud | Click **Manage app → Reboot app** |
| Port already in use locally | Run `streamlit run app.py --server.port 8502` |
| `prophet` install fails | Install `pystan` first: `pip install pystan==2.19.1.1` |

---

## 👤 Author

**Ravi Attada** — [GitHub](https://github.com/Ravi-attada)