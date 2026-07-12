# 📊 End-to-End Sales Forecasting & Demand Intelligence System

An interactive, multi-page Streamlit dashboard for sales forecasting, demand analysis, anomaly detection, and customer segmentation — powered by Prophet, SARIMA, XGBoost, and more.

---

## 🚀 One-Click Deploy (Streamlit Cloud) — Recommended

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account
2. Click **"New app"**
3. Fill in the fields:
   - **Repository:** `Ravi-attada/End-to-End-Sales-Forecasting-and-Demand-Intelligence-System`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**

> ✅ No setup needed. Streamlit Cloud reads `requirements.txt` and installs everything automatically.

---

## 🖥️ Run Locally

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ravi-attada/End-to-End-Sales-Forecasting-and-Demand-Intelligence-System.git
cd End-to-End-Sales-Forecasting-and-Demand-Intelligence-System
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📦 Tech Stack

| Library | Purpose |
|--------|---------|
| `streamlit` | Interactive dashboard UI |
| `plotly` | Interactive charts & visualizations |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `prophet` | Time-series forecasting |
| `statsmodels` | SARIMA / statistical models |
| `xgboost` | ML-based forecasting |
| `scikit-learn` | Clustering & anomaly detection |
| `seaborn` / `matplotlib` | Static visualizations |
| `python-docx` | Report generation |

---

## 📁 Project Structure

```
├── app.py                    # Main Streamlit dashboard (4 pages)
├── requirements.txt          # All dependencies
├── train.csv                 # Sales dataset
├── vgsales.csv               # Video game sales dataset
├── task1_eda.py              # Exploratory Data Analysis
├── task2_timeseries.py       # Time-series decomposition
├── task3_forecasting.py      # Prophet / SARIMA / XGBoost forecasting
├── task4_segment_forecast.py # Segment-level forecasting
├── task5_anomaly.py          # Anomaly detection
├── task6_clustering.py       # Customer segmentation
├── task8_report.py           # Report generation
├── run_all.py                # Run all tasks sequentially
└── charts/                   # Generated chart images
```

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `streamlit: command not found` | Run `pip install streamlit` first |
| App won't start on Streamlit Cloud | Click **Manage app → Reboot app** |
| Port already in use | Run `streamlit run app.py --server.port 8502` |

---

## 👤 Author

**Ravi Attada** — [GitHub](https://github.com/Ravi-attada)