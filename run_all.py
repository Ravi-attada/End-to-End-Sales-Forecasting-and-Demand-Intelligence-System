"""
=============================================================
MASTER RUN SCRIPT — Executes all Tasks 1-8 sequentially
=============================================================
Run:  python run_all.py
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

tasks = [
    ("Task 1 — EDA",                     "task1_eda.py"),
    ("Task 2 — Time Series Analysis",    "task2_timeseries.py"),
    ("Task 3 — Forecasting Models",      "task3_forecasting.py"),
    ("Task 4 — Segment Forecasting",     "task4_segment_forecast.py"),
    ("Task 5 — Anomaly Detection",       "task5_anomaly.py"),
    ("Task 6 — K-Means Clustering",      "task6_clustering.py"),
    ("Task 8 — Executive Report",        "task8_report.py"),
]

print("=" * 60)
print("🚀 SALES FORECASTING PROJECT — Running All Tasks")
print("=" * 60)

for name, script in tasks:
    print(f"\n{'─'*60}")
    print(f"▶  Running: {name}")
    print(f"{'─'*60}")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ ERROR in {script} — exit code {result.returncode}")
    else:
        print(f"✅ {name} DONE")

print("\n" + "="*60)
print("✅ All tasks complete!")
print("   Charts saved in: charts/")
print("   Report saved:   summary.docx")
print("   Anomalies:      anomalies.csv")
print("   Clusters:       clusters.csv")
print("\nTo launch the dashboard:")
print("   streamlit run app.py")
print("="*60)
