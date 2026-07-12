"""
=============================================================
TASK 4 — Category & Region Level Forecasting (Prophet)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet

plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'grid.color': '#334155', 'text.color': '#e2e8f0',
    'font.family': 'DejaVu Sans', 'axes.titlecolor': '#f1f5f9',
    'figure.dpi': 130
})
PALETTE = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e']

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

SEGMENTS = {
    'Furniture':       {'col': 'Category',  'val': 'Furniture'},
    'Technology':      {'col': 'Category',  'val': 'Technology'},
    'Office Supplies': {'col': 'Category',  'val': 'Office Supplies'},
    'West Region':     {'col': 'Region',    'val': 'West'},
    'East Region':     {'col': 'Region',    'val': 'East'},
}

forecasts = {}   # segment → forecast df

def fit_prophet_segment(segment_series):
    pdf = pd.DataFrame({'ds': segment_series.index, 'y': segment_series.values})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_mode='additive'
    )
    model.fit(pdf)
    future = model.make_future_dataframe(periods=3, freq='ME')
    fc = model.predict(future)
    return pdf, fc

# ── Run Prophet for each segment ─────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))

for idx, (seg_name, seg_info) in enumerate(SEGMENTS.items()):
    subset = df[df[seg_info['col']] == seg_info['val']]
    monthly_seg = subset.groupby('Order Date')['Sales'].sum().resample('ME').sum()
    pdf, fc = fit_prophet_segment(monthly_seg)
    forecasts[seg_name] = fc

    color = PALETTE[idx]
    # Plot historical
    ax.plot(pdf['ds'], pdf['y'], color=color, linewidth=1.5, alpha=0.7)
    # Plot forecast (future only)
    future_fc = fc[fc['ds'] > pdf['ds'].max()]
    ax.plot(future_fc['ds'], future_fc['yhat'],
            color=color, linewidth=2.5, linestyle='--',
            marker='o', markersize=8, label=f"{seg_name}")
    ax.fill_between(future_fc['ds'],
                    future_fc['yhat_lower'].clip(lower=0),
                    future_fc['yhat_upper'],
                    alpha=0.07, color=color)

    # Print summary
    print(f"✅ {seg_name:20s} → 3-month forecast avg: "
          f"${future_fc['yhat'].mean():,.0f}")

# ── Mark historical vs future ─────────────────────────────────
all_hist_max = df['Order Date'].max()
ax.axvline(x=all_hist_max, color='#f43f5e', linestyle=':', linewidth=1.5,
           label='Forecast Start')

ax.set_title('Task 4 — 3-Month Forecast by Category & Region (Prophet)',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Monthly Sales ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/task4_segment_forecasts.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Segment forecast chart saved")

# ── Which segment has strongest upcoming growth? ──────────────
print("\n=== UPCOMING GROWTH ANALYSIS ===")
for seg_name, fc in forecasts.items():
    future_fc = fc.tail(3)
    hist_avg  = fc[:-3]['yhat'].mean()
    fut_avg   = future_fc['yhat'].mean()
    growth    = (fut_avg - hist_avg) / hist_avg * 100
    print(f"  {seg_name:20s}: forecast avg ${fut_avg:,.0f}  ({growth:+.1f}% vs historical avg)")

print("\n✅ Task 4 COMPLETE")
