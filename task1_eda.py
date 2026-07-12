"""
=============================================================
TASK 1 — Data Loading, Merging & Deep Exploration
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── styling ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'grid.color': '#334155', 'text.color': '#e2e8f0',
    'font.family': 'DejaVu Sans', 'axes.titlecolor': '#f1f5f9',
    'figure.dpi': 130
})
PALETTE = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e',
           '#a78bfa', '#34d399', '#fb923c']

# ── 1.1  Load Superstore dataset ─────────────────────────────
df = pd.read_csv('train.csv')
print(f"✅ Loaded train.csv  → Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}\n")

# ── 1.2  Parse dates ──────────────────────────────────────────
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)

# ── 1.3  Extract time features ───────────────────────────────
df['Year']       = df['Order Date'].dt.year
df['Month']      = df['Order Date'].dt.month
df['WeekNumber'] = df['Order Date'].dt.isocalendar().week.astype(int)
df['DayOfWeek']  = df['Order Date'].dt.day_name()
df['Quarter']    = df['Order Date'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]:  return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else:                     return 'Fall'

df['Season'] = df['Month'].apply(get_season)
df['ShipDelay'] = (df['Ship Date'] - df['Order Date']).dt.days

# ── 1.4  Data quality checks ─────────────────────────────────
print("=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  None found ✅")

print(f"\n=== DUPLICATES ===")
dups = df.duplicated().sum()
print(f"  Duplicate rows: {dups}")

print(f"\n=== DATA TYPES ===")
print(df.dtypes.to_string())

# ── 1.5  Aggregations ────────────────────────────────────────
daily_sales   = df.groupby('Order Date')['Sales'].sum().reset_index()
daily_sales.columns = ['Date', 'Sales']
daily_sales.set_index('Date', inplace=True)

weekly_sales  = daily_sales.resample('W').sum()
monthly_sales = daily_sales.resample('ME').sum()

print(f"\n✅ Daily  sales shape : {daily_sales.shape}")
print(f"✅ Weekly sales shape : {weekly_sales.shape}")
print(f"✅ Monthly sales shape: {monthly_sales.shape}")

# ── 1.6  Business Questions ──────────────────────────────────
print("\n" + "="*55)
print("BUSINESS QUESTION 1: Highest Revenue Category")
cat_rev = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print(cat_rev.to_string())

print("\nBUSINESS QUESTION 2: Most Consistent Sales Growth by Region")
reg_yr = df.groupby(['Region', 'Year'])['Sales'].sum().unstack()
reg_growth = reg_yr.pct_change(axis=1).mean(axis=1).sort_values(ascending=False)
print(reg_growth.to_string())

print("\nBUSINESS QUESTION 3: Avg Ship Delay by Region")
ship = df.groupby('Region')['ShipDelay'].mean().sort_values()
print(ship.to_string())

print("\nBUSINESS QUESTION 4: Consistent Monthly Sales Spikes")
mo_yr = df.groupby(['Year', 'Month'])['Sales'].sum().unstack(level=0)
mo_avg = mo_yr.mean(axis=1)
top_months = mo_avg.sort_values(ascending=False).head(3)
print(top_months.to_string())

# ── 1.7  Save charts ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Task 1 — Sales Exploration Dashboard', fontsize=16,
             fontweight='bold', color='#f1f5f9', y=0.98)

# Chart A — Revenue by Category
ax = axes[0, 0]
bars = ax.bar(cat_rev.index, cat_rev.values, color=PALETTE[:3], edgecolor='#0f172a', linewidth=0.8)
ax.set_title('Total Revenue by Category', fontweight='bold')
ax.set_ylabel('Total Sales ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5000,
            f'${b.get_height():,.0f}', ha='center', va='bottom', fontsize=9, color='#f1f5f9')
ax.grid(axis='y', alpha=0.4)

# Chart B — Avg Sales Growth by Region
ax = axes[0, 1]
colors_r = [PALETTE[i] for i in range(len(reg_growth))]
bars2 = ax.bar(reg_growth.index, reg_growth.values * 100, color=colors_r, edgecolor='#0f172a')
ax.set_title('Avg YoY Sales Growth Rate by Region (%)', fontweight='bold')
ax.set_ylabel('Avg YoY Growth (%)')
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
            f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=9, color='#f1f5f9')
ax.grid(axis='y', alpha=0.4)

# Chart C — Ship Delay by Region
ax = axes[1, 0]
ax.barh(ship.index, ship.values, color=PALETTE[4:8], edgecolor='#0f172a')
ax.set_title('Avg Shipping Delay by Region (days)', fontweight='bold')
ax.set_xlabel('Avg Days to Ship')
for i, v in enumerate(ship.values):
    ax.text(v + 0.02, i, f'{v:.2f}d', va='center', fontsize=9, color='#f1f5f9')
ax.grid(axis='x', alpha=0.4)

# Chart D — Monthly Sales Seasonality
ax = axes[1, 1]
mo_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.bar(range(1, 13), mo_avg.values, color=PALETTE, edgecolor='#0f172a')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(mo_names, fontsize=8)
ax.set_title('Avg Monthly Sales Across All Years', fontweight='bold')
ax.set_ylabel('Avg Sales ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig('charts/task1_exploration.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Chart saved: charts/task1_exploration.png")

# ── Return dataframes for use in other tasks ──────────────────
print("\n✅ Task 1 COMPLETE")
