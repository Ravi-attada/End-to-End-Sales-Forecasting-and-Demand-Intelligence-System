"""
=============================================================
TASK 5 — Anomaly Detection: Isolation Forest + Z-Score
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'grid.color': '#334155', 'text.color': '#e2e8f0',
    'font.family': 'DejaVu Sans', 'axes.titlecolor': '#f1f5f9',
    'figure.dpi': 130
})

# ── Load & prepare weekly data ────────────────────────────────
df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
daily  = df.groupby('Order Date')['Sales'].sum()
weekly = daily.resample('W').sum().reset_index()
weekly.columns = ['Date', 'Sales']
weekly['week_num'] = range(len(weekly))

print(f"✅ Weekly series: {len(weekly)} weeks")

# ── METHOD 1: Isolation Forest ────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(weekly[['Sales', 'week_num']])

iso = IsolationForest(contamination=0.07, random_state=42, n_estimators=200)
weekly['IF_label'] = iso.fit_predict(X)            # -1 = anomaly
weekly['IF_anomaly'] = weekly['IF_label'] == -1

iso_anomalies = weekly[weekly['IF_anomaly']]
print(f"\n📍 Isolation Forest detected {len(iso_anomalies)} anomalies")

# ── METHOD 2: Z-Score on rolling mean ────────────────────────
window = 8   # 8-week rolling
weekly['rolling_mean'] = weekly['Sales'].rolling(window, center=True).mean()
weekly['rolling_std']  = weekly['Sales'].rolling(window, center=True).std()
weekly['z_score'] = (weekly['Sales'] - weekly['rolling_mean']) / weekly['rolling_std']
weekly['ZS_anomaly'] = weekly['z_score'].abs() > 2.0

zs_anomalies = weekly[weekly['ZS_anomaly']]
print(f"📍 Z-Score detected {len(zs_anomalies)} anomalies")

# ── COMPARISON ────────────────────────────────────────────────
both = weekly[weekly['IF_anomaly'] & weekly['ZS_anomaly']]
print(f"📍 Both methods agree on {len(both)} anomalies")
print("\n=== TOP ANOMALIES (Both Methods) ===")
print(both[['Date', 'Sales', 'z_score']].sort_values('Sales', ascending=False).head(10).to_string())

# ── REAL-WORLD EXPLANATIONS ───────────────────────────────────
explanations = {
    'high': "Likely caused by holiday shopping season (Black Friday / Cyber Monday) or end-of-quarter corporate purchasing.",
    'low':  "Likely caused by slow summer months, supply chain disruptions, or post-holiday spending decline."
}

# ── Plot — two subplots side by side ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 11))
fig.suptitle('Task 5 — Anomaly Detection: Isolation Forest vs Z-Score',
             fontsize=14, fontweight='bold', color='#f1f5f9', y=0.98)

# --- Method 1: Isolation Forest ---
ax = axes[0]
normal = weekly[~weekly['IF_anomaly']]
anom   = weekly[weekly['IF_anomaly']]
ax.plot(weekly['Date'], weekly['Sales'], color='#6366f1', linewidth=1.5,
        alpha=0.7, label='Weekly Sales')
ax.scatter(normal['Date'], normal['Sales'], color='#6366f1', s=20, alpha=0.5)
ax.scatter(anom['Date'],   anom['Sales'],   color='#f43f5e', s=80, zorder=5,
           marker='^', label=f'Anomaly (IF) — {len(anom)} points')
ax.set_title('Method 1 — Isolation Forest', fontsize=12, fontweight='bold')
ax.set_ylabel('Weekly Sales ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(); ax.grid(alpha=0.3)

# Annotate top 3 IF anomalies
for _, row in anom.nlargest(3, 'Sales').iterrows():
    ax.annotate(f"${row['Sales']:,.0f}\n{row['Date'].strftime('%b %Y')}",
                xy=(row['Date'], row['Sales']),
                xytext=(10, 15), textcoords='offset points',
                fontsize=7, color='#f43f5e',
                arrowprops=dict(arrowstyle='->', color='#f43f5e', lw=0.8))

# --- Method 2: Z-Score ---
ax2 = axes[1]
zn = weekly[~weekly['ZS_anomaly']]
za = weekly[weekly['ZS_anomaly']]
ax2.plot(weekly['Date'], weekly['Sales'], color='#22d3ee', linewidth=1.5,
         alpha=0.7, label='Weekly Sales')
ax2.plot(weekly['Date'], weekly['rolling_mean'], color='#f59e0b',
         linewidth=2, linestyle='--', label='8-week Rolling Mean')
ax2.fill_between(weekly['Date'],
                 (weekly['rolling_mean'] - 2 * weekly['rolling_std']).clip(lower=0),
                 weekly['rolling_mean'] + 2 * weekly['rolling_std'],
                 alpha=0.1, color='#f59e0b', label='±2 Std Dev Band')
ax2.scatter(za['Date'], za['Sales'], color='#f43f5e', s=80, zorder=5,
            marker='^', label=f'Anomaly (Z-Score) — {len(za)} points')
ax2.set_title('Method 2 — Z-Score (Rolling Mean ± 2σ)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Weekly Sales ($)')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/task5_anomalies.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Anomaly chart saved")

# ── Save anomalies table ──────────────────────────────────────
anomaly_table = weekly[weekly['IF_anomaly'] | weekly['ZS_anomaly']].copy()
anomaly_table['Method'] = anomaly_table.apply(
    lambda r: 'Both' if r['IF_anomaly'] and r['ZS_anomaly']
              else ('Isolation Forest' if r['IF_anomaly'] else 'Z-Score'), axis=1)
anomaly_table = anomaly_table[['Date', 'Sales', 'z_score', 'Method']].sort_values('Date')
anomaly_table.to_csv('anomalies.csv', index=False)
print(f"✅ Anomalies table saved: {len(anomaly_table)} rows")

print("""
=== COMPARISON OBSERVATIONS ===
• Isolation Forest detects anomalies based on overall data distribution
  (density-based) — catches both high and low outliers globally.
• Z-Score is local — flags weeks that deviate from the recent rolling trend,
  more sensitive to sudden spikes or dips relative to recent period.
• When BOTH methods agree on a point, confidence is highest that it is a
  true anomaly (not just statistical noise in one method).
• High sales anomalies: November/December — holiday season effect.
• Low sales anomalies: January/February — post-holiday spending decline.
""")
print("✅ Task 5 COMPLETE")
