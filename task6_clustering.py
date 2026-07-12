"""
=============================================================
TASK 6 — Product Demand Segmentation using K-Means
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'grid.color': '#334155', 'text.color': '#e2e8f0',
    'font.family': 'DejaVu Sans', 'axes.titlecolor': '#f1f5f9',
    'figure.dpi': 130
})
CLUSTER_COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e']

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Year']  = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

# ── Feature Engineering per sub-category ─────────────────────
sub = df.groupby('Sub-Category').agg(
    total_sales     = ('Sales', 'sum'),
    avg_order_value = ('Sales', 'mean'),
    order_count     = ('Sales', 'count'),
).reset_index()

# Sales growth rate: YoY (2016→2017)
yr = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack(fill_value=0)
if 2016 in yr.columns and 2017 in yr.columns:
    growth = ((yr[2017] - yr[2016]) / yr[2016].replace(0, np.nan) * 100).reset_index()
    growth.columns = ['Sub-Category', 'growth_rate']
else:
    growth = pd.DataFrame({'Sub-Category': sub['Sub-Category'], 'growth_rate': 0.0})

# Sales volatility: std of monthly sales
vol = df.groupby(['Sub-Category', df['Order Date'].dt.to_period('M')])['Sales'].sum()
vol = vol.groupby(level=0).std().reset_index()
vol.columns = ['Sub-Category', 'volatility']

# Merge features
feat = sub.merge(growth, on='Sub-Category').merge(vol, on='Sub-Category')
feat.fillna(0, inplace=True)

print("✅ Feature matrix shape:", feat.shape)
print(feat[['Sub-Category', 'total_sales', 'growth_rate', 'volatility', 'avg_order_value']].to_string())

# ── Scale features ────────────────────────────────────────────
feature_cols = ['total_sales', 'growth_rate', 'volatility', 'avg_order_value']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feat[feature_cols])

# ── Elbow Method ──────────────────────────────────────────────
inertia = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_range, inertia, 'o-', color='#6366f1', linewidth=2.5, markersize=8)
ax.axvline(x=4, color='#f43f5e', linestyle='--', linewidth=1.5, label='Optimal K=4')
ax.set_title('Elbow Method — Optimal Number of Clusters', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Clusters (K)'); ax.set_ylabel('Inertia (WCSS)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('charts/task6_elbow.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Elbow chart saved")

# ── Final K-Means (K=4) ───────────────────────────────────────
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
feat['Cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters based on centroids
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=feature_cols
)
print("\n=== CLUSTER CENTERS ===")
print(centers.to_string())

# Assign meaningful labels
# Sort by total_sales to determine high/low volume
centers_sorted = centers.sort_values('total_sales')
labels_map = {}
for i, row in centers.iterrows():
    ts = row['total_sales']
    gr = row['growth_rate']
    vol_val = row['volatility']
    if ts > centers['total_sales'].median() and gr > 5:
        labels_map[i] = 'High Volume, Growing Demand'
    elif ts > centers['total_sales'].median() and gr <= 5:
        labels_map[i] = 'High Volume, Stable Demand'
    elif ts <= centers['total_sales'].median() and vol_val > centers['volatility'].median():
        labels_map[i] = 'Low Volume, High Volatility'
    else:
        labels_map[i] = 'Low Volume, Declining Demand'

feat['Cluster_Label'] = feat['Cluster'].map(labels_map)

print("\n=== SUB-CATEGORIES BY CLUSTER ===")
for label in feat['Cluster_Label'].unique():
    items = feat[feat['Cluster_Label'] == label]['Sub-Category'].tolist()
    print(f"\n  [{label}]")
    for item in items:
        row = feat[feat['Sub-Category'] == item].iloc[0]
        print(f"    • {item:25s}  Sales=${row['total_sales']:>10,.0f}  "
              f"Growth={row['growth_rate']:+.1f}%  Vol=${row['volatility']:,.0f}")

# ── PCA 2D Scatter ────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
feat['PCA1'] = X_pca[:, 0]
feat['PCA2'] = X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Task 6 — Product Demand Segmentation (K-Means)',
             fontsize=14, fontweight='bold', color='#f1f5f9')

# Left: PCA scatter
ax = axes[0]
for idx, label in labels_map.items():
    mask = feat['Cluster'] == idx
    ax.scatter(feat.loc[mask, 'PCA1'], feat.loc[mask, 'PCA2'],
               s=120, color=CLUSTER_COLORS[idx],
               label=label, edgecolors='white', linewidth=0.6, zorder=3)
    # Annotate sub-categories
    for _, row in feat[mask].iterrows():
        ax.annotate(row['Sub-Category'], (row['PCA1'], row['PCA2']),
                    fontsize=6.5, color='#cbd5e1',
                    xytext=(4, 4), textcoords='offset points')
ax.set_title('PCA 2D Cluster Visualization', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)

# Right: stocking strategy table
ax2 = axes[1]
ax2.axis('off')
strategy = [
    ['Cluster', 'Stocking Strategy'],
    ['High Volume, Growing', 'Increase buffer stock 20–30%;\nPrioritize supplier contracts'],
    ['High Volume, Stable', 'Maintain current levels;\nOptimize reorder points'],
    ['Low Vol, High Volatility', 'Safety stock + fast-replenishment;\nMonitor weekly'],
    ['Low Vol, Declining', 'Reduce inventory;\nPhase out slow movers'],
]
tbl = ax2.table(cellText=strategy[1:], colLabels=strategy[0],
                loc='center', cellLoc='left')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.3, 2.8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#334155')
    if r == 0:
        cell.set_facecolor('#6366f1')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#1e293b' if r % 2 == 0 else '#0f172a')
        cell.set_text_props(color='#e2e8f0')
ax2.set_title('Stocking Strategy per Cluster', fontweight='bold',
              color='#f1f5f9', pad=20)

plt.tight_layout()
plt.savefig('charts/task6_clusters.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Cluster chart saved")

# Save cluster assignments
feat[['Sub-Category', 'total_sales', 'growth_rate',
      'volatility', 'avg_order_value', 'Cluster', 'Cluster_Label']]\
    .to_csv('clusters.csv', index=False)
print("✅ Cluster table saved to clusters.csv")
print("\n✅ Task 6 COMPLETE")
