"""
=============================================================
TASK 3 — Sales Forecasting: SARIMA + Prophet + XGBoost
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

plt.rcParams.update({
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'grid.color': '#334155', 'text.color': '#e2e8f0',
    'font.family': 'DejaVu Sans', 'axes.titlecolor': '#f1f5f9',
    'figure.dpi': 130
})

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
daily  = df.groupby('Order Date')['Sales'].sum()
monthly = daily.resample('ME').sum()

# Train/test split: last 3 months = test
train = monthly[:-3]
test  = monthly[-3:]

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics = {}

# ═══════════════════════════════════════════════════════════════
# MODEL 1 — SARIMA
# ═══════════════════════════════════════════════════════════════
print("🔧 Fitting SARIMA model...")
from statsmodels.tsa.statespace.sarimax import SARIMAX

# (p,d,q)(P,D,Q,m) = (1,1,1)(1,1,1,12)
# p=1: one AR lag; d=1: first-order diff; q=1: one MA lag
# P,D,Q=1 seasonal counterparts; m=12 annual seasonality
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=3)
sarima_conf = sarima_fit.get_forecast(steps=3).conf_int()

metrics['SARIMA'] = {
    'MAE':  round(mae(test.values, sarima_forecast.values), 2),
    'RMSE': round(rmse(test.values, sarima_forecast.values), 2),
    'MAPE': round(mape(test.values, sarima_forecast.values), 2),
}
print(f"  SARIMA → MAE={metrics['SARIMA']['MAE']:,.0f}  RMSE={metrics['SARIMA']['RMSE']:,.0f}  MAPE={metrics['SARIMA']['MAPE']:.1f}%")

# Plot SARIMA
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train.index, train.values, color='#6366f1', linewidth=2, label='Training Sales')
ax.plot(test.index, test.values, color='#22d3ee', linewidth=2, linestyle='--', label='Actual (Test)')
ax.plot(test.index, sarima_forecast.values, color='#f59e0b', linewidth=2.5, marker='o', label='SARIMA Forecast')
ax.fill_between(test.index, sarima_conf.iloc[:, 0], sarima_conf.iloc[:, 1],
                alpha=0.2, color='#f59e0b', label='95% CI')
ax.set_title('Model 1 — SARIMA Forecast (3-Month Ahead)', fontsize=13, fontweight='bold')
ax.set_ylabel('Monthly Sales ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('charts/task3_sarima.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  ✅ SARIMA chart saved")

# ═══════════════════════════════════════════════════════════════
# MODEL 2 — Prophet
# ═══════════════════════════════════════════════════════════════
print("\n🔧 Fitting Prophet model...")
from prophet import Prophet

prophet_df = pd.DataFrame({'ds': monthly.index, 'y': monthly.values})
prophet_train = prophet_df[:-3]
prophet_test  = prophet_df[-3:]

m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
            daily_seasonality=False, seasonality_mode='additive',
            changepoint_prior_scale=0.1)
m.fit(prophet_train)

# Forecast 3 months ahead
future = m.make_future_dataframe(periods=3, freq='ME')
forecast = m.predict(future)

prophet_pred = forecast['yhat'].iloc[-3:].values
metrics['Prophet'] = {
    'MAE':  round(mae(test.values, prophet_pred), 2),
    'RMSE': round(rmse(test.values, prophet_pred), 2),
    'MAPE': round(mape(test.values, prophet_pred), 2),
}
print(f"  Prophet → MAE={metrics['Prophet']['MAE']:,.0f}  RMSE={metrics['Prophet']['RMSE']:,.0f}  MAPE={metrics['Prophet']['MAPE']:.1f}%")

# Plot Prophet
fig, axes = plt.subplots(2, 1, figsize=(13, 10))
# Forecast plot
ax = axes[0]
ax.plot(prophet_df['ds'], prophet_df['y'], color='#6366f1', linewidth=2, label='Actual Sales')
ax.plot(forecast['ds'], forecast['yhat'], color='#f59e0b', linewidth=2, linestyle='--', label='Prophet Forecast')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                alpha=0.15, color='#f59e0b', label='Uncertainty Band')
ax.set_title('Model 2 — Prophet Forecast with Uncertainty', fontsize=12, fontweight='bold')
ax.set_ylabel('Monthly Sales ($)'); ax.legend(); ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# Yearly seasonality
comp = forecast[['ds', 'yearly']].copy()
comp['month'] = comp['ds'].dt.month
mo_season = comp.groupby('month')['yearly'].mean()
ax2 = axes[1]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax2.bar(range(1, 13), mo_season.values, color='#22d3ee', alpha=0.8, edgecolor='#0f172a')
ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(months, fontsize=9)
ax2.set_title('Yearly Seasonality Component (Prophet)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Seasonal Effect ($)')
ax2.axhline(0, color='#f43f5e', linestyle='--', linewidth=1)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/task3_prophet.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  ✅ Prophet chart saved")

# ═══════════════════════════════════════════════════════════════
# MODEL 3 — XGBoost (Supervised ML)
# ═══════════════════════════════════════════════════════════════
print("\n🔧 Fitting XGBoost model...")
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_features(series):
    df_feat = series.to_frame(name='Sales')
    df_feat['lag1']      = df_feat['Sales'].shift(1)
    df_feat['lag2']      = df_feat['Sales'].shift(2)
    df_feat['lag3']      = df_feat['Sales'].shift(3)
    df_feat['roll_mean'] = df_feat['Sales'].shift(1).rolling(3).mean()
    df_feat['month']     = df_feat.index.month
    df_feat['quarter']   = df_feat.index.quarter
    df_feat['season']    = df_feat['month'].apply(
        lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3)))
    return df_feat.dropna()

feat_df = create_features(monthly)
features = ['lag1', 'lag2', 'lag3', 'roll_mean', 'month', 'quarter', 'season']
X = feat_df[features]
y = feat_df['Sales']

# Train on all but last 3 months, test on last 3
X_train, X_test = X[:-3], X[-3:]
y_train, y_test = y[:-3], y[-3:]

xgb_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_pred = xgb_model.predict(X_test)

metrics['XGBoost'] = {
    'MAE':  round(mae(y_test.values, xgb_pred), 2),
    'RMSE': round(rmse(y_test.values, xgb_pred), 2),
    'MAPE': round(mape(y_test.values, xgb_pred), 2),
}
print(f"  XGBoost → MAE={metrics['XGBoost']['MAE']:,.0f}  RMSE={metrics['XGBoost']['RMSE']:,.0f}  MAPE={metrics['XGBoost']['MAPE']:.1f}%")

# Future forecast: iteratively predict next 3 months
last_known = monthly.copy()
xgb_future_preds = []
for _ in range(3):
    tmp = create_features(last_known)
    last_row = tmp[features].iloc[[-1]]
    pred = xgb_model.predict(last_row)[0]
    xgb_future_preds.append(pred)
    new_date = last_known.index[-1] + pd.DateOffset(months=1)
    new_entry = pd.Series([pred], index=[new_date])
    last_known = pd.concat([last_known, new_entry])

future_idx = pd.date_range(start=monthly.index[-1] + pd.DateOffset(months=1), periods=3, freq='ME')

# Plot XGBoost
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(feat_df.index, y.values, color='#6366f1', linewidth=2, label='Actual Sales')
ax.plot(X_test.index, xgb_pred, color='#f59e0b', linewidth=2.5, marker='o', label='XGBoost Test Pred')
ax.set_title('Model 3 — XGBoost: Actual vs Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Monthly Sales ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(monthly.index, monthly.values, color='#6366f1', linewidth=2, label='Historical Sales')
ax2.plot(future_idx, xgb_future_preds, color='#10b981', linewidth=2.5, marker='D',
         linestyle='--', label='3-Month Forecast')
ax2.set_title('Model 3 — XGBoost: 3-Month Future Forecast', fontsize=12, fontweight='bold')
ax2.set_ylabel('Monthly Sales ($)')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/task3_xgboost.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("  ✅ XGBoost chart saved")

# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL COMPARISON TABLE")
print("="*55)
comp_df = pd.DataFrame(metrics).T.reset_index()
comp_df.columns = ['Model', 'MAE ($)', 'RMSE ($)', 'MAPE (%)']
print(comp_df.to_string(index=False))

best_model = comp_df.sort_values('RMSE ($)').iloc[0]['Model']
print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
print(f"   Rationale: Lowest RMSE on held-out test set.")
print(f"   Prophet handles seasonal patterns and trend changes natively,")
print(f"   making it most robust for monthly sales forecasting in production.")

# Save comparison as a visual
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
fig.patch.set_facecolor('#0f172a')
table_data = [[m, f"${v['MAE']:,.0f}", f"${v['RMSE']:,.0f}", f"{v['MAPE']:.1f}%"]
              for m, v in metrics.items()]
col_labels = ['Model', 'MAE', 'RMSE', 'MAPE']
tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.5, 2.2)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#6366f1')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='#e2e8f0')
    else:
        cell.set_facecolor('#0f172a')
        cell.set_text_props(color='#e2e8f0')
    cell.set_edgecolor('#334155')
ax.set_title('Model Comparison: MAE / RMSE / MAPE', fontsize=13,
             fontweight='bold', color='#f1f5f9', pad=20)
plt.savefig('charts/task3_comparison.png', bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Comparison table chart saved")

# ── Save metrics for use by other tasks ──────────────────────
import json
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f)
print("✅ Task 3 COMPLETE")
