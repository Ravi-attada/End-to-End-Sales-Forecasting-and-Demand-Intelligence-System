"""
=============================================================
TASK 7 — Streamlit Interactive Dashboard (4 Pages)
=============================================================
Run with:  streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sales Forecasting Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f172a; }
    .stSidebar { background: #1e293b !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(99,102,241,0.1);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6366f1; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .section-title {
        font-size: 1.6rem; font-weight: 700; color: #f1f5f9;
        border-left: 4px solid #6366f1;
        padding-left: 12px; margin-bottom: 20px;
    }
    .stSelectbox > div > div { background: #1e293b; border: 1px solid #334155; }
    div[data-testid="stMetricValue"] { color: #6366f1; font-size: 1.8rem !important; }
    .highlight-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #6366f1;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    plot_bgcolor='#1e293b', paper_bgcolor='#0f172a',
    font_color='#e2e8f0', title_font_color='#f1f5f9',
    xaxis=dict(gridcolor='#334155', linecolor='#334155'),
    yaxis=dict(gridcolor='#334155', linecolor='#334155'),
    legend=dict(bgcolor='rgba(30,41,59,0.8)', bordercolor='#334155', borderwidth=1),
)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']    = df['Order Date'].dt.year
    df['Month']   = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['ShipDelay'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

@st.cache_data
def load_anomalies():
    try:
        a = pd.read_csv('anomalies.csv')
        a['Date'] = pd.to_datetime(a['Date'])
        return a
    except:
        return pd.DataFrame()

@st.cache_data
def load_clusters():
    try:
        return pd.read_csv('clusters.csv')
    except:
        return pd.DataFrame()

@st.cache_data
def get_monthly_series(df):
    daily = df.groupby('Order Date')['Sales'].sum()
    return daily.resample('ME').sum().reset_index()
    
@st.cache_data
def get_weekly_series(df):
    daily = df.groupby('Order Date')['Sales'].sum()
    return daily.resample('W').sum().reset_index()

df    = load_data()
anom  = load_anomalies()
clust = load_clusters()
monthly_df = get_monthly_series(df)
weekly_df  = get_weekly_series(df)

# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0'>
        <div style='font-size:2.5rem'>📊</div>
        <div style='font-size:1.2rem; font-weight:700; color:#f1f5f9'>Sales Intelligence</div>
        <div style='font-size:0.75rem; color:#64748b'>Forecasting & Analytics</div>
    </div>
    <hr style='border-color:#334155'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate to",
        ["🏠  Sales Overview", "🔮  Forecast Explorer",
         "🚨  Anomaly Report", "🧩  Demand Segments"],
        label_visibility="collapsed"
    )
    st.markdown("<hr style='border-color:#334155'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#64748b; text-align:center'>
        Dataset: {len(df):,} orders<br>
        Period: {df['Order Date'].min().strftime('%b %Y')} –
                {df['Order Date'].max().strftime('%b %Y')}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="section-title">🏠 Sales Overview Dashboard</div>', unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────
    total_sales  = df['Sales'].sum()
    total_orders = len(df)
    avg_order    = df['Sales'].mean()
    avg_delay    = df['ShipDelay'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue",  f"${total_sales:,.0f}")
    c2.metric("📦 Total Orders",   f"{total_orders:,}")
    c3.metric("🛒 Avg Order Value", f"${avg_order:,.0f}")
    c4.metric("🚚 Avg Ship Delay",  f"{avg_delay:.1f} days")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sel_region = st.multiselect("Filter by Region", df['Region'].unique(),
                                    default=list(df['Region'].unique()))
    with col_f2:
        sel_category = st.multiselect("Filter by Category", df['Category'].unique(),
                                      default=list(df['Category'].unique()))
    with col_f3:
        sel_year = st.multiselect("Filter by Year", sorted(df['Year'].unique()),
                                  default=sorted(df['Year'].unique()))

    fdf = df[df['Region'].isin(sel_region) &
             df['Category'].isin(sel_category) &
             df['Year'].isin(sel_year)]

    # ── Chart 1: Annual Revenue Bar ────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        yr_sales = fdf.groupby('Year')['Sales'].sum().reset_index()
        fig = px.bar(yr_sales, x='Year', y='Sales',
                     color='Sales', color_continuous_scale='Purples',
                     title='📅 Total Sales by Year',
                     text=yr_sales['Sales'].apply(lambda x: f'${x:,.0f}'))
        fig.update_traces(textposition='outside')
        fig.update_layout(**PLOTLY_DARK, showlegend=False,
                          coloraxis_showscale=False, height=360)
        fig.update_xaxes(tickvals=yr_sales['Year'])
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2: Category Pie ───────────────────────────────────
    with col2:
        cat_sales = fdf.groupby('Category')['Sales'].sum().reset_index()
        fig2 = px.pie(cat_sales, names='Category', values='Sales',
                      title='🏷️ Revenue Share by Category',
                      color_discrete_sequence=['#6366f1', '#22d3ee', '#f59e0b'])
        fig2.update_layout(**PLOTLY_DARK, height=360)
        fig2.update_traces(textinfo='label+percent', textfont_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Monthly Trend Line ─────────────────────────────
    fmonthly = fdf.groupby('Order Date')['Sales'].sum().resample('ME').sum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=fmonthly['Order Date'], y=fmonthly['Sales'],
        fill='tozeroy', fillcolor='rgba(99,102,241,0.15)',
        line=dict(color='#6366f1', width=2.5),
        name='Monthly Sales'
    ))
    fig3.update_layout(**PLOTLY_DARK, title='📈 Monthly Sales Trend',
                       height=300, yaxis_tickprefix='$',
                       yaxis_tickformat=',.0f')
    st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Region + Category Heatmap ─────────────────────
    col3, col4 = st.columns(2)
    with col3:
        reg_cat = fdf.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
        fig4 = px.bar(reg_cat, x='Region', y='Sales', color='Category',
                      barmode='group', title='🗺️ Sales by Region & Category',
                      color_discrete_sequence=['#6366f1', '#22d3ee', '#f59e0b'])
        fig4.update_layout(**PLOTLY_DARK, height=360,
                           yaxis_tickprefix='$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        sub_sales = fdf.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=True).reset_index()
        fig5 = px.bar(sub_sales.tail(10), x='Sales', y='Sub-Category',
                      orientation='h', title='🏅 Top 10 Sub-Categories by Revenue',
                      color='Sales', color_continuous_scale='Purples')
        fig5.update_layout(**PLOTLY_DARK, height=360,
                           xaxis_tickprefix='$', xaxis_tickformat=',.0f',
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════
elif "Forecast" in page:
    st.markdown('<div class="section-title">🔮 Forecast Explorer</div>', unsafe_allow_html=True)

    from prophet import Prophet

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        seg_type = st.selectbox("Select Segment Type", ["Overall", "Category", "Region"])
    with col_sel2:
        if seg_type == "Category":
            seg_val = st.selectbox("Select Category", df['Category'].unique())
        elif seg_type == "Region":
            seg_val = st.selectbox("Select Region", df['Region'].unique())
        else:
            seg_val = None

    horizon = st.slider("Forecast Horizon (months ahead)", 1, 3, 3)

    if seg_type == "Overall":
        series_data = df.groupby('Order Date')['Sales'].sum().resample('ME').sum()
        title_suffix = "Overall Sales"
    elif seg_type == "Category":
        series_data = df[df['Category'] == seg_val].groupby(
            'Order Date')['Sales'].sum().resample('ME').sum()
        title_suffix = f"Category: {seg_val}"
    else:
        series_data = df[df['Region'] == seg_val].groupby(
            'Order Date')['Sales'].sum().resample('ME').sum()
        title_suffix = f"Region: {seg_val}"

    with st.spinner("🔧 Training Prophet model..."):
        prophet_input = pd.DataFrame({'ds': series_data.index, 'y': series_data.values})
        train_p = prophet_input[:-horizon]
        test_p  = prophet_input[-horizon:]

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, changepoint_prior_scale=0.15)
        model.fit(train_p)
        future = model.make_future_dataframe(periods=horizon + 3, freq='ME')
        forecast = model.predict(future)

        pred_test = forecast[forecast['ds'].isin(test_p['ds'])]['yhat'].values
        mae_v  = float(np.mean(np.abs(test_p['y'].values - pred_test)))
        rmse_v = float(np.sqrt(np.mean((test_p['y'].values - pred_test)**2)))

    # KPI row
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("📉 MAE",  f"${mae_v:,.0f}",  help="Mean Absolute Error on held-out test months")
    kc2.metric("📉 RMSE", f"${rmse_v:,.0f}", help="Root Mean Squared Error")
    next_month_fc = forecast.tail(horizon)['yhat'].iloc[0]
    kc3.metric("🔮 Next Month Forecast", f"${next_month_fc:,.0f}")

    # Forecast chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=prophet_input['ds'], y=prophet_input['y'],
        name='Actual Sales', line=dict(color='#6366f1', width=2.5)
    ))
    fc_full = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_fc = fc_full[fc_full['ds'] > prophet_input['ds'].max()]
    fig_fc.add_trace(go.Scatter(
        x=future_fc['ds'], y=future_fc['yhat'].clip(lower=0),
        name='Prophet Forecast', line=dict(color='#f59e0b', width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=10, color='#f59e0b')
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([future_fc['ds'], future_fc['ds'][::-1]]),
        y=pd.concat([future_fc['yhat_upper'],
                     future_fc['yhat_lower'].clip(lower=0)[::-1]]),
        fill='toself', fillcolor='rgba(245,158,11,0.12)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'
    ))
    fig_fc.update_layout(**PLOTLY_DARK, title=f'Prophet Forecast — {title_suffix}',
                         height=420, yaxis_tickprefix='$', yaxis_tickformat=',.0f')
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    st.markdown("#### 📋 Forecast Details")
    fc_tbl = future_fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    fc_tbl.columns = ['Date', 'Forecast ($)', 'Lower Bound ($)', 'Upper Bound ($)']
    fc_tbl['Date'] = fc_tbl['Date'].dt.strftime('%B %Y')
    for c in ['Forecast ($)', 'Lower Bound ($)', 'Upper Bound ($)']:
        fc_tbl[c] = fc_tbl[c].apply(lambda x: f'${max(x, 0):,.0f}')
    st.dataframe(fc_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════
elif "Anomaly" in page:
    st.markdown('<div class="section-title">🚨 Anomaly Report</div>', unsafe_allow_html=True)

    # Recompute anomalies inline if CSV not found
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    daily  = df.groupby('Order Date')['Sales'].sum()
    weekly = daily.resample('W').sum().reset_index()
    weekly.columns = ['Date', 'Sales']
    weekly['week_num'] = range(len(weekly))

    sc = StandardScaler()
    X  = sc.fit_transform(weekly[['Sales', 'week_num']])
    iso = IsolationForest(contamination=0.07, random_state=42, n_estimators=200)
    weekly['IF_label'] = iso.fit_predict(X)
    weekly['IF_anomaly'] = weekly['IF_label'] == -1

    window = 8
    weekly['roll_mean'] = weekly['Sales'].rolling(window, center=True).mean()
    weekly['roll_std']  = weekly['Sales'].rolling(window, center=True).std()
    weekly['z_score']   = (weekly['Sales'] - weekly['roll_mean']) / weekly['roll_std']
    weekly['ZS_anomaly'] = weekly['z_score'].abs() > 2.0

    weekly['Method'] = weekly.apply(
        lambda r: 'Both' if r['IF_anomaly'] and r['ZS_anomaly']
                  else ('Isolation Forest' if r['IF_anomaly']
                        else ('Z-Score' if r['ZS_anomaly'] else 'Normal')), axis=1)

    # KPI row
    n_if = weekly['IF_anomaly'].sum()
    n_zs = weekly['ZS_anomaly'].sum()
    n_both = ((weekly['IF_anomaly']) & (weekly['ZS_anomaly'])).sum()
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("🔴 Isolation Forest", f"{n_if} anomalies")
    ac2.metric("🟠 Z-Score Method",   f"{n_zs} anomalies")
    ac3.metric("🚩 Agreed By Both",   f"{n_both} anomalies")

    # Chart
    color_map = {'Both': '#f43f5e', 'Isolation Forest': '#f59e0b',
                 'Z-Score': '#a78bfa', 'Normal': '#6366f1'}
    fig_a = go.Figure()
    for method, color in color_map.items():
        mask = weekly['Method'] == method
        if method == 'Normal':
            fig_a.add_trace(go.Scatter(
                x=weekly.loc[mask, 'Date'], y=weekly.loc[mask, 'Sales'],
                mode='lines', name='Normal Sales',
                line=dict(color=color, width=1.5)
            ))
        else:
            fig_a.add_trace(go.Scatter(
                x=weekly.loc[mask, 'Date'], y=weekly.loc[mask, 'Sales'],
                mode='markers', name=f'Anomaly ({method})',
                marker=dict(color=color, size=12, symbol='triangle-up',
                            line=dict(color='white', width=1))
            ))
    fig_a.update_layout(**PLOTLY_DARK,
                        title='Weekly Sales Anomalies — Isolation Forest vs Z-Score',
                        height=420, yaxis_tickprefix='$', yaxis_tickformat=',.0f')
    st.plotly_chart(fig_a, use_container_width=True)

    # Anomaly Table
    st.markdown("#### 📋 Detected Anomaly Dates & Sales Values")
    anom_tbl = weekly[weekly['Method'] != 'Normal'][
        ['Date', 'Sales', 'z_score', 'Method']].copy()
    anom_tbl = anom_tbl.sort_values('Date')
    anom_tbl['Date']    = anom_tbl['Date'].dt.strftime('%d %b %Y')
    anom_tbl['Sales']   = anom_tbl['Sales'].apply(lambda x: f'${x:,.0f}')
    anom_tbl['z_score'] = anom_tbl['z_score'].apply(
        lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
    anom_tbl.columns = ['Date', 'Weekly Sales', 'Z-Score', 'Detection Method']
    st.dataframe(anom_tbl, use_container_width=True, hide_index=True)

    # Business context
    st.markdown("#### 💡 Business Context")
    st.info("""
    **High-Sales Anomalies** (Nov–Dec): Driven by Black Friday, Cyber Monday, and holiday 
    corporate purchasing. Recommend pre-positioning inventory by October.

    **Low-Sales Anomalies** (Jan–Feb): Post-holiday spending decline. 
    Use this window for inventory optimisation and supplier renegotiations.
    """)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ══════════════════════════════════════════════════════════════
elif "Segment" in page:
    st.markdown('<div class="section-title">🧩 Product Demand Segments</div>', unsafe_allow_html=True)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Feature engineering
    sub = df.groupby('Sub-Category').agg(
        total_sales=('Sales', 'sum'),
        avg_order_value=('Sales', 'mean'),
        order_count=('Sales', 'count'),
    ).reset_index()

    yr = df.groupby(['Sub-Category', df['Order Date'].dt.year])['Sales'].sum().unstack(fill_value=0)
    if 2016 in yr.columns and 2017 in yr.columns:
        growth = ((yr[2017] - yr[2016]) / yr[2016].replace(0, np.nan) * 100).reset_index()
        growth.columns = ['Sub-Category', 'growth_rate']
    else:
        growth = pd.DataFrame({'Sub-Category': sub['Sub-Category'], 'growth_rate': 0.0})

    vol = df.groupby(['Sub-Category', df['Order Date'].dt.to_period('M')])['Sales'].sum()
    vol = vol.groupby(level=0).std().reset_index()
    vol.columns = ['Sub-Category', 'volatility']

    feat = sub.merge(growth, on='Sub-Category').merge(vol, on='Sub-Category')
    feat.fillna(0, inplace=True)

    feature_cols = ['total_sales', 'growth_rate', 'volatility', 'avg_order_value']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat[feature_cols])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    feat['Cluster'] = kmeans.fit_predict(X_scaled)

    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
    labels_map = {}
    for i, row in centers.iterrows():
        ts = row['total_sales'];  gr = row['growth_rate']; vl = row['volatility']
        if ts > centers['total_sales'].median() and gr > 5:
            labels_map[i] = 'High Volume, Growing'
        elif ts > centers['total_sales'].median():
            labels_map[i] = 'High Volume, Stable'
        elif vl > centers['volatility'].median():
            labels_map[i] = 'Low Volume, High Volatility'
        else:
            labels_map[i] = 'Low Volume, Declining'

    feat['Cluster_Label'] = feat['Cluster'].map(labels_map)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    feat['PCA1'] = X_pca[:, 0]; feat['PCA2'] = X_pca[:, 1]

    COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981']

    # KPI
    sk1, sk2, sk3 = st.columns(3)
    sk1.metric("🏷️ Sub-Categories Analysed", len(feat))
    sk2.metric("🔵 Clusters Found", 4)
    best_cluster = feat.groupby('Cluster_Label')['growth_rate'].mean().idxmax()
    sk3.metric("🚀 Highest Growth Cluster", best_cluster)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        fig_c = px.scatter(
            feat, x='PCA1', y='PCA2', color='Cluster_Label',
            text='Sub-Category', size='total_sales',
            title='Product Demand Clusters (PCA 2D)',
            color_discrete_sequence=COLORS,
            labels={'PCA1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)',
                    'PCA2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)'},
            size_max=50
        )
        fig_c.update_traces(textposition='top center', textfont_size=9)
        fig_c.update_layout(**PLOTLY_DARK, height=480)
        st.plotly_chart(fig_c, use_container_width=True)

    with col_r:
        st.markdown("#### 📊 Elbow Curve")
        inertia = []
        for k in range(2, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(2, 9)), y=inertia,
            mode='lines+markers', line=dict(color='#6366f1', width=2.5),
            marker=dict(size=10, color='#6366f1')
        ))
        fig_elbow.add_vline(x=4, line_dash='dash', line_color='#f43f5e',
                            annotation_text='K=4', annotation_font_color='#f43f5e')
        fig_elbow.update_layout(**PLOTLY_DARK, title='Elbow Method', height=220,
                                xaxis_title='K', yaxis_title='Inertia')
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.markdown("#### 💡 Stocking Strategy")
        strategies = {
            'High Volume, Growing':      '📈 Increase buffer 20–30%',
            'High Volume, Stable':       '✅ Maintain & optimise reorder',
            'Low Volume, High Volatility': '⚡ Safety stock + weekly monitor',
            'Low Volume, Declining':     '📉 Reduce inventory, phase out',
        }
        for cluster, strategy in strategies.items():
            st.markdown(f"""
            <div class="highlight-box">
                <b style="color:#6366f1">{cluster}</b><br>
                <span style="color:#94a3b8">{strategy}</span>
            </div>
            """, unsafe_allow_html=True)

    # Sub-category table
    st.markdown("#### 📋 Sub-Category Cluster Assignments")
    tbl = feat[['Sub-Category', 'Cluster_Label', 'total_sales',
                'growth_rate', 'avg_order_value']].sort_values('Cluster_Label')
    tbl['total_sales']    = tbl['total_sales'].apply(lambda x: f'${x:,.0f}')
    tbl['growth_rate']    = tbl['growth_rate'].apply(lambda x: f'{x:+.1f}%')
    tbl['avg_order_value'] = tbl['avg_order_value'].apply(lambda x: f'${x:,.0f}')
    tbl.columns = ['Sub-Category', 'Demand Cluster', 'Total Sales', 'YoY Growth', 'Avg Order']
    st.dataframe(tbl, use_container_width=True, hide_index=True)
