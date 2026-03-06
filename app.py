# app.py

import streamlit as st
import pandas as pd
import joblib
import os
from src.data_loader import load_data
from datetime import date, timedelta

st.set_page_config(
    page_title="Demand Forecasting",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium Dark Theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #080a12;
    color: #dde1f0;
}
.block-container {
    padding-top: 0 !important;
    max-width: 1060px;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* ── Page Header ── */
.page-header {
    text-align: center;
    padding: 3.5rem 0 2.8rem;
    border-bottom: 1px solid #141726;
    margin-bottom: 0;
}
.page-eyebrow {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 0.8rem;
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin: 0 0 0.6rem;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #44495f;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.gold-rule {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, #c9a84c 0%, transparent 100%);
    margin: 1.2rem auto 0;
    border: none;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #141726;
    padding: 0;
    margin-bottom: 2.4rem;
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    padding: 0 20px;
    color: #44495f;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    background: transparent;
    transition: color 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #8b90a8;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom-color: #c9a84c !important;
    font-weight: 700;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Section Typography ── */
.section-eyebrow {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 0.35rem;
}
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 0.35rem;
    line-height: 1.2;
}
.section-desc {
    font-size: 0.8rem;
    color: #44495f;
    margin-bottom: 1.8rem;
    line-height: 1.65;
    max-width: 560px;
}

/* ── Metric Row ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 1.4rem;
}
.metric-box {
    flex: 1;
    background: #0d0f1c;
    border: 1px solid #141726;
    border-radius: 10px;
    padding: 1.3rem 1.2rem;
    text-align: center;
}
.metric-lbl {
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #44495f;
    margin-bottom: 0.55rem;
}
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #c9a84c;
    line-height: 1;
    letter-spacing: -1px;
}
.metric-unit {
    font-size: 0.68rem;
    color: #44495f;
    margin-top: 0.35rem;
}

/* ── Alerts ── */
.alert {
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.83rem;
    font-weight: 500;
    line-height: 1.55;
    margin-top: 1.1rem;
}
.alert-ok   { background: rgba(76,175,129,0.06); border: 1px solid rgba(76,175,129,0.18); color: #4caf81; }
.alert-warn { background: rgba(232,149,109,0.06); border: 1px solid rgba(232,149,109,0.18); color: #e8956d; }
.alert-info { background: rgba(74,158,255,0.06); border: 1px solid rgba(74,158,255,0.18); color: #4a9eff; }
.alert-err  { background: rgba(210,72,72,0.06); border: 1px solid rgba(210,72,72,0.18); color: #d24848; }

/* ── Form Controls ── */
div[data-baseweb="select"] > div {
    background-color: #0d0f1c !important;
    border: 1px solid #1a1d2e !important;
    border-radius: 8px !important;
    color: #dde1f0 !important;
    font-size: 0.85rem !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
}
label,
.stSelectbox label,
.stSlider label,
.stRadio label,
.stDateInput label {
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.2px !important;
    color: #44495f !important;
    text-transform: uppercase !important;
}
.stRadio [data-baseweb="radio"] label {
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    color: #8b90a8 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c 0%, #a8882e 100%);
    color: #080a12;
    border: none;
    border-radius: 7px;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 0.58rem 2rem;
    height: auto;
    transition: all 0.18s ease;
    box-shadow: 0 2px 12px rgba(201,168,76,0.15);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(201,168,76,0.28);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Slider ── */
.stSlider [role="slider"] {
    background: #c9a84c !important;
    box-shadow: 0 0 0 4px rgba(201,168,76,0.15) !important;
}
[data-baseweb="slider"] div[data-testid="stThumbValue"] {
    color: #c9a84c !important;
}

/* ── Date Input ── */
input[type="date"] {
    background: #0d0f1c !important;
    border: 1px solid #1a1d2e !important;
    border-radius: 8px !important;
    color: #dde1f0 !important;
    font-size: 0.85rem !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid #141726 !important;
    border-radius: 10px !important;
    overflow: hidden;
}
.stDataFrame th {
    background: #0d0f1c !important;
    color: #c9a84c !important;
    font-size: 0.65rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 700;
}
.stDataFrame td {
    font-size: 0.82rem !important;
    color: #8b90a8 !important;
}

/* ── Spinners ── */
.stSpinner > div > div {
    border-top-color: #c9a84c !important;
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data("data/retail_warehouse_inventory_dataset.csv")

df = get_data()
store_ids   = sorted(df['Store ID'].unique())
product_ids = sorted(df['Product ID'].unique())

def load_model(store, product):
    path = f"models/prophet_models/{store}_{product}_prophet.joblib"
    return joblib.load(path) if os.path.exists(path) else None

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-eyebrow">Retail Intelligence Platform</div>
    <h1 class="page-title">Demand Forecasting</h1>
    <p class="page-subtitle">Prophet-powered inventory intelligence across stores and products</p>
    <div class="gold-rule"></div>
</div>
""", unsafe_allow_html=True)

# ── Navigation Tabs ───────────────────────────────────────────────────────────
tabs = st.tabs([
    "Demand Forecast",
    "Trend Detection",
    "Inventory Planning",
    "Date Forecast",
    "Trend Analysis",
    "Product Comparison",
    "Restocking Alerts",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1 — Demand Forecast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[0]:
    st.markdown("""
        <div class="section-eyebrow">Forecasting</div>
        <h2 class="section-heading">Predicted Units Sold</h2>
        <p class="section-desc">Select a store and product to generate a demand forecast over the next 7 or 30 days.</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        store1   = st.selectbox("Store", store_ids, key="store1")
    with col2:
        product1 = st.selectbox("Product", product_ids, key="product1")
    with col3:
        period   = st.radio("Horizon", ["7 Days", "30 Days"], horizontal=True)

    if st.button("Run Forecast", key="forecast1"):
        model = load_model(store1, product1)
        if model:
            days     = 7 if period == "7 Days" else 30
            future   = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            chart_df = forecast[['ds', 'yhat']].tail(days).set_index('ds')
            st.line_chart(chart_df, color=["#c9a84c"])
            total = int(forecast.tail(days)['yhat'].sum())
            avg   = round(forecast.tail(days)['yhat'].mean(), 1)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-lbl">Total Forecast ({days}d)</div>
                    <div class="metric-val">{total:,}</div>
                    <div class="metric-unit">units</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">Daily Average</div>
                    <div class="metric-val">{avg}</div>
                    <div class="metric-unit">units / day</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-err">No trained model found for this store-product combination.</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2 — Trend Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[1]:
    st.markdown("""
        <div class="section-eyebrow">Analysis</div>
        <h2 class="section-heading">Demand Rise & Drop Detection</h2>
        <p class="section-desc">Determine whether demand is trending upward or downward over a custom forecast horizon.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        store2   = st.selectbox("Store", store_ids, key="store2")
    with col2:
        product2 = st.selectbox("Product", product_ids, key="product2")

    days2 = st.slider("Forecast horizon (days)", 7, 60, 30)

    if st.button("Detect Trend", key="trend2"):
        model = load_model(store2, product2)
        if model:
            future   = model.make_future_dataframe(periods=days2)
            forecast = model.predict(future)
            recent   = forecast.tail(days2)
            change   = recent['yhat'].diff().mean()
            st.line_chart(recent[['ds', 'yhat']].set_index('ds'), color=["#4a9eff"])
            if change > 0:
                st.markdown(f'<div class="alert alert-ok">Demand is expected to <strong>rise</strong> by an average of <strong>{change:.2f} units/day</strong> over the next {days2} days.</div>', unsafe_allow_html=True)
            elif change < 0:
                st.markdown(f'<div class="alert alert-warn">Demand is expected to <strong>fall</strong> by an average of <strong>{abs(change):.2f} units/day</strong> over the next {days2} days.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert alert-info">Demand is expected to remain <strong>stable</strong> over the next {days2} days.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-err">No trained model found for this store-product combination.</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 3 — Inventory Planning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[2]:
    st.markdown("""
        <div class="section-eyebrow">Planning</div>
        <h2 class="section-heading">Inventory Stock Recommendation</h2>
        <p class="section-desc">Calculate the recommended stock level required to meet predicted demand over the next 30 days.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        store3   = st.selectbox("Store", store_ids, key="store3")
    with col2:
        product3 = st.selectbox("Product", product_ids, key="product3")

    if st.button("Calculate Recommendation", key="stock3"):
        model = load_model(store3, product3)
        if model:
            future   = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            tail30   = forecast.tail(30)
            total    = int(tail30['yhat'].sum())
            peak     = int(tail30['yhat'].max())
            avg      = round(tail30['yhat'].mean(), 1)
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-lbl">Recommended Stock</div>
                    <div class="metric-val">{total:,}</div>
                    <div class="metric-unit">units for 30 days</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">Peak Day Demand</div>
                    <div class="metric-val">{peak:,}</div>
                    <div class="metric-unit">units</div>
                </div>
                <div class="metric-box">
                    <div class="metric-lbl">Avg Daily Demand</div>
                    <div class="metric-val">{avg}</div>
                    <div class="metric-unit">units / day</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-err">No trained model found for this store-product combination.</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 4 — Date Forecast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[3]:
    st.markdown("""
        <div class="section-eyebrow">Precision</div>
        <h2 class="section-heading">Single-Date Forecast</h2>
        <p class="section-desc">Retrieve the predicted sales figure for any specific date within a 4-year forecast window.</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        store4   = st.selectbox("Store", store_ids, key="store4")
    with col2:
        product4 = st.selectbox("Product", product_ids, key="product4")
    with col3:
        today      = date.today()
        date_input = st.date_input("Target Date", min_value=today, max_value=today + timedelta(days=1460))

    if st.button("Get Forecast", key="date4"):
        model = load_model(store4, product4)
        if model:
            future   = model.make_future_dataframe(periods=1460)
            forecast = model.predict(future)
            result   = forecast[forecast['ds'] == pd.to_datetime(date_input)]
            if not result.empty:
                units = result['yhat'].values[0]
                low   = result['yhat_lower'].values[0]
                high  = result['yhat_upper'].values[0]
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-lbl">Forecast — {date_input}</div>
                        <div class="metric-val">{units:.0f}</div>
                        <div class="metric-unit">units expected</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">Lower Bound</div>
                        <div class="metric-val">{low:.0f}</div>
                        <div class="metric-unit">units (pessimistic)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">Upper Bound</div>
                        <div class="metric-val">{high:.0f}</div>
                        <div class="metric-unit">units (optimistic)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-warn">Selected date is outside the forecast range.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-err">No trained model found for this store-product combination.</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 5 — Trend Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[4]:
    st.markdown("""
        <div class="section-eyebrow">Intelligence</div>
        <h2 class="section-heading">Portfolio Trend Analysis</h2>
        <p class="section-desc">Scan every store-product pair to surface the strongest rising and falling demand trends across your portfolio.</p>
    """, unsafe_allow_html=True)

    if st.button("Run Analysis", key="trend5"):
        trend_summary = []
        with st.spinner("Scanning store-product models..."):
            for store in store_ids:
                for product in product_ids:
                    model = load_model(store, product)
                    if model:
                        future   = model.make_future_dataframe(periods=30)
                        forecast = model.predict(future)
                        diff     = forecast['yhat'].diff().tail(30).mean()
                        trend_summary.append((store, product, round(diff, 3)))

        if trend_summary:
            trend_df = pd.DataFrame(trend_summary, columns=["Store", "Product", "Avg Daily Change"])
            st.dataframe(
                trend_df.sort_values("Avg Daily Change", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.markdown('<div class="alert alert-warn">No models available for analysis.</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 6 — Product Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[5]:
    st.markdown("""
        <div class="section-eyebrow">Comparison</div>
        <h2 class="section-heading">Product Forecast Comparison</h2>
        <p class="section-desc">Compare the 30-day demand forecast for two products within the same store side by side.</p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        store6   = st.selectbox("Store", store_ids, key="store6")
    with col2:
        productA = st.selectbox("Product A", product_ids, key="productA")
    with col3:
        productB = st.selectbox("Product B", product_ids, key="productB")

    if st.button("Compare", key="compare6"):
        with st.spinner("Generating comparison..."):
            modelA = load_model(store6, productA)
            modelB = load_model(store6, productB)
            if modelA and modelB:
                fa = modelA.predict(modelA.make_future_dataframe(periods=30)).tail(30)[['ds', 'yhat']].set_index('ds')
                fb = modelB.predict(modelB.make_future_dataframe(periods=30)).tail(30)[['ds', 'yhat']].set_index('ds')
                combined = pd.concat([
                    fa.rename(columns={'yhat': productA}),
                    fb.rename(columns={'yhat': productB}),
                ], axis=1)
                st.line_chart(combined)
                avg_a = round(fa['yhat'].mean(), 1)
                avg_b = round(fb['yhat'].mean(), 1)
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-lbl">{productA} — Avg Daily</div>
                        <div class="metric-val">{avg_a}</div>
                        <div class="metric-unit">units / day</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-lbl">{productB} — Avg Daily</div>
                        <div class="metric-val">{avg_b}</div>
                        <div class="metric-unit">units / day</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-err">One or both models were not found.</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 7 — Restocking Alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tabs[6]:
    st.markdown("""
        <div class="section-eyebrow">Alerts</div>
        <h2 class="section-heading">Restocking Requirements</h2>
        <p class="section-desc">Identify every store-product pair where predicted demand over the next 7 days falls below your defined threshold.</p>
    """, unsafe_allow_html=True)

    threshold = st.slider("Demand threshold (units)", 0, 100, 50)

    if st.button("Scan Inventory", key="restock7"):
        low_stock = []
        with st.spinner("Analyzing all store-product pairs..."):
            for store in store_ids:
                for product in product_ids:
                    model = load_model(store, product)
                    if model:
                        future   = model.make_future_dataframe(periods=7)
                        forecast = model.predict(future)
                        avg_pred = forecast.tail(7)['yhat'].mean()
                        if avg_pred < threshold:
                            low_stock.append((store, product, round(avg_pred, 1)))

        if low_stock:
            restock_df = pd.DataFrame(low_stock, columns=["Store", "Product", "Avg Predicted Units (7d)"])
            restock_df = restock_df.sort_values("Avg Predicted Units (7d)").reset_index(drop=True)
            st.markdown(f'<div class="alert alert-warn">{len(low_stock)} store-product pair(s) require restocking attention.</div>', unsafe_allow_html=True)
            st.dataframe(restock_df, use_container_width=True)
        else:
            st.markdown('<div class="alert alert-ok">All store-product pairs are above the defined threshold. No restocking required.</div>', unsafe_allow_html=True)
