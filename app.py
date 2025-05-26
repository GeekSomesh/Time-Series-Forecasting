# app.py

import streamlit as st
import pandas as pd
import joblib
import os
from src.data_loader import load_data
from src.preprocessing import prepare_time_series
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demand Forecasting App", layout="wide")
st.title("📈 Time-Series Demand Forecasting")

# Load dataset
data_path = "data/retail_warehouse_inventory_dataset.csv"
df = load_data(data_path)

# Get unique IDs
store_ids = sorted(df['Store ID'].unique())
product_ids = sorted(df['Product ID'].unique())

# Helper: load model
def load_model(store, product):
    fname = f"models/prophet_models/{store}_{product}_prophet.joblib"
    if os.path.exists(fname):
        return joblib.load(fname)
    return None

# 1. Predict units sold next week/month
st.subheader("📦 1. Predict units sold (Next Week/Month)")
store1 = st.selectbox("Select Store ID", store_ids, key="store1")
product1 = st.selectbox("Select Product ID", product_ids, key="product1")
period = st.radio("Forecast for:", ["Next 7 days", "Next 30 days"], horizontal=True)

if st.button("Predict", key="forecast1"):
    ts = prepare_time_series(df, store1, product1)
    model = load_model(store1, product1)
    if model:
        future = model.make_future_dataframe(periods=7 if period == "Next 7 days" else 30)
        forecast = model.predict(future)
        st.line_chart(forecast[['ds', 'yhat']].set_index('ds').tail(30))
        st.success("Forecast completed.")
    else:
        st.error("Model not found.")

# 2. When will demand drop/rise?
st.subheader("📉 2. Detect Demand Rise/Drop")
store2 = st.selectbox("Store ID", store_ids, key="store2")
product2 = st.selectbox("Product ID", product_ids, key="product2")
days = st.slider("Check for how many days ahead?", 7, 60, 30)

if st.button("Detect Trend", key="trend2"):
    model = load_model(store2, product2)
    if model:
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        recent = forecast.tail(days)
        change = recent['yhat'].diff().mean()
        if change > 0:
            st.info(f"Demand is expected to RISE on average by {change:.2f} units/day")
        elif change < 0:
            st.warning(f"Demand is expected to DROP on average by {abs(change):.2f} units/day")
        else:
            st.info("Demand is stable")
    else:
        st.error("Model not found.")

# 3. Inventory stock recommendation for next month
st.subheader("📦 3. Inventory Stock Recommendation (Next Month)")
store3 = st.selectbox("Store ID", store_ids, key="store3")
product3 = st.selectbox("Product ID", product_ids, key="product3")

if st.button("Recommend Stock", key="stock3"):
    model = load_model(store3, product3)
    if model:
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        total_units = forecast.tail(30)['yhat'].sum()
        st.success(f"Recommended inventory for next month: {int(total_units)} units")
    else:
        st.error("Model not found.")

# 4. Forecast sales for a specific date
from datetime import date, timedelta

st.subheader("📅 4. Forecast for a Specific Date")
store4 = st.selectbox("Store ID", store_ids, key="store4")
product4 = st.selectbox("Product ID", product_ids, key="product4")

today = date.today()
max_date = today + timedelta(days=1460)
date_input = st.date_input("Select a future date", min_value=today, max_value=max_date)

if st.button("Forecast Date", key="date4"):
    model = load_model(store4, product4)
    if model:
        future = model.make_future_dataframe(periods=1460)
        forecast = model.predict(future)
        result = forecast[forecast['ds'] == pd.to_datetime(date_input)]
        if not result.empty:
            units = result['yhat'].values[0]
            st.success(f"Expected units sold on {date_input}: {units:.0f} units")
        else:
            st.warning("Selected date is out of forecast range.")
    else:
        st.error("Model not found.")

# 5. Rising or Falling Demand Trend
st.subheader("📊 5. Identify Rising/Falling Demand Trends")

if st.button("Run Trend Analysis"):
    trend_summary = []
    with st.spinner("Analyzing trends across store-product pairs..."):
        for store in store_ids:
            for product in product_ids:
                model = load_model(store, product)
                if model:
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    diff = forecast['yhat'].diff().tail(30).mean()
                    trend_summary.append((store, product, diff))

    if trend_summary:
        trend_df = pd.DataFrame(trend_summary, columns=["Store ID", "Product ID", "Avg Change"])
        st.dataframe(trend_df.sort_values("Avg Change", ascending=False).reset_index(drop=True))
    else:
        st.warning("No trends found.")

# 6. Compare Forecasts of Two Products
st.subheader("🔍 6. Compare Product Forecasts (Same Store)")
store6 = st.selectbox("Store ID", store_ids, key="store6")
productA = st.selectbox("Product A", product_ids, key="productA")
productB = st.selectbox("Product B", product_ids, key="productB")

if st.button("Compare Forecasts"):
    with st.spinner("Comparing product forecasts..."):
        modelA = load_model(store6, productA)
        modelB = load_model(store6, productB)
        if modelA and modelB:
            future = modelA.make_future_dataframe(periods=30)
            fa = modelA.predict(future).tail(30)[['ds', 'yhat']].set_index('ds')
            fb = modelB.predict(future).tail(30)[['ds', 'yhat']].set_index('ds')
            combined = pd.concat([fa.rename(columns={'yhat': f'Product {productA}'}),
                                  fb.rename(columns={'yhat': f'Product {productB}'})], axis=1)
            st.line_chart(combined)
        else:
            st.error("One or both models not found.")

# 7. Stores that need restocking
st.subheader("🚨 7. Stores Needing Restocking Soon")

threshold = st.slider("Sales below which value? (Last 7 days avg)", 0, 100, 50)

if st.button("Check Restocking Needs"):
    low_stock = []

    with st.spinner("Analyzing inventory needs across all store-product pairs..."):
        for store in store_ids:
            for product in product_ids:
                model = load_model(store, product)
                if model:
                    future = model.make_future_dataframe(periods=7)
                    forecast = model.predict(future)
                    avg_pred = forecast.tail(7)['yhat'].mean()
                    if avg_pred < threshold:
                        low_stock.append((store, product, int(avg_pred)))

    if low_stock:
        st.success("✅ Restocking needed for some store-product pairs.")
        st.dataframe(pd.DataFrame(
            low_stock,
            columns=["Store ID", "Product ID", "Avg Predicted Units (Next 7 Days)"]
        ))
    else:
        st.info("👍 No store-product pairs below the threshold.")
