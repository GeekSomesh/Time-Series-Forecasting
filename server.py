from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from prophet import Prophet
from datetime import date
from src.data_loader import load_data
from src.preprocessing import prepare_time_series

app = FastAPI(title="Demand Forecasting API")

# Load dataset
data_path = "data/retail_warehouse_inventory_dataset.csv"
df = load_data(data_path)

# Store/Product IDs
store_ids = sorted(df['Store ID'].unique())
product_ids = sorted(df['Product ID'].unique())

# Model loader
def load_model(store, product):
    model_path = f"models/prophet_models/{store}_{product}_prophet.joblib"
    return joblib.load(model_path) if os.path.exists(model_path) else None

# Request schemas
class ForecastRequest(BaseModel):
    store_id: str
    product_id: str
    period: int

class DateForecastRequest(BaseModel):
    store_id: str
    product_id: str
    forecast_date: date

class CompareRequest(BaseModel):
    store_id: str
    productA: str
    productB: str

# 1. Predict units sold for next week/month
@app.post("/predict-units/")
def predict_units(data: ForecastRequest):
    model = load_model(data.store_id, data.product_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    future = model.make_future_dataframe(periods=data.period)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(data.period).to_dict(orient="records")

# 2. Detect rising/falling demand
@app.post("/detect-trend/")
def detect_trend(data: ForecastRequest):
    model = load_model(data.store_id, data.product_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    future = model.make_future_dataframe(periods=data.period)
    forecast = model.predict(future)
    change = forecast.tail(data.period)['yhat'].diff().mean()
    trend = "rise" if change > 0 else "drop" if change < 0 else "stable"
    return {"trend": trend, "avg_change": round(change, 2)}

# 3. Recommend inventory stock for next month
@app.post("/recommend-stock/")
def recommend_stock(data: ForecastRequest):
    model = load_model(data.store_id, data.product_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    future = model.make_future_dataframe(periods=data.period)
    forecast = model.predict(future)
    total_units = forecast.tail(data.period)['yhat'].sum()
    return {"recommended_inventory": int(total_units)}

# 4. Forecast sales for a specific date
@app.post("/forecast-date/")
def forecast_specific_date(data: DateForecastRequest):
    model = load_model(data.store_id, data.product_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    future = model.make_future_dataframe(periods=1460)
    forecast = model.predict(future)
    result = forecast[forecast['ds'] == pd.to_datetime(data.forecast_date)]
    if result.empty:
        raise HTTPException(status_code=404, detail="Date out of forecast range.")
    return {"date": str(data.forecast_date), "predicted_units": round(result['yhat'].values[0], 2)}

# 5. Trend analysis for all store-product pairs
@app.get("/trend-analysis/")
def trend_analysis():
    results = []
    for store in store_ids:
        for product in product_ids:
            model = load_model(store, product)
            if model:
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                change = forecast['yhat'].diff().tail(30).mean()
                results.append({"store_id": store, "product_id": product, "avg_change": round(change, 2)})
    return sorted(results, key=lambda x: x['avg_change'], reverse=True)

# 6. Compare forecasts of two products (same store)
@app.post("/compare-products/")
def compare_products(data: CompareRequest):
    modelA = load_model(data.store_id, data.productA)
    modelB = load_model(data.store_id, data.productB)
    if not modelA or not modelB:
        raise HTTPException(status_code=404, detail="One or both models not found.")
    future = modelA.make_future_dataframe(periods=30)
    fa = modelA.predict(future).tail(30)[['ds', 'yhat']].rename(columns={"yhat": f"Product_{data.productA}"})
    fb = modelB.predict(future).tail(30)[['ds', 'yhat']].rename(columns={"yhat": f"Product_{data.productB}"})
    comparison = pd.merge(fa, fb, on='ds')
    return comparison.to_dict(orient="records")

# 7. Find store-product pairs needing restocking
@app.get("/restocking-needed/")
def restocking_needed(threshold: int = 50):
    low_stock = []
    for store in store_ids:
        for product in product_ids:
            model = load_model(store, product)
            if model:
                future = model.make_future_dataframe(periods=7)
                forecast = model.predict(future)
                avg = forecast.tail(7)['yhat'].mean()
                if avg < threshold:
                    low_stock.append({
                        "store_id": store,
                        "product_id": product,
                        "avg_predicted_units": round(avg, 2)
                    })
    return low_stock
