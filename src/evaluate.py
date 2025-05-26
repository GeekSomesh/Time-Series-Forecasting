import os
import joblib
import matplotlib.pyplot as plt
from pandas import date_range
from src.data_loader import load_data
from src.preprocessing import prepare_time_series


def evaluate_models(data_path, models_dir):
    df = load_data(data_path)
    for fname in os.listdir(models_dir):
        if not fname.endswith('.joblib'): continue
        store, product, _ = fname.split('_')
        model = joblib.load(os.path.join(models_dir, fname))
        ts = prepare_time_series(df, store, product)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Plot
        plt.figure(figsize=(10, 4))
        model.plot(forecast)
        plt.title(f"Forecast for {store}-{product}")
        plt.show()