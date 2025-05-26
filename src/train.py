import os
import joblib
from data_loader import load_data
from preprocessing import prepare_time_series
from forecasting import fit_prophet


def train_models(data_path):
    df = load_data(data_path)
    combos = df[['Store ID', 'Product ID']].drop_duplicates()
    for _, row in combos.iterrows():
        store, product = row['Store ID'], row['Product ID']
        ts = prepare_time_series(df, store, product)
        model = fit_prophet(ts)
        fname = f"{store}_{product}_prophet.joblib"
        joblib.dump(model, os.path.join('models/prophet_models', fname))
        print(f"Trained Prophet for {store}-{product}")