import pandas as pd

def prepare_time_series(df, store_id, product_id, freq='D'):
    # Filter and aggregate
    ts = (
        df[(df['Store ID'] == store_id) & (df['Product ID'] == product_id)]
        .groupby('Date')['Units Sold'].sum()
        .resample(freq).sum()
        .reset_index()
        .rename(columns={'Date': 'ds', 'Units Sold': 'y'})
    )
    # Ensure continuous date range
    ts = ts.set_index('ds').asfreq(freq, fill_value=0).reset_index()
    return ts
