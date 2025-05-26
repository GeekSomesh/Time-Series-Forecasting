from prophet import Prophet


def fit_prophet(ts):
    model = Prophet(
        yearly_seasonality=True,  # Capture annual patterns
        weekly_seasonality=True,  # Weekly seasonality
        daily_seasonality=False,  # Usually not useful for retail monthly data
    )

    model.fit(ts)
    return model
