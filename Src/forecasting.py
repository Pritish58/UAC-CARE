import pandas as pd
import numpy as np
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "Data"
    / "Processed"
    / "uac_cleaned.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "Models"
    / "sarima_forecasting_model.pkl"
)

FORECAST_FILE = (
    BASE_DIR
    / "reports"
    / "sarima_forecast.csv"
)

RESULT_FILE = (
    BASE_DIR
    / "reports"
    / "forecast_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")


# ============================================================
# CREATE SYSTEM LOAD
# ============================================================

df["Total System Load"] = (
    df["Children in CBP custody"]
    + df["Children in HHS Care"]
)


# ============================================================
# CREATE REGULAR DAILY TIME SERIES
# ============================================================

ts = (
    df[
        [
            "Date",
            "Total System Load"
        ]
    ]
    .set_index("Date")
    .resample("D")
    .mean()
)


# Fill missing calendar dates using interpolation
ts["Total System Load"] = (
    ts["Total System Load"]
    .interpolate(method="linear")
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("UAC CARE ANALYTICS - TIME SERIES FORECASTING")
print("=" * 65)

print("\nTime-series observations:", len(ts))

print(
    "Date range:",
    ts.index.min().date(),
    "to",
    ts.index.max().date()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(ts) * 0.80
)

train = ts.iloc[:split_index]

test = ts.iloc[split_index:]


print("\nTraining observations:", len(train))
print("Testing observations :", len(test))


# ============================================================
# SARIMA MODEL
# ============================================================

print("\nTraining SARIMA model...")

model = SARIMAX(
    train["Total System Load"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_model = model.fit(
    disp=False
)


# ============================================================
# FORECAST TEST PERIOD
# ============================================================

forecast = fitted_model.forecast(
    steps=len(test)
)


forecast = pd.Series(
    forecast,
    index=test.index
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    test["Total System Load"],
    forecast
)

rmse = np.sqrt(
    mean_squared_error(
        test["Total System Load"],
        forecast
    )
)


# ============================================================
# SAVE TEST FORECAST
# ============================================================

forecast_df = pd.DataFrame({
    "Date": test.index,
    "Actual Load": test["Total System Load"].values,
    "Forecast Load": forecast.values
})

forecast_df["Forecast Error"] = (
    forecast_df["Actual Load"]
    - forecast_df["Forecast Load"]
)


forecast_df.to_csv(
    FORECAST_FILE,
    index=False
)


# ============================================================
# FUTURE FORECAST
# ============================================================

print("\nGenerating future 30-day forecast...")

future_forecast = fitted_model.forecast(
    steps=30
)


future_dates = pd.date_range(
    start=ts.index.max() + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)


future_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast Load": future_forecast.values
})


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    fitted_model,
    MODEL_FILE
)


# ============================================================
# SAVE METRICS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE"
    ],
    "Value": [
        round(mae, 4),
        round(rmse, 4)
    ]
})


results.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SARIMA FORECASTING RESULTS")
print("=" * 65)

print(f"\nMAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


print("\nNext 30-day forecast:")

print(
    future_df.to_string(
        index=False
    )
)


print("\n" + "=" * 65)
print("TIME SERIES FORECASTING COMPLETED")
print("=" * 65)

print(f"\nModel   : {MODEL_FILE}")
print(f"Forecast: {FORECAST_FILE}")
print(f"Results : {RESULT_FILE}")