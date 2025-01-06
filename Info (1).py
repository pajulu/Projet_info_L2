# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Fetching the S&P 500 data
sp500_ticker = yf.Ticker("NKE")
sp500 = sp500_ticker.history(start="2024-10-29")

# Save the fetched data to a CSV file (optional)
sp500.to_csv("sp500.csv")

# Function to compute the Simple Moving Average (SMA)
def compute_SMA(data, window):
    return data.rolling(window=window).mean()

# Function to compute the Exponential Moving Average (EMA)
def compute_EMA(data, span):
    return data.ewm(span=span, adjust=False).mean()

# Function to compute the Relative Strength Index (RSI)
def compute_RSI(data, window):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()

    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute the MACD and Signal Line
def compute_MACD(data, span1, span2, signal):
    ema1 = compute_EMA(data, span1)
    ema2 = compute_EMA(data, span2)
    macd = ema1 - ema2
    signal_line = compute_EMA(macd, signal)
    return macd, signal_line

# Function to process the data and compute technical indicators
def process_data(data):
    del data["Dividends"]
    del data["Stock Splits"]
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data["SMA_30"] = compute_SMA(data["Close"], 30)
    data["EMA_30"] = compute_EMA(data["Close"], 30)
    data["RSI_14"] = compute_RSI(data["Close"], 14)
    data["MACD"], data["Signal_Line"] = compute_MACD(data["Close"], 12, 26, 9)
    data = data.dropna()  # Drop rows with NaN values after adding indicators
    return data

# Process the data
sp500.index = pd.to_datetime(sp500.index)
sp500_processed = process_data(sp500)

# Define predictors
predictors = ['Close', 'Volume', 'Open', 'High', 'Low', 'SMA_30', 'EMA_30', 'RSI_14', 'MACD', 'Signal_Line']

# Split the data into training and testing sets
X = sp500_processed[predictors]
y = sp500_processed["Target"]

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, min_samples_split=2, max_depth=10, random_state=1)
model.fit(X, y)

# Predict the next day's target
latest_data = sp500_processed.tail(1).copy()
tomorrow_features = latest_data[predictors]
tomorrow_prediction = model.predict(tomorrow_features)
prediction_proba = model.predict_proba(tomorrow_features)

# Print the prediction and confidence score
prediction_message = " Le prix serai plus élevé." if int(tomorrow_prediction[0]) == 1 else "Le prix serai plus bas."
print(f"Prediction: {prediction_message}")
print(f"Date De la prediction: {latest_data.index[-1].strftime('%Y-%m-%d')}")