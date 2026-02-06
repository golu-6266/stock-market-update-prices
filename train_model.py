import yfinance as yf
import numpy as np
import pandas as pd
from indicators import add_indicators
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# Download Data
df = yf.download("TCS.NS", period="5y")
df = add_indicators(df)

features = df[['Close','SMA','EMA','RSI','BB_High','BB_Low']]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)

# LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=32)

# Save model & scaler
os.makedirs("model", exist_ok=True)
os.makedirs("scaler", exist_ok=True)

model.save("model/lstm_model.h5")
joblib.dump(scaler, "scaler/scaler.pkl")

print("✅ Model training completed & saved")
