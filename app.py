import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from indicators import add_indicators
import altair as alt
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Pro Stock Dashboard v3.0", layout="wide")
st.title("📈  Stock Market  Analysis")

# ----------------------------
# Auto refresh every 60 seconds
# ----------------------------
st_autorefresh(interval=60 * 1000, key="datarefresh")

# ----------------------------
# Nifty 50 sample companies
# ----------------------------
nifty_50 = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "Larsen & Toubro": "LT.NS",
    "HCL Tech": "HCLTECH.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS"
}

# ----------------------------
# Load LSTM model & scaler
# ----------------------------
model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("scaler/scaler.pkl")

# ----------------------------
# Select company
# ----------------------------
company_name = st.selectbox("Select Company", list(nifty_50.keys()))
symbol = nifty_50[company_name]

# ----------------------------
# Load historical data with caching
# ----------------------------
@st.cache_data(show_spinner=True)
def load_data(symbol):
    df = yf.download(symbol, period="10y", interval="1d")
    df.reset_index(inplace=True)
    return df

df = load_data(symbol)

# ----------------------------
# Today's price
# ----------------------------
stock = yf.Ticker(symbol)
today_data = stock.history(period="1d")
today_price = today_data['Close'][0]
st.info(f"📌 Today's Price of {company_name}: ₹ {today_price:.2f}")

# ----------------------------
# Add technical indicators
# ----------------------------
df = add_indicators(df)

# ----------------------------
# LSTM Prediction
# ----------------------------
features = df[['Close','SMA','EMA','RSI','BB_High','BB_Low']].copy()
scaled = scaler.transform(features)
X = np.array([scaled[-60:]])  # last 60 days
pred = model.predict(X)
pred_price = scaler.inverse_transform(np.hstack((pred, np.zeros((1, features.shape[1]-1)))))[0][0]

rsi = df['RSI'].iloc[-1]
st.success(f"📊 Predicted Next Close Price: ₹ {pred_price:.2f}")

if rsi < 30:
    st.info("🟢 BUY Signal (Oversold)")
elif rsi > 70:
    st.warning("🔴 SELL Signal (Overbought)")
else:
    st.write("🟡 HOLD")

# ----------------------------
# Day-by-day Table with Date
# ----------------------------
st.subheader("📅 Day-by-Day Stock Result")

min_date = df['Date'].min()
max_date = df['Date'].max()
start_date, end_date = st.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
df_day = df.loc[mask, ['Date','Close']].copy()
df_day['Change'] = df_day['Close'].diff()
df_day['Trend'] = np.where(df_day['Change'] > 0,'Up','Down')

st.dataframe(df_day)

# Download CSV button
csv = df_day.to_csv(index=False).encode()
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"{company_name}_day_by_day.csv",
    mime='text/csv'
)

# ----------------------------
# Closing price line chart
# ----------------------------
st.subheader("📉 Closing Price Trend (10 Years)")

for col in ['Open','High','Low','Close']:
    df[col] = df[col].astype(float)

df['Trend'] = np.where(df['Close'].diff() > 0,'Up','Down')

line_chart = alt.Chart(df).mark_line().encode(
    x='Date:T',
    y='Close:Q',
    color=alt.Color('Trend:N', scale=alt.Scale(domain=['Up','Down'], range=['green','red']))
).properties(title=f"Closing Price Trend for {company_name}")

st.altair_chart(line_chart, use_container_width=True)

# ----------------------------
# Candlestick chart
# ----------------------------
st.subheader("🕯️ Candlestick View")

candlestick = alt.Chart(df).mark_bar().encode(
    x='Date:T',
    y='Open:Q',
    y2='Close:Q',
    color=alt.condition(
        "datum.Close > datum.Open",
        alt.value("green"),
        alt.value("red")
    )
)

st.altair_chart(candlestick, use_container_width=True)
