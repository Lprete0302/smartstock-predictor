import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="SmartStock Predictor", layout="wide")
st.title("📈 SmartStock Predictor")
st.markdown("Get stock prices and a 7-day prediction!")

# Ticker input
ticker = st.text_input("Enter a stock symbol (e.g. AAPL, TSLA, MSFT):", "AAPL").upper()

# Fetch data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    data = data.dropna()
    return data

data = load_data(ticker)

# Display data
if not data.empty:
    st.subheader(f"{ticker} Closing Price (Last 6 Months)")
    st.line_chart(data['Close'])

    # Prediction
    st.subheader(f"{ticker} 7-Day Forecast (Linear Regression)")

    df = data.reset_index()
    df['Day'] = range(len(df))
    X = df[['Day']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_days = 7
    future_X = pd.DataFrame({'Day': range(len(df), len(df) + future_days)})
    future_preds = model.predict(future_X)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df['Date'], y, label='Closing Price')
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    plt.plot(future_dates, future_preds, 'r--', label='Predicted Price (Next 7 Days)')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{ticker} Stock Price & Forecast")
    plt.grid()
    st.pyplot(plt)
else:
    st.warning("No data found for this ticker.")
