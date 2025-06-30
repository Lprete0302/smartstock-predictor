import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

def fetch_stock_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d")
    return data

def predict_next_7_days(data):
    data = data.reset_index()
    data['Date_ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    
    # Prepare training data
    X = data['Date_ordinal'].values.reshape(-1, 1)
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict for next 7 days
    last_date = data['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_ordinals)

    return future_dates, predictions

def plot_stock_with_predictions(data, ticker, future_dates, predictions):
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label=f'{ticker} Closing Price')
    plt.plot(future_dates, predictions, label='Predicted Price (Next 7 Days)', linestyle='--')
    plt.title(f'{ticker} Stock Price & 7-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter stock symbol (e.g. AAPL, TSLA): ").upper()
    df = fetch_stock_data(ticker)
    if not df.empty:
        future_dates, predictions = predict_next_7_days(df)
        plot_stock_with_predictions(df, ticker, future_dates, predictions)
    else:
        print("No data found for this ticker.")
