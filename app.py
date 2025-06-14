import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.title('Stock Price Trend Prediction with LSTM Model :bar_chart:')
st.write("This application predicts stock prices using a pre-trained LSTM model. You can select a stock ticker and view the predicted stock prices along with moving averages and RSI indicators.")

model = load_model('LSTM_model.keras')

ticker = st.selectbox("Select a stock:", ['NVDA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT'])

start_date = '2010-01-01'
end_date = '2024-12-31'

data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("No data found for the specified ticker and date range.")  

else:
    st.write(f"Data for {ticker} from {start_date} to {end_date}:")
    st.dataframe(data)

    stock_prices = data['Close'].values.reshape(-1, 1)
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(stock_prices)
    scaled_prices = scaled_prices.reshape(-1, 1)

    def create_sequences(data, days):
        X, y = [], []
        for i in range(len(data) - days):
            X.append(data[i:i + days])
            y.append(data[i + days])
        return np.array(X), np.array(y)


    # Assuming the number of days we base our predictions on is 60
    days = 60
    X, y = create_sequences(scaled_prices, days)

    #Splitting the data into train-test sets
    training_size = int(len(X) * 0.8)
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    # Make predictions
    prediction_prices = model.predict(X_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    y_test_price = scaler.inverse_transform(y_test)

    test_data_index = data.index[-len(y_test):]

    st.subheader(f"Predicted stock prices for {ticker} using the LSTM model:", divider=True)
    st.write("The following plot shows the actual stock prices and the predicted stock prices.")
    st.write("The model was trained on historical stock data and is now making predictions based on the last 60 days of data.")

    # Plot the results
    plt.figure(figsize=(13, 7))
    plt.plot(test_data_index, y_test_price, label='Actual NVIDIA Stock Price')
    plt.plot(test_data_index, prediction_prices, label='Predicted NVIDIA Stock Price')
    plt.title('NVIDIA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    st.pyplot(plt)

    st.subheader(f"Moving Averages for {ticker} Stock Prices:", divider=True)

    # Moving Averages
    SMA_50 = data['Close'].rolling(window=50).mean()
    SMA_200 = data['Close'].rolling(window=200).mean()
    st.write("The following plot shows the 50-day and 200-day Simple Moving Averages (SMA) of the stock prices along with the predicted prices.")
    st.write("Moving averages help smooth out price data by creating a constantly updated average price. They are used to identify the direction of the trend.")

    plt.figure(figsize=(13, 7))
    plt.plot(data.index, data['Close'], label=f'{ticker} Stock Price', color='blue', alpha=0.7)
    plt.plot(test_data_index, prediction_prices, label=f'{ticker} Predicted Price', color='green', )
    plt.plot(SMA_50.index, SMA_50, label='50-Day SMA', color='pink')
    plt.plot(SMA_200.index, SMA_200, label='200-Day SMA', color='red')
    plt.title(f'Stock Closing Prices with Moving Averages and Predictions for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    # RSI
    st.subheader(f"Relative Strength Index (RSI) for {ticker} Stock Prices:", divider=True)
    window = st.selectbox("Select RSI window size:", [7, 14, 21], index=1)

    st.write("""
    **Default (e.g., 14):**
    This is the most common setting and is often considered a good balance between sensitivity to price changes and generating reliable signals.

    **Shorter periods (e.g., 7):**
    These are used for faster signals, often in intraday or swing trading, as they react more quickly to recent price fluctuations.

    **Longer periods (e.g., 21):**
    These are used to reduce noise and focus on larger trends, making the RSI less susceptible to minor price swings.
    """)
    
    def calculate_rsi(data, window=13):
        diff = data['Close'].diff()
        loss = -diff.where(diff < 0, 0)
        gain = diff.where(diff > 0, 0)
        avg_loss = loss.rolling(window=window).mean()
        avg_gain = gain.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Plotting the RSI
    RSI = calculate_rsi(data, window)
    plt.figure(figsize=(13, 5))
    plt.plot(RSI.index, RSI, label = 'RSI', color='magenta')
    plt.axhline(70, label = 'Overbought (70)', linestyle='--', color = 'red')
    plt.axhline(30, label='Oversold (30)', linestyle='--', color = 'green')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()
    plt.show()
    st.pyplot(plt)

    st.write("The above plot shows the RSI, which indicates overbought or oversold conditions in the stock market. An RSI above 70 typically indicates that a stock is overbought, while an RSI below 30 indicates that it is oversold.")
    

