# 📈 Stock Price Prediction with LSTM and Technical Indicators

This project predicts future stock prices using historical data and technical indicators such as
Simple Moving Average (SMA) and Relative Strength Index (RSI). It uses an LSTM (Long Short-Term Memory)
model built with Keras and optionally visualizes insights through a Streamlit dashboard.

---

# 🚀 Features
- Fetch historical stock data using `yfinance`
- Build and train LSTM model to predict future stock prices
- Visualize predictions vs actual values
- Calculate technical indicators (SMA, RSI)
- Deploy a dashboard using Streamlit

---

# 🛠️ Tools & Libraries Used
- Python
- Pandas, NumPy — Data manipulation
- yfinance — Fetching historical stock data
- Keras (TensorFlow backend) — LSTM model
- Matplotlib — Data visualization
- Technical analysis indicators (SMA, RSI)
- Streamlit — Dashboard deployment

---


# ▶️ Usage

Run the LSTM predictor:
    python main.ipynb

(Optional) Run the Streamlit dashboard:
    streamlit run app.py

---

# 📌 Notes
- Model uses a window size of 60 days to predict the next day's close.
- SMA and RSI indicators help incorporate trend and momentum.
- Can be extended to support multiple tickers, timeframes, or other indicators.

---

# Streamlit APP
- https://hm-spp-lstm.streamlit.app/

---
