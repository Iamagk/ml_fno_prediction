import streamlit as st
import requests
import xgboost as xgb
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import os
import time
from datetime import datetime, timedelta
from scipy.stats import norm
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained XGBoost model
model_path = "models/fno_xgboost_model.json"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()
model = xgb.XGBClassifier()
model.load_model(model_path)

# Define expected features
FEATURE_NAMES = [
    "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
    "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
    "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
    "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
]

def get_next_expiry():
    today = datetime.today()
    weekday = today.weekday()  # Monday = 0, Sunday = 6
    if weekday >= 3:
        days_until_next_expiry = (3 - weekday) + 7
    else:
        days_until_next_expiry = 3 - weekday
    next_expiry = today + timedelta(days=days_until_next_expiry)
    return next_expiry.strftime("%Y-%m-%d")

def fetch_stock_data(symbol):
    if symbol.upper() == "^NSEI":
        possible_symbols = [symbol]
    else:
        possible_symbols = [f"{symbol}.NS", f"{symbol}.BO", symbol]
    for ticker in possible_symbols:
        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period="10y")
                stock_data = stock_data.dropna(subset=['Close'])
                if not stock_data.empty:
                    # Calculate additional features
                    stock_data['ATR'] = calculate_atr(stock_data, period=14)
                    stock_data['SMA_5'] = calculate_sma(stock_data, window=5)
                    stock_data['SMA_10'] = calculate_sma(stock_data, window=10)
                    stock_data['RSI_14'] = calculate_rsi(stock_data, period=14)
                    stock_data['MACD'], stock_data['MACD_Signal'] = calculate_macd(stock_data)
                    stock_data['MACD_Hist'] = calculate_macd_hist(stock_data)
                    stock_data['STOCH_K'], stock_data['STOCH_D'] = calculate_stochastic(stock_data)
                    stock_data['ROC_10'] = calculate_roc(stock_data)
                    stock_data['OBV'] = calculate_obv(stock_data)
                    stock_data['VWAP'] = calculate_vwap(stock_data)
                    stock_data['ADX'] = calculate_adx(stock_data)
                    stock_data['CCI'] = calculate_cci(stock_data)
                    stock_data['WILLR_14'] = calculate_willr(stock_data)
                    stock_data['MOM_10'] = calculate_momentum(stock_data)
                    stock_data['CMF'] = calculate_cmf(stock_data)
                    stock_data['PSAR'] = calculate_psar(stock_data)
                    stock_data['Aroon_Up'], stock_data['Aroon_Down'] = calculate_aroon(stock_data)
                    stock_data['Return'] = stock_data['Close'].pct_change() * 100
                    latest_data = stock_data.iloc[-1]
                    live_data = {col: latest_data.get(col, 0) for col in FEATURE_NAMES}
                    return pd.DataFrame([live_data])
                time.sleep(2 ** attempt)
            except Exception as e:
                time.sleep(2 ** attempt)
    return None

def get_model_confidence(model, df):
    probabilities = model.predict_proba(df)
    confidence = max(probabilities[0]) * 100
    return round(confidence, 2)

def get_nearest_strike_price(symbol, current_price):
    try:
        ticker = yf.Ticker(symbol)
        expiry_dates = ticker.options
        if not expiry_dates:
            return None
        nearest_expiry = expiry_dates[0]
        options_chain = ticker.option_chain(nearest_expiry)
        available_strikes = sorted(set(options_chain.calls['strike'].tolist() + options_chain.puts['strike'].tolist()))
        nearest_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        return nearest_strike
    except Exception:
        return None

# --- Feature calculation functions (same as your backend, omitted for brevity) ---
# Please copy all your calculate_* functions here (ATR, SMA, RSI, MACD, etc.)

# ... (Insert all calculate_* functions here, unchanged) ...

# For brevity, only one is shown:
def calculate_atr(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# (Add all other calculate_* functions here...)

def estimate_option_price(symbol, current_price, strike_price, expiry_date, option_type):
    try:
        stock_data = fetch_stock_data(symbol)
        if stock_data is None or stock_data.empty:
            return None
        sigma = calculate_historical_volatility(stock_data)
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            return None
        today = datetime.today()
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        T = (expiry - today).days / 365.0
        if T <= 0 or np.isnan(T):
            return None
        if current_price <= 0 or strike_price <= 0 or np.isnan(current_price) or np.isnan(strike_price):
            return None
        r = 0.05
        d1 = (math.log(current_price / strike_price) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            option_price = (current_price * norm.cdf(d1)) - (strike_price * math.exp(-r * T) * norm.cdf(d2))
        elif option_type == "put":
            option_price = (strike_price * math.exp(-r * T) * norm.cdf(-d2)) - (current_price * norm.cdf(-d1))
        else:
            return None
        return round(option_price, 2)
    except Exception:
        return None

def calculate_historical_volatility(stock_data, period=30, fallback_volatility=0.2):
    try:
        if len(stock_data) < period:
            return fallback_volatility
        stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        volatility = stock_data['Log_Returns'].rolling(window=period).std() * np.sqrt(252)
        if volatility.iloc[-1] <= 0 or np.isnan(volatility.iloc[-1]):
            return fallback_volatility
        return volatility.iloc[-1]
    except Exception:
        return fallback_volatility

# --- Streamlit UI ---
st.title("ML FNO Prediction App")

symbol = st.text_input("Enter Stock Symbol (e.g., TCS, RELIANCE, ^NSEI):")

if st.button("Predict with Options"):
    if symbol:
        with st.spinner("Fetching prediction..."):
            live_data = fetch_stock_data(symbol)
            if live_data is None or live_data.empty:
                st.error("No data available for the symbol.")
            else:
                feature_data = {col: float(live_data[col].values[0]) for col in FEATURE_NAMES if col in live_data}
                df = pd.DataFrame([feature_data])
                prediction = model.predict(df)[0]
                current_price = feature_data.get('Close', 0)
                confidence = get_model_confidence(model, df)
                strike_price = get_nearest_strike_price(symbol, current_price) or current_price
                if prediction == 1:
                    strike_price = math.ceil(strike_price / 50) * 50
                    option_type = "call"
                else:
                    strike_price = math.floor(strike_price / 50) * 50
                    option_type = "put"
                atr = feature_data.get('ATR', 0)
                support_level = feature_data.get('Support', strike_price - atr)
                resistance_level = feature_data.get('Resistance', strike_price + atr)
                if prediction == 1:
                    stop_loss_strike = support_level
                    exit_price_strike = resistance_level
                else:
                    stop_loss_strike = resistance_level
                    exit_price_strike = support_level
                stop_loss_strike = round(float(stop_loss_strike), 2)
                exit_price_strike = round(float(exit_price_strike), 2)
                expiry_date = get_next_expiry()
                option_price = estimate_option_price(symbol, current_price, strike_price, expiry_date, option_type)
                if option_price is None or np.isnan(option_price):
                    stop_loss_option = None
                    exit_price_option = None
                else:
                    if prediction == 1:
                        stop_loss_option = option_price - max((current_price - support_level), 0)
                        exit_price_option = option_price + max((resistance_level - current_price), 0)
                    else:
                        stop_loss_option = option_price - max((resistance_level - current_price), 0)
                        exit_price_option = option_price + max((current_price - support_level), 0)
                    if stop_loss_option <= 0:
                        stop_loss_option = option_price * 0.8
                    stop_loss_option = round(float(stop_loss_option), 2)
                    exit_price_option = round(float(exit_price_option), 2)
                confidence = round(float(confidence), 3)
                response = {
                    "prediction": int(prediction),
                    "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
                    "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
                    "stop_loss_strike": stop_loss_strike,
                    "exit_price_strike": exit_price_strike,
                    "stop_loss_option": stop_loss_option,
                    "exit_price_option": exit_price_option,
                    "expiry": expiry_date,
                    "confidence": confidence,
                    "option_price": option_price
                }
                st.success("Prediction fetched!")
                st.json(response)
    else:
        st.warning("Please enter a stock symbol.")