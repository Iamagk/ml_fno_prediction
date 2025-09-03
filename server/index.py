from fastapi import FastAPI
import xgboost as xgb
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import time
from datetime import datetime, timedelta  # Import datetime
from zoneinfo import ZoneInfo
from scipy.stats import norm
import math
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
import requests

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://*.vercel.app",   # Vercel deployments
        "https://vercel.app",     # Vercel domain
        "*"  # Allow all origins for production (adjust as needed)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_next_expiry(holidays=None, tz: str = "Asia/Kolkata"):
    """
    Next weekly expiry: Tuesday (IST). If Tuesday is a holiday, move to Monday.
    """
    today_local = datetime.now(ZoneInfo(tz)).date()
    weekday = today_local.weekday()  # Mon=0, Tue=1, ... Sun=6
    target_weekday = 1  # Tuesday
    days_ahead = (target_weekday - weekday) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_expiry = today_local + timedelta(days=days_ahead)
    if holidays:
        iso = next_expiry.strftime("%Y-%m-%d")
        if iso in holidays:
            next_expiry = next_expiry - timedelta(days=1)  # Monday
    return next_expiry.strftime("%Y-%m-%d")

@app.get("/")
def home():
    return {"message": "API is running!"}

# Load trained XGBoost model
model_path = "models/fno_xgboost_model.json"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = xgb.XGBClassifier()
model.load_model(model_path)

# Define expected features
FEATURE_NAMES = [
    "STRIKE_PR", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI",
    "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_SIGNAL", "EMA_9", "EMA_21", "EMA_50", "EMA_200",
    "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "MACD_HIST", "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV",
    "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10", "CMF", "PSAR", "AROON_UP", "AROON_DOWN", "RETURN"
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to fetch stock data with retries
def fetch_stock_data(symbol):
    # Handle NIFTY symbol mapping
    if symbol.upper() in ["NIFTY", "NIFTY50", "NIFTY 50"]:
        possible_symbols = ["^NSEI"]  # Direct NIFTY 50 symbol
    elif symbol.upper() == "^NSEI":
        possible_symbols = [symbol]  # Already correct
    else:
        possible_symbols = [f"{symbol}.NS", f"{symbol}.BO", symbol]  # NSE, BSE, and direct symbol

    for ticker in possible_symbols:
        for attempt in range(3):  # Retry up to 3 times
            try:
                logger.info(f"Fetching data for {ticker} (Attempt {attempt+1})")
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period="1y")  # Reduced from 10y to 1y for better performance
                stock_data = stock_data.dropna(subset=['Close'])  # Ensure 'Close' column is not empty
                # After fetching stock_data
                stock_data.columns = [col.upper() for col in stock_data.columns]

                if not stock_data.empty:
                    logger.info(f"Fetched data for {ticker}:\n{stock_data.tail()}")

                    # Validate data
                    if stock_data.isnull().values.any():
                        logger.warning(f"Data contains NaN values for {ticker}: {stock_data.isnull().sum()}")
                        continue

                    # Calculate additional features
                    stock_data['ATR'] = calculate_atr(stock_data, period=14)
                    stock_data['SMA_5'] = calculate_sma(stock_data, window=5)
                    stock_data['SMA_10'] = calculate_sma(stock_data, window=10)
                    stock_data['RSI_14'] = calculate_rsi(stock_data, period=14)
                    stock_data['MACD'], stock_data['MACD_SIGNAL'] = calculate_macd(stock_data)
                    stock_data['MACD_HIST'] = calculate_macd_hist(stock_data)
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
                    stock_data['AROON_UP'], stock_data['AROON_DOWN'] = calculate_aroon(stock_data)
                    
                    # Add missing EMA calculations
                    stock_data['EMA_9'] = stock_data['CLOSE'].ewm(span=9).mean()
                    stock_data['EMA_21'] = stock_data['CLOSE'].ewm(span=21).mean()
                    stock_data['EMA_50'] = stock_data['CLOSE'].ewm(span=50).mean()
                    stock_data['EMA_200'] = stock_data['CLOSE'].ewm(span=200).mean()
                    
                    # Add Bollinger Bands
                    sma_20 = stock_data['CLOSE'].rolling(window=20).mean()
                    std_20 = stock_data['CLOSE'].rolling(window=20).std()
                    stock_data['BB_UPPER'] = sma_20 + (std_20 * 2)
                    stock_data['BB_MIDDLE'] = sma_20
                    stock_data['BB_LOWER'] = sma_20 - (std_20 * 2)
                    
                    stock_data['RETURN'] = stock_data['CLOSE'].pct_change() * 100  # Percentage change in closing price

                    # Extract the latest row of data
                    latest_data = stock_data.iloc[-1]
                    logger.info(f"Latest data for {ticker}: {latest_data}")

                    # Helper to get value or 0.0 if missing
                    def get_feature(name):
                        # Try exact, uppercase, and lowercase
                        for key in [name, name.upper(), name.lower()]:
                            if key in latest_data and not pd.isnull(latest_data[key]):
                                return float(latest_data[key])
                        # For missing trading-specific features, provide default values
                        if name in ['STRIKE_PR', 'SETTLE_PR']:
                            return float(latest_data.get('CLOSE', 0))  # Use close price as default
                        elif name in ['CONTRACTS', 'VAL_INLAKH', 'OPEN_INT', 'CHG_IN_OI']:
                            return 0.0  # These are futures/options specific
                        return 0.0

                    # Build live_data dict using FEATURE_NAMES
                    live_data = {feat: get_feature(feat) for feat in FEATURE_NAMES}

                    return pd.DataFrame([live_data])

                logger.warning(f"No data found for {ticker}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                logger.error(f"Error fetching {ticker} (Attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Wait before retrying

    logger.error(f"Failed to fetch stock data for {symbol} after multiple attempts.")
    return None

@app.get("/fetch_yfinance")
def fetch_yfinance(symbol: str):
    try:
        logger.info(f"Fetching Yahoo Finance data for: {symbol}")
        stock_data = fetch_stock_data(symbol)

        if stock_data is None or stock_data.empty:
            logger.warning(f"No stock data found for {symbol}")
            return {"error": f"No stock data found for {symbol}"}

        # Convert DataFrame to dictionary for JSON response
        response = stock_data.iloc[0].to_dict()

        # Validate response fields
        for key, value in response.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                logger.warning(f"Invalid value for {key}: {value}")
                response[key] = "N/A"  # Replace invalid values with "N/A"

        logger.info(f"Response sent to frontend: {response}")
        return response

    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data: {e}")
        return {"error": "Failed to fetch data"}
    
@app.get("/predict_live")
def predict_live(symbol: str):
    try:
        logger.info(f"Received request for symbol: {symbol}")

        # Fetch live stock data
        live_data = fetch_stock_data(symbol)

        if live_data is None or live_data.empty:
            logger.warning(f"No data available for symbol: {symbol}")
            return {
                "prediction": "No Data Available",
                "suggested_action": "N/A",
                "strike_price": "N/A",
                "stop_loss": "N/A",
                "expiry": "N/A",
                "confidence": "N/A"
            }

        logger.info(f"Live data fetched for {symbol}: {live_data}")

        # Ensure only the expected features are passed
        df = live_data[FEATURE_NAMES].copy()
        feature_data = df.iloc[0].to_dict()

        # Make prediction
        prediction = model.predict(df)[0]

        # Extract current price for calculations
        current_price = feature_data.get('CLOSE', 0)
        if current_price == 0:
            raise ValueError("Current price of the stock is unavailable.")

        # Get actual model confidence score
        confidence = get_model_confidence(model, df)

        # Fetch the real strike price from Yahoo Finance
        strike_price = get_nearest_strike_price(symbol, current_price) or current_price

        # Calculate stop loss using technical indicators
        atr = feature_data.get('ATR', 0)  # Average True Range
        support_level = feature_data.get('Support', strike_price - atr)
        resistance_level = feature_data.get('Resistance', strike_price + atr)

        if prediction == 1:
            stop_loss = int(support_level)  # Use support level as stop loss for a buy trade
        else:
            stop_loss = int(resistance_level)  # Use resistance level as stop loss for a sell trade

        # Fetch holidays dynamically
        api_key = "A1j9Nr72uN9scpcfcLmBJL2wGuOfVPXM"  # Replace with your Calendarific API key
        holidays = fetch_holidays(api_key, country="IN", year=datetime.now().year)
        logger.info(f"Fetched holidays: {holidays}")

        # Use dynamic expiry date
        expiry_date = get_next_expiry(holidays=holidays, tz="Asia/Kolkata")
        logger.info(f"Calculated expiry date: {expiry_date}")

        # Log prediction history
        update_trade_history(symbol, prediction, current_price)

        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
            "stop_loss": stop_loss,
            "expiry": expiry_date,
            "confidence": float(confidence)
        }

        logger.info(f"API Response: {response}")
        return response

    except Exception as e:
        logger.error(f"Error in API: {str(e)}")
        return {
            "prediction": "Error",
            "suggested_action": "N/A",
            "strike_price": "N/A",
            "stop_loss": "N/A",
            "expiry": "N/A",
            "confidence": "N/A"
        }


class StockInput(BaseModel):
    symbol: str
def get_nearest_strike_price(symbol, current_price):
    try:
        ticker = yf.Ticker(symbol)
        expiry_dates = ticker.options  # Get available expiry dates
        if not expiry_dates:
            logger.warning(f"No expiry dates available for {symbol}")
            return None

        nearest_expiry = expiry_dates[0]  # Choose the closest expiry
        options_chain = ticker.option_chain(nearest_expiry)

        # Combine call and put strikes, then find the nearest one
        available_strikes = sorted(set(options_chain.calls['strike'].tolist() + options_chain.puts['strike'].tolist()))
        logger.info(f"Available strikes for {symbol}: {available_strikes}")
        nearest_strike = min(available_strikes, key=lambda x: abs(x - current_price))

        logger.info(f"Nearest strike price for {symbol} at current price {current_price}: {nearest_strike}")
        return nearest_strike
    except Exception as e:
        logger.error(f"Error fetching strike price for {symbol}: {e}")
        return None
    

def get_model_confidence(model, df):
    """
    Get the actual confidence score from the model's prediction.
    """
    probabilities = model.predict_proba(df)  # Get probability for each class
    confidence = max(probabilities[0]) * 100  # Convert to percentage
    return round(confidence, 2)

def calculate_atr(data, period=14):
    high = data['HIGH']
    low = data['LOW']
    close = data['CLOSE']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_sma(data, window):
    return data['CLOSE'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    delta = data['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['CLOSE'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['CLOSE'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_macd_hist(data):
    macd, signal = calculate_macd(data)
    macd_hist = macd - signal
    return macd_hist

def calculate_stochastic(data, period=14):
    low_min = data['LOW'].rolling(window=period).min()
    high_max = data['HIGH'].rolling(window=period).max()
    stoch_k = 100 * (data['CLOSE'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()  # 3-period moving average of %K
    return stoch_k, stoch_d

def calculate_roc(data, period=10):
    roc = ((data['CLOSE'] - data['CLOSE'].shift(period)) / data['CLOSE'].shift(period)) * 100
    return roc

def calculate_obv(data):
    obv = (np.sign(data['CLOSE'].diff()) * data['VOLUME']).fillna(0).cumsum()
    return obv

def calculate_vwap(data):
    vwap = (data['CLOSE'] * data['VOLUME']).cumsum() / data['VOLUME'].cumsum()
    return vwap

def calculate_adx(data, period=14):
    high = data['HIGH']
    low = data['LOW']
    close = data['CLOSE']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr_smooth = true_range.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = abs(minus_dm.rolling(window=period).sum())

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

    adx = dx.rolling(window=period).mean()
    return adx

def calculate_cci(data, period=20):
    typical_price = (data['HIGH'] + data['LOW'] + data['CLOSE']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_willr(data, period=14):
    high_max = data['HIGH'].rolling(window=period).max()
    low_min = data['LOW'].rolling(window=period).min()
    willr = -100 * ((high_max - data['CLOSE']) / (high_max - low_min))
    return willr

def calculate_momentum(data, period=10):
    momentum = data['CLOSE'] - data['CLOSE'].shift(period)
    return momentum

def calculate_cmf(data, period=20):
    money_flow_multiplier = ((data['CLOSE'] - data['LOW']) - (data['HIGH'] - data['CLOSE'])) / (data['HIGH'] - data['LOW'])
    money_flow_volume = money_flow_multiplier * data['VOLUME']
    cmf = money_flow_volume.rolling(window=period).sum() / data['VOLUME'].rolling(window=period).sum()
    return cmf

def calculate_psar(data, step=0.02, max_step=0.2):
    high = data['HIGH']
    low = data['LOW']
    close = data['CLOSE']

    psar = close.copy()
    bull = True
    af = step
    ep = low.iloc[0]

    for i in range(1, len(close)):
        prev_psar = psar.iloc[i - 1]
        if bull:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                af = step
                ep = high.iloc[i]

        if bull:
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
        else:
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)

    return psar

def calculate_aroon(data, period=25):
    aroon_up = 100 * (period - data['HIGH'].rolling(window=period).apply(lambda x: period - np.argmax(x), raw=True)) / period
    aroon_down = 100 * (period - data['LOW'].rolling(window=period).apply(lambda x: period - np.argmin(x), raw=True)) / period
    return aroon_up, aroon_down

def calculate_historical_volatility(stock_data, period=30, fallback_volatility=0.2):
    """
    Calculate historical volatility using log returns.
    """
    try:
        # Ensure there is enough data to calculate rolling volatility
        if len(stock_data) < period:
            logger.warning(f"Not enough data to calculate historical volatility. Required: {period}, Available: {len(stock_data)}")
            logger.info(f"Using fallback volatility: {fallback_volatility}")
            return fallback_volatility

        # Calculate log returns
        stock_data['Log_Returns'] = np.log(stock_data['CLOSE'] / stock_data['CLOSE'].shift(1))
        logger.info(f"Log returns for volatility calculation:\n{stock_data['Log_Returns'].tail(period)}")

        # Calculate rolling standard deviation of log returns (annualized)
        volatility = stock_data['Log_Returns'].rolling(window=period).std() * np.sqrt(252)

        # Ensure volatility is valid
        if volatility.iloc[-1] <= 0 or np.isnan(volatility.iloc[-1]):
            logger.error(f"Calculated volatility is invalid: {volatility.iloc[-1]}")
            logger.info(f"Using fallback volatility: {fallback_volatility}")
            return fallback_volatility

        logger.info(f"Calculated historical volatility: {volatility.iloc[-1]}")
        return volatility.iloc[-1]  # Return the latest volatility value
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {e}")
        logger.info(f"Using fallback volatility: {fallback_volatility}")
        return fallback_volatility

@app.post("/predict")
def predict(stock: StockInput):
    return predict_live(stock.symbol)

trade_history = []  # Global list to track predictions

def update_trade_history(symbol, prediction, actual_price):
    """
    Store the model's predictions and compare them to actual market behavior.
    """
    trade_history.append({
        "symbol": symbol,
        "prediction": prediction,
        "actual_price": actual_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Keep only the last 100 trades to prevent memory issues
    if len(trade_history) > 100:
        trade_history.pop(0)



def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def get_individual_model_predictions(ensemble_model, df):
    """
    Returns:
      - by_name: { model_name: "Call"/"Put" }
      - details: [ { model, prediction, proba_call, proba_put } ]
    Uses scaler if present for linear models.
    """
    by_name = {}
    details = []

    # Determine estimators and names from common wrappers
    estimators = []
    names = []

    if hasattr(ensemble_model, "models") and ensemble_model.models:
        estimators = list(ensemble_model.models)
        names = getattr(ensemble_model, "model_names", None) or [m.__class__.__name__ for m in estimators]
    elif hasattr(ensemble_model, "estimators_") and ensemble_model.estimators_:
        names = [n for n, _ in ensemble_model.estimators_]
        estimators = [e for _, e in ensemble_model.estimators_]
    else:
        estimators = [ensemble_model]
        names = [getattr(ensemble_model, "__class__", type(ensemble_model)).__name__]

    scaler = getattr(ensemble_model, "scaler", None)

    for idx, est in enumerate(estimators):
        name = names[idx] if idx < len(names) else est.__class__.__name__
        X_in = df

        try:
            # Scale if estimator is linear/logistic and scaler exists
            needs_scale = any(k in name.lower() for k in ["logistic", "linear", "svm"])
            if needs_scale and scaler is not None:
                X_in = scaler.transform(df)

            # Predict class
            pred = est.predict(X_in)
            pred_val = int(pred[0]) if hasattr(pred, "__len__") else int(pred)

            # Predict probabilities if available, else approximate via decision_function
            proba_call = None
            proba_put = None

            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X_in)
                if proba is not None and np.ndim(proba) == 2 and proba.shape[1] >= 2:
                    proba_call = float(proba[0, 1])
                    proba_put = float(proba[0, 0])
            elif hasattr(est, "decision_function"):
                score = est.decision_function(X_in)
                score0 = float(score[0]) if hasattr(score, "__len__") else float(score)
                p1 = float(_sigmoid(score0))
                proba_call = p1
                proba_put = 1.0 - p1

            label = "Call" if pred_val == 1 else "Put"
            by_name[name] = label
            details.append(
                {
                    "model": name,
                    "prediction": label,
                    "proba_call": None if proba_call is None else round(proba_call, 4),
                    "proba_put": None if proba_put is None else round(proba_put, 4),
                }
            )
        except Exception as e:
            by_name[name] = f"error: {e}"
            details.append({"model": name, "error": str(e)})

    return {"by_name": by_name, "details": details}

@app.get("/predict_with_options")
def predict_with_options(symbol: str):
    try:
        logger.info(f"Received request for /predict_with_options with symbol: {symbol}")

        # Always define expiry_date at the start
        expiry_date = get_next_expiry()
        logger.info(f"Expiry date: {expiry_date}")

        # Fetch live stock data
        live_data = fetch_stock_data(symbol)

        if live_data is None or live_data.empty:
            logger.warning(f"No data available for symbol: {symbol}")
            return {"error": f"No data available for the symbol: {symbol}"}

        logger.info(f"Live data fetched for {symbol}: {live_data}")

        # Ensure only the expected features are passed and in correct order
        df = live_data[FEATURE_NAMES].copy()
        df = df.reindex(columns=FEATURE_NAMES)  # Ensure column order matches model
        df = df.astype(float)
        df = df.reset_index(drop=True)
        feature_data = df.iloc[0].to_dict()

        # Log feature data
        logger.info(f"Feature data for prediction: {df}")
        logger.info(f"API feature columns: {df.columns.tolist()}")

        # Check if model is loaded
        if model is None:
            logger.error("Ensemble model is not loaded. Cannot make predictions.")
            return {
                "error": "Model Not Loaded",
                "prediction": "Model Not Loaded",
                "suggested_action": "N/A",
                "strike_price": "N/A",
                "stop_loss_strike": "N/A",
                "exit_price_strike": "N/A",
                "stop_loss_option": "N/A",
                "exit_price_option": "N/A",
                "expiry": "N/A",
                "confidence": "N/A",
                "option_price": "N/A",
                "individual_model_predictions": {},
                "individual_model_details": [],
            }

        # Make ensemble prediction
        prediction = model.predict(df)[0]
        logger.info(f"Model prediction: {prediction}")

        # Individual model predictions (labels + probabilities)
        indiv = get_individual_model_predictions(model, df)
        individual_preds = indiv["by_name"]

        # Extract current price for calculations
        current_price = df['CLOSE'].iloc[0]
        if current_price == 0:
            raise ValueError("Current price of the stock is unavailable.")

        # Get actual model confidence score
        confidence = get_model_confidence(model, df)

        # Fetch the real strike price from Yahoo Finance
        strike_price = get_nearest_strike_price(symbol, current_price) or current_price
        # For call: round up to nearest 50, for put: round down to nearest 50
        if prediction == 1:
            strike_price = math.ceil(float(strike_price) / 50) * 50
        else:
            strike_price = math.floor(float(strike_price) / 50) * 50

        # Robust fallback for ATR
        atr = feature_data.get('ATR', 0)
        if atr is None or np.isnan(atr) or atr == 0:
            atr = max(0.01 * strike_price, 1.0)  # fallback: 1% of strike or 1

        # Robust fallback for support/resistance
        support_level = feature_data.get('Support', strike_price - atr) if 'Support' in feature_data else strike_price - atr
        resistance_level = feature_data.get('Resistance', strike_price + atr) if 'Resistance' in feature_data else strike_price + atr
        if support_level is None or np.isnan(support_level):
            support_level = strike_price - atr
        if resistance_level is None or np.isnan(resistance_level):
            resistance_level = strike_price + atr

        # Calculate stop loss and exit price for strike price
        if prediction == 1:  # Call option
            stop_loss_strike = round(float(support_level), 2)
            exit_price_strike = round(float(resistance_level), 2)
        else:  # Put option
            stop_loss_strike = round(float(resistance_level), 2)
            exit_price_strike = round(float(support_level), 2)

        # Estimate option price using Black-Scholes model
        option_type = "call" if prediction == 1 else "put"
        option_price = estimate_option_price(symbol, current_price, strike_price, expiry_date, option_type)

        # Calculate stop loss and exit price for options price
        if option_price is not None and not np.isnan(option_price):
            if prediction == 1:  # Call option
                stop_loss_option = round(float(option_price - max((current_price - support_level), 0)), 2)
                exit_price_option = round(float(option_price + max((resistance_level - current_price), 0)), 2)
            else:  # Put option
                stop_loss_option = round(float(option_price - max((resistance_level - current_price), 0)), 2)
                exit_price_option = round(float(option_price + max((current_price - support_level), 0)), 2)
            if stop_loss_option <= 0 or np.isnan(stop_loss_option):
                stop_loss_option = round(float(option_price * 0.8), 2)
        else:
            stop_loss_option = round(float(strike_price * 0.8), 2)
            exit_price_option = round(float(strike_price * 1.2), 2)

        # Round confidence to 3 decimal places
        confidence = round(float(confidence), 3)

        # Use dynamic expiry date (already Tuesday via get_next_expiry)
        expiry_date = get_next_expiry()
        logger.info(f"Expiry date: {expiry_date}")

        # Format strike price for display
        strike_price_fmt = f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE"

        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": strike_price_fmt,
            "option_price": option_price,
            "stop_loss_option": stop_loss_option,
            "exit_price_option": exit_price_option,
            "stop_loss_strike": stop_loss_strike,
            "exit_price_strike": exit_price_strike,
            "expiry": expiry_date,
            "confidence": confidence,
            "individual_model_predictions": individual_preds,
            "individual_model_details": indiv["details"],
        }

        logger.info(f"API Response: {response}")
        return response

    except Exception as e:
        logger.error(f"Error in /predict_with_options: {str(e)}")
        return {"error": "Failed to fetch prediction"}

def estimate_option_price(symbol, current_price, strike_price, expiry_date, option_type):
    """
    Estimate the option price using the Black-Scholes model.
    """
    try:
        # Fetch stock data for volatility calculation
        stock_data = fetch_stock_data(symbol)
        if stock_data is None or stock_data.empty:
            logger.error("Failed to fetch stock data for option price estimation")
            return None

        # Calculate historical volatility
        sigma = calculate_historical_volatility(stock_data)
        if sigma is None or sigma <= 0 or np.isnan(sigma):
            logger.error(f"Invalid volatility (sigma): {sigma}")
            return None

        # Calculate time to expiry in years
        today = datetime.today()
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        T = (expiry - today).days / 365.0
        if T <= 0 or np.isnan(T):
            logger.error(f"Invalid time to expiry (T): {T}")
            return None

        # Validate current price and strike price
        if current_price <= 0 or strike_price <= 0 or np.isnan(current_price) or np.isnan(strike_price):
            logger.error(f"Invalid current price or strike price: current_price={current_price}, strike_price={strike_price}")
            return None

        # Log all inputs for debugging
        logger.info(f"Inputs for Black-Scholes: current_price={current_price}, strike_price={strike_price}, "
                    f"sigma={sigma}, T={T}, option_type={option_type}")

        # Risk-free interest rate (assumed)
        r = 0.05

        # Black-Scholes formula
        d1 = (math.log(current_price / strike_price) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        logger.info(f"Calculated d1={d1}, d2={d2}")

        if option_type == "call":
            option_price = (current_price * norm.cdf(d1)) - (strike_price * math.exp(-r * T) * norm.cdf(d2))
        elif option_type == "put":
            option_price = (strike_price * math.exp(-r * T) * norm.cdf(-d2)) - (current_price * norm.cdf(-d1))
        else:
            logger.error("Invalid option type")
            return None

        logger.info(f"Calculated option price: {option_price}")
        return round(option_price, 2)
    except Exception as e:
        logger.error(f"Error estimating option price: {e}")
        return None

def fetch_holidays(api_key, country="IN", year=None):
    """
    Fetch holidays dynamically using the Calendarific API.
    """
    if year is None:
        year = datetime.now().year  # Default to the current year

    url = "https://calendarific.com/api/v2/holidays"
    params = {
        "api_key": api_key,
        "country": country,
        "year": year
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract holiday dates
        holidays = [
            holiday["date"]["iso"]
            for holiday in data["response"]["holidays"]
            if holiday["type"] and "National holiday" in holiday["type"]
        ]
        logger.info(f"Fetched holidays for {year}: {holidays}")
        return holidays
    except Exception as e:
        logger.error(f"Error fetching holidays: {e}")
        return []

# Example usage
api_key = "A1j9Nr72uN9scpcfcLmBJL2wGuOfVPXM"  # Replace with your API key
holidays = fetch_holidays(api_key, country="IN", year=2025)
print(f"Holidays in 2025: {holidays}")

@app.get("/debug_expiry")
def debug_expiry():
    tz = "Asia/Kolkata"
    today_local = datetime.now(ZoneInfo(tz)).date()
    weekday = today_local.weekday()
    expiry = get_next_expiry(holidays=[], tz=tz)
    return {
        "today_ist": today_local.strftime("%Y-%m-%d"),
        "today_weekday": weekday,  # Mon=0, Tue=1
        "expiry_ist_tuesday": expiry
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

