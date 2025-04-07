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
from scipy.stats import norm
import math

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_next_expiry():
    today = datetime.today()
    weekday = today.weekday()  # Monday = 0, Sunday = 6

    # If today is Thursday or after, get next Thursday
    if weekday >= 3:  
        days_until_next_expiry = (3 - weekday) + 7  # Move to next Thursday
    else:
        days_until_next_expiry = 3 - weekday  # This week's Thursday

    next_expiry = today + timedelta(days=days_until_next_expiry)
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
    "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
    "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
    "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
    "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to fetch stock data with retries
def fetch_stock_data(symbol):
    # Ensure NIFTY50 is fetched directly without suffix
    if symbol.upper() == "^NSEI":
        possible_symbols = [symbol]
    else:
        possible_symbols = [f"{symbol}.NS", f"{symbol}.BO", symbol]  # NSE, BSE, and direct symbol

    for ticker in possible_symbols:
        for attempt in range(3):  # Retry up to 3 times
            try:
                logger.info(f"Fetching data for {ticker} (Attempt {attempt+1})")
                stock = yf.Ticker(ticker)
                stock_data = stock.history(period="1y")  # Fetch 90 days of data
                stock_data = stock_data.dropna(subset=['Close'])

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
                    stock_data['Return'] = stock_data['Close'].pct_change() * 100  # Percentage change in closing price

                    # Extract the latest row of data
                    latest_data = stock_data.iloc[-1]
                    logger.info(f"Latest data for {ticker}: {latest_data}")

                    # Fill missing values with default placeholders
                    live_data = {
                        "Close": latest_data["Close"],
                        "High": latest_data["High"],
                        "Low": latest_data["Low"],
                        "Open": latest_data["Open"],
                        "Volume": latest_data["Volume"],
                        "SMA_5": latest_data["SMA_5"],
                        "SMA_10": latest_data["SMA_10"],
                        "RSI_14": latest_data["RSI_14"],
                        "MACD": latest_data["MACD"],
                        "MACD_Signal": latest_data["MACD_Signal"],
                        "ATR": latest_data["ATR"],
                        "EMA_9": latest_data["Close"],  # Placeholder
                        "EMA_21": latest_data["Close"],  # Placeholder
                        "EMA_50": latest_data["Close"],  # Placeholder
                        "EMA_200": latest_data["Close"],  # Placeholder
                        "BB_upper": latest_data["Close"],  # Placeholder
                        "BB_middle": latest_data["Close"],  # Placeholder
                        "BB_lower": latest_data["Close"],  # Placeholder
                        "MACD_Hist": latest_data["MACD_Hist"],
                        "STOCH_K": latest_data["STOCH_K"],
                        "STOCH_D": latest_data["STOCH_D"],
                        "ROC_10": latest_data["ROC_10"],
                        "OBV": latest_data["OBV"],
                        "VWAP": latest_data["VWAP"],
                        "ADX": latest_data["ADX"],
                        "CCI": latest_data["CCI"],
                        "WILLR_14": latest_data["WILLR_14"],
                        "MOM_10": latest_data["MOM_10"],
                        "CMF": latest_data["CMF"],
                        "PSAR": latest_data["PSAR"],
                        "Aroon_Up": latest_data["Aroon_Up"],
                        "Aroon_Down": latest_data["Aroon_Down"],
                        "Return": latest_data["Return"],  # Use the calculated return
                    }
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
        feature_data = {col: float(live_data[col].values[0]) for col in FEATURE_NAMES if col in live_data}
        df = pd.DataFrame([feature_data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Extract current price for calculations
        current_price = feature_data.get('Close', 0)
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

        # Use dynamic expiry date
        expiry_date = get_next_expiry()

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
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_macd_hist(data):
    macd, signal = calculate_macd(data)
    macd_hist = macd - signal
    return macd_hist

def calculate_stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()  # 3-period moving average of %K
    return stoch_k, stoch_d

def calculate_roc(data, period=10):
    roc = ((data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)) * 100
    return roc

def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_vwap(data):
    vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']

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
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

def calculate_willr(data, period=14):
    high_max = data['High'].rolling(window=period).max()
    low_min = data['Low'].rolling(window=period).min()
    willr = -100 * ((high_max - data['Close']) / (high_max - low_min))
    return willr

def calculate_momentum(data, period=10):
    momentum = data['Close'] - data['Close'].shift(period)
    return momentum

def calculate_cmf(data, period=20):
    money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    money_flow_volume = money_flow_multiplier * data['Volume']
    cmf = money_flow_volume.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
    return cmf

def calculate_psar(data, step=0.02, max_step=0.2):
    high = data['High']
    low = data['Low']
    close = data['Close']

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
    aroon_up = 100 * (period - data['High'].rolling(window=period).apply(lambda x: period - np.argmax(x), raw=True)) / period
    aroon_down = 100 * (period - data['Low'].rolling(window=period).apply(lambda x: period - np.argmin(x), raw=True)) / period
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
        stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
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

@app.get("/predict_with_options")
def predict_with_options(symbol: str):
    try:
        logger.info(f"Received request for /predict_with_options with symbol: {symbol}")

        # Fetch live stock data
        live_data = fetch_stock_data(symbol)

        if live_data is None or live_data.empty:
            logger.warning(f"No data available for symbol: {symbol}")
            return {"error": f"No data available for the symbol: {symbol}"}

        logger.info(f"Live data fetched for {symbol}: {live_data}")

        # Ensure only the expected features are passed
        feature_data = {col: float(live_data[col].values[0]) for col in FEATURE_NAMES if col in live_data}
        df = pd.DataFrame([feature_data])

        # Log feature data
        logger.info(f"Feature data for prediction: {feature_data}")

        # Make prediction
        prediction = model.predict(df)[0]
        logger.info(f"Model prediction: {prediction}")

        # Extract current price for calculations
        current_price = feature_data.get('Close', 0)
        if current_price == 0:
            raise ValueError("Current price of the stock is unavailable.")

        # Get actual model confidence score
        confidence = get_model_confidence(model, df)
        logger.info(f"Model confidence: {confidence}")

        # Fetch the real strike price from Yahoo Finance
        strike_price = get_nearest_strike_price(symbol, current_price) or current_price
        logger.info(f"Nearest strike price: {strike_price}")

        # Round strike price to the nearest denomination of 50
        if prediction == 1:  # Call option
            strike_price = math.ceil(strike_price / 50) * 50  # Round up to the nearest 50
            option_type = "call"
        else:  # Put option
            strike_price = math.floor(strike_price / 50) * 50  # Round down to the nearest 50
            option_type = "put"

        logger.info(f"Adjusted strike price: {strike_price}")

        # Calculate stop loss for strike price using technical indicators
        atr = feature_data.get('ATR', 0)  # Average True Range
        support_level = feature_data.get('Support', strike_price - atr)
        resistance_level = feature_data.get('Resistance', strike_price + atr)

        if prediction == 1:
            stop_loss_strike = support_level  # Use support level as stop loss for a buy trade
        else:
            stop_loss_strike = resistance_level  # Use resistance level as stop loss for a sell trade

        # Round stop loss for strike price to 2 decimal places
        stop_loss_strike = round(float(stop_loss_strike), 2)
        logger.info(f"Rounded stop loss for strike price: {stop_loss_strike}")

        # Use dynamic expiry date
        expiry_date = get_next_expiry()
        logger.info(f"Expiry date: {expiry_date}")

        # Estimate option price
        option_price = estimate_option_price(symbol, current_price, strike_price, expiry_date, option_type)

        if option_price is None or np.isnan(option_price):
            logger.error("Option price could not be calculated")
            option_price = None  # Ensure it's explicitly set to None
        else:
            option_price = round(float(option_price), 3)  # Convert to Python float and round to 3 decimal places
            logger.info(f"Rounded option price: {option_price}")

        # Calculate stop loss for options price
        if option_price is not None:
            if prediction == 1:  # Call option
                stop_loss_option = option_price - atr  # Subtract ATR for call option
            else:  # Put option
                stop_loss_option = option_price + atr  # Add ATR for put option

            # Round stop loss for options price to 2 decimal places
            stop_loss_option = round(float(stop_loss_option), 2)
            logger.info(f"Rounded stop loss for options price: {stop_loss_option}")
        else:
            stop_loss_option = None  # If option price is None, stop loss cannot be calculated

        # Round confidence to 3 decimal places
        confidence = round(float(confidence), 3)
        logger.info(f"Rounded confidence: {confidence}")

        # Construct the response
        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
            "stop_loss_strike": stop_loss_strike,
            "stop_loss_option": stop_loss_option,
            "expiry": expiry_date,
            "confidence": confidence,
            "option_price": option_price
        }

        logger.info(f"API Response sent to frontend: {response}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

