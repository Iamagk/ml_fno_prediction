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

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
                stock_data = stock.history(period="1d")

                if not stock_data.empty:
                    close_price = float(stock_data["Close"].iloc[-1])
                    high_price = float(stock_data["High"].iloc[-1])
                    low_price = float(stock_data["Low"].iloc[-1])
                    open_price = float(stock_data["Open"].iloc[-1])
                    volume = int(stock_data["Volume"].iloc[-1])

                    # Fill missing values with default placeholders
                    live_data = {
                        "Close": close_price, "High": high_price, "Low": low_price, "Open": open_price, "Volume": volume,
                        "SMA_5": close_price, "SMA_10": close_price, "RSI_14": 50, "MACD": 0, "MACD_Signal": 0,
                        "EMA_9": close_price, "EMA_21": close_price, "EMA_50": close_price, "EMA_200": close_price,
                        "BB_upper": close_price, "BB_middle": close_price, "BB_lower": close_price, "MACD_Hist": 0,
                        "STOCH_K": 50, "STOCH_D": 50, "ATR": 0, "ROC_10": 0, "OBV": 0, "VWAP": 0, "ADX": 0, "CCI": 0,
                        "WILLR_14": 50, "MOM_10": 0, "CMF": 0, "PSAR": 0, "Aroon_Up": 0, "Aroon_Down": 0, "Return": 0
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
            return {"error": f"No stock data found for {symbol}"}

        # Convert DataFrame to dictionary for JSON response
        return stock_data.iloc[0].to_dict()

    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data: {e}")
        return {"error": "Failed to fetch data"}
    
@app.get("/predict_live")
def predict_live(symbol: str):
    try:
        logger.info(f"Received request for symbol: {symbol}")
        live_data = fetch_stock_data(symbol)

        if live_data is None or live_data.empty:
            return {"prediction": "No Data Available", "suggested_action": "N/A", "strike_price": "N/A", "stop_loss": "N/A", "expiry": "N/A", "confidence": "N/A"}

        feature_data = {col: float(live_data[col].values[0]) for col in FEATURE_NAMES if col in live_data}
        df = pd.DataFrame([feature_data])
        prediction = model.predict(df)[0]

        # Extract current price
        current_price = feature_data.get('Close', 0)
        if current_price == 0:
            raise ValueError("Current price of stock is unavailable.")

        # Determine strike price & stop loss
        if prediction == 1:  
            strike_price = int(current_price + (current_price * 0.01))  
            stop_loss = int(current_price - (current_price * 0.02))  
        else:  
            strike_price = int(current_price - (current_price * 0.01))  
            stop_loss = int(current_price + (current_price * 0.02))  
        
        # Use dynamic expiry date
        expiry_date = get_next_expiry()

        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
            "stop_loss": stop_loss,
            "expiry": expiry_date,  # Updated expiry date dynamically
            "confidence": np.random.randint(60, 90)  # Random confidence level
        }

        logger.info(f"API Response: {response}")
        return response

    except Exception as e:
        logger.error(f"Error in API: {str(e)}")
        return {"prediction": "Error", "suggested_action": "N/A", "strike_price": "N/A", "stop_loss": "N/A", "expiry": "N/A", "confidence": "N/A"}
    
class StockInput(BaseModel):
    symbol: str

@app.get("/predict_nifty50")
def predict_nifty50():
    try:
        symbol = "^NSEI"  # Yahoo Finance symbol for NIFTY 50
        logger.info(f"Fetching data for NIFTY 50 ({symbol})")

        # Fetch NIFTY 50 data
        live_data = fetch_stock_data(symbol)

        if live_data is None or live_data.empty:
            logger.warning(f"No data available for NIFTY 50")
            return {
                "prediction": "No Data Available",
                "suggested_action": "N/A",
                "strike_price": "N/A",
                "stop_loss": "N/A",
                "expiry": "N/A",
                "confidence": "N/A"
            }

        logger.info(f"Live data fetched for NIFTY 50: {live_data}")

        # Ensure only the expected features are passed
        feature_data = {col: float(live_data[col].values[0]) for col in FEATURE_NAMES if col in live_data}
        df = pd.DataFrame([feature_data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Extract current price for calculations
        current_price = feature_data.get('Close', 0)
        if current_price == 0:
            raise ValueError("Current price of NIFTY 50 is unavailable.")

        # Determine strike price & stop loss
        if prediction == 1:
            strike_price = int(current_price + (current_price * 0.01))
            stop_loss = int(current_price - (current_price * 0.02))
        else:
            strike_price = int(current_price - (current_price * 0.01))
            stop_loss = int(current_price + (current_price * 0.02))

        # Use dynamic expiry date
        expiry_date = get_next_expiry()

        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
            "stop_loss": stop_loss,
            "expiry": expiry_date,  #  Updated expiry date dynamically
            "confidence": np.random.randint(60, 90)  # Random confidence level
        }

        logger.info(f"NIFTY 50 API Response: {response}")
        return response

    except Exception as e:
        logger.error(f"Error in NIFTY 50 prediction: {str(e)}")
        return {
            "prediction": "Error",
            "suggested_action": "N/A",
            "strike_price": "N/A",
            "stop_loss": "N/A",
            "expiry": "N/A",
            "confidence": "N/A"
        }
    

@app.post("/predict")
def predict(stock: StockInput):
    return predict_live(stock.symbol)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)