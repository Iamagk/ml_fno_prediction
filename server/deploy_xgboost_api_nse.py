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
from datetime import datetime, timedelta
import requests  # Import requests for NSE API

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
    weekday = today.weekday()
    if weekday >= 3:  
        days_until_next_expiry = (3 - weekday) + 7  
    else:
        days_until_next_expiry = 3 - weekday  
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

FEATURE_NAMES = [
    "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
    "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
    "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
    "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to fetch nearest strike price from NSE API
def get_nearest_strike_price(symbol, current_price):
    try:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.upper()}"
        headers = {"User-Agent": "Mozilla/5.0"}  # NSE API requires a user-agent
        response = requests.get(url, headers=headers)
        data = response.json()
        available_strikes = sorted([entry['strikePrice'] for entry in data['records']['data']])
        nearest_strike = min(available_strikes, key=lambda x: abs(x - current_price))
        logger.info(f"Nearest strike price for {symbol} at {current_price}: {nearest_strike}")
        return nearest_strike
    except Exception as e:
        logger.error(f"Error fetching strike price from NSE API for {symbol}: {e}")
        return None

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
        current_price = feature_data.get('Close', 0)
        confidence = get_model_confidence(model, df)
        strike_price = get_nearest_strike_price(symbol, current_price) or current_price
        expiry_date = get_next_expiry()
        response = {"prediction": int(prediction), "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option", "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE", "stop_loss": strike_price, "expiry": expiry_date, "confidence": float(confidence)}
        return response
    except Exception as e:
        return {"prediction": "Error", "suggested_action": "N/A", "strike_price": "N/A", "stop_loss": "N/A", "expiry": "N/A", "confidence": "N/A"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
