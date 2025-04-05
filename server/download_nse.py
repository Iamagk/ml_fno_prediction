import yfinance as yf
ticker = yf.Ticker("^NSEI")  # Try "TCS.BO" if needed
data = ticker.history(period="1d")

if data.empty:
    print("No stock data found for TCS")
else:
    print(data)


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
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if data.empty:
            raise ValueError("No data found for the given symbol.")
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
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

        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
            "stop_loss": stop_loss,
            "expiry": "This Week",
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

        response = {
            "prediction": int(prediction),
            "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
            "strike_price": f"{strike_price} CE" if prediction == 1 else f"{strike_price} PE",
            "stop_loss": stop_loss,
            "expiry": "This Week",
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