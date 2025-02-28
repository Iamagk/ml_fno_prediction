from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import List
import yfinance as yf  # Added Yahoo Finance
# from kiteconnect import KiteConnect  # Commented out for now
from fastapi.middleware.cors import CORSMiddleware

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

# Load the trained model
model_path = "models/fno_xgboost_model.json"
model = xgb.XGBClassifier()
model.load_model(model_path)

# Define expected features
EXPECTED_FEATURES = 33  

# Define actual feature names
FEATURE_NAMES = [
    "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
    "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
    "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
    "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
]

# Zerodha Kite API Credentials (Keep this for future integration)
# API_KEY = "your_api_key"
# API_SECRET = "your_api_secret"
# ACCESS_TOKEN = "your_access_token"

# Initialize KiteConnect (Commented out for now)
# kite = KiteConnect(api_key=API_KEY)
# kite.set_access_token(ACCESS_TOKEN)

# Define stock instrument (Replace this dynamically)
INSTRUMENT = "RELIANCE.NS"  # Use NSE ticker symbol for Yahoo Finance


# 🔹 Fetch Live Market Data from Yahoo Finance
def fetch_live_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1d", interval="1m")  # Get the latest 1-minute data

        if hist.empty:
            return None  # No data available

        latest_data = hist.iloc[-1]  # Get the most recent row

        features = {
            "Close": latest_data["Close"],
            "High": latest_data["High"],
            "Low": latest_data["Low"],
            "Open": latest_data["Open"],
            "Volume": latest_data["Volume"],
            # Placeholder values for other technical indicators
            **{feature: np.random.uniform(1, 100) for feature in FEATURE_NAMES if feature not in ["Close", "High", "Low", "Open", "Volume"]}
        }

        return features
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        return None


# 🔹 Predict Using Live Stock Data (Yahoo Finance)
@app.get("/predict_live")
def predict_live(symbol: str = "RELIANCE.NS"):
    try:
        live_data = fetch_live_data(symbol)

        if not live_data:
            response = {
                "prediction": "No Live Data",
                "suggested_action": "N/A",
                "strike_price": "N/A",
                "stop_loss": "N/A",
                "expiry": "N/A",
                "confidence": "N/A"
            }
        else:
            df = pd.DataFrame([live_data])
            prediction = model.predict(df)[0]

            response = {
                "prediction": int(prediction),
                "suggested_action": "Buy Call Option" if prediction == 1 else "Buy Put Option",
                "strike_price": "22,100 CE" if prediction == 1 else "21,900 PE",
                "stop_loss": 21900 if prediction == 1 else 22100,
                "expiry": "This Week",
                "confidence": np.random.randint(60, 90)  # Random confidence score
            }

        print("API Response:", response)  # Debugging
        return response

    except Exception as e:
        print("Error in API:", str(e))
        return {
            "prediction": "Error",
            "suggested_action": "N/A",
            "strike_price": "N/A",
            "stop_loss": "N/A",
            "expiry": "N/A",
            "confidence": "N/A"
        }


# 🔹 Endpoint for list-based input
class MarketData(BaseModel):
    features: List[float]

@app.post("/predict")
def predict_list(data: MarketData):
    try:
        if len(data.features) != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail=f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}")
        df = pd.DataFrame([data.features], columns=FEATURE_NAMES)
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 🔹 Endpoint for named feature input
class ModelInput(BaseModel):
    Close: float
    High: float
    Low: float
    Open: float
    Volume: float
    SMA_5: float
    SMA_10: float
    RSI_14: float
    MACD: float
    MACD_Signal: float
    EMA_9: float
    EMA_21: float
    EMA_50: float
    EMA_200: float
    BB_upper: float
    BB_middle: float
    BB_lower: float
    MACD_Hist: float
    STOCH_K: float
    STOCH_D: float
    ATR: float
    ROC_10: float
    OBV: float
    VWAP: float
    ADX: float
    CCI: float
    WILLR_14: float
    MOM_10: float
    CMF: float
    PSAR: float
    Aroon_Up: float
    Aroon_Down: float
    Return: float

@app.post("/predict_named")
def predict_named(data: ModelInput):
    try:
        df = pd.DataFrame([[getattr(data, feature) for feature in FEATURE_NAMES]], columns=FEATURE_NAMES)
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)