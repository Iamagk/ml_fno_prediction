from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import List
from kiteconnect import KiteConnect

# 🔹 Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running!"}

# 🔹 Load the trained model
model_path = "models/fno_xgboost_model.json"
model = xgb.XGBClassifier()
model.load_model(model_path)

# 🔹 Define expected features
EXPECTED_FEATURES = 33  

# 🔹 Define actual feature names
FEATURE_NAMES = [
    "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
    "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
    "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
    "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
]

# 🔹 Zerodha Kite API Credentials (Replace with your credentials)
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
ACCESS_TOKEN = "your_access_token"

# 🔹 Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# 🔹 Define stock instrument (e.g., RELIANCE)
INSTRUMENT = "NSE:RELIANCE"

# 🔹 Fetch Live Market Data
def fetch_live_data():
    try:
        quote = kite.ltp(INSTRUMENT)
        last_price = quote[INSTRUMENT]["last_price"]

        # Simulate other technical indicators (Replace with real calculations if needed)
        features = {
            "Close": last_price,
            "High": last_price + 2,
            "Low": last_price - 2,
            "Open": last_price - 1,
            "Volume": np.random.randint(100000, 500000),
            "SMA_5": last_price - 0.5,
            "SMA_10": last_price - 1.0,
            "RSI_14": np.random.uniform(30, 70),
            "MACD": np.random.uniform(-2, 2),
            "MACD_Signal": np.random.uniform(-2, 2),
            "EMA_9": last_price - 0.3,
            "EMA_21": last_price - 0.6,
            "EMA_50": last_price - 1.2,
            "EMA_200": last_price - 5,
            "BB_upper": last_price + 3,
            "BB_middle": last_price,
            "BB_lower": last_price - 3,
            "MACD_Hist": np.random.uniform(-1, 1),
            "STOCH_K": np.random.uniform(20, 80),
            "STOCH_D": np.random.uniform(20, 80),
            "ATR": np.random.uniform(1, 5),
            "ROC_10": np.random.uniform(-1, 1),
            "OBV": np.random.randint(1000000, 5000000),
            "VWAP": last_price + 0.1,
            "ADX": np.random.uniform(10, 40),
            "CCI": np.random.uniform(-100, 100),
            "WILLR_14": np.random.uniform(-100, 0),
            "MOM_10": np.random.uniform(-1, 1),
            "CMF": np.random.uniform(-1, 1),
            "PSAR": last_price - 1.5,
            "Aroon_Up": np.random.uniform(0, 100),
            "Aroon_Down": np.random.uniform(0, 100),
            "Return": np.random.uniform(-1, 1),
        }

        return features
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# 🔹 Endpoint to Predict Using Real-Time Data
@app.get("/predict_live")
def predict_live():
    try:
        # Fetch live data
        live_data = fetch_live_data()
        if not live_data:
            raise HTTPException(status_code=500, detail="Failed to fetch live data")

        # Convert to DataFrame
        df = pd.DataFrame([live_data])

        # Make prediction
        prediction = model.predict(df)[0]

        return {"prediction": int(prediction), "live_data": live_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 🔹 Endpoint for list-based input
class MarketData(BaseModel):
    features: List[float]

@app.post("/predict")
def predict_list(data: MarketData):
    try:
        if len(data.features) != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail=f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}")

        input_array = np.array(data.features).reshape(1, -1)
        df = pd.DataFrame(input_array, columns=FEATURE_NAMES)

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
        input_array = np.array([[getattr(data, feature) for feature in FEATURE_NAMES]])
        prediction = model.predict(input_array)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 🔹 Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)