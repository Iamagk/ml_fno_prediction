import React, { useState } from "react";
import { predictMarket } from "./api";
import "./index.css";

const FEATURE_NAMES = [
  "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
  "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
  "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
  "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
];

const App = () => {
  const [features, setFeatures] = useState(Array(33).fill(""));
  const [prediction, setPrediction] = useState(null);
  const [stockSymbol, setStockSymbol] = useState("RELIANCE.NS"); // Default stock

  const handleChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handleSubmit = async () => {
    const featureValues = features.map(val => parseFloat(val) || 0);
    const result = await predictMarket(featureValues);
    setPrediction(result);
  };

  const fetchYahooFinanceData = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/fetch_yfinance?symbol=${stockSymbol}`);
      const data = await response.json();

      console.log("Yahoo Finance Data:", data); // Debugging

      if (data.error) {
        console.error("Error fetching data:", data.error);
        return;
      }

      // Fill feature values from Yahoo Finance data
      const newFeatures = FEATURE_NAMES.map(feature => data[feature] || 0);
      setFeatures(newFeatures);
    } catch (error) {
      console.error("Error fetching Yahoo Finance data:", error);
    }
  };

  const handlePredictLive = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/predict_live");
      const data = await response.json();

      console.log("Received API Data:", data); // Debugging
      setPrediction(data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div className="container">
      <h1>F&O Market Prediction</h1>

      <div className="stock-input">
        <label>Stock Symbol:</label>
        <input
          type="text"
          value={stockSymbol}
          onChange={(e) => setStockSymbol(e.target.value)}
          placeholder="Enter stock symbol (e.g., RELIANCE.NS)"
        />
        <button onClick={fetchYahooFinanceData}>Fetch Data</button>
      </div>

      <div className="form-container">
        {FEATURE_NAMES.map((name, index) => (
          <div key={index} className="input-group">
            <label>{name}</label>
            <input
              type="number"
              value={features[index]}
              onChange={(e) => handleChange(index, e.target.value)}
            />
          </div>
        ))}
      </div>

      <button onClick={handleSubmit}>Predict</button>
      <button onClick={handlePredictLive}>Predict Live</button>

      {prediction && (
        <div>
          <h2>Prediction: {prediction.prediction}</h2>
          <p>🔹 Suggested Action: {prediction.suggested_action || "N/A"}</p>
          <p>🔹 Strike Price: {prediction.strike_price || "N/A"}</p>
          <p>🔹 Stop Loss: {prediction.stop_loss || "N/A"}</p>
          <p>🔹 Expiry: {prediction.expiry || "N/A"}</p>
          <p>🔹 Confidence: {prediction.confidence ? `${prediction.confidence}%` : "N/A"}</p>
        </div>
      )}
    </div>
  );
};

export default App;