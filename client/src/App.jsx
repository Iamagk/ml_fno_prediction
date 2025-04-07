import React, { useState } from "react";
import { predictMarket } from "./api";
import "./index.css";
import logo from "./symbol.jpg"; // Ensure this is correctly placed in 'src'

const FEATURE_NAMES = [
  "Close", "High", "Low", "Open", "Volume", "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_Signal",
  "EMA_9", "EMA_21", "EMA_50", "EMA_200", "BB_upper", "BB_middle", "BB_lower", "MACD_Hist",
  "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV", "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10",
  "CMF", "PSAR", "Aroon_Up", "Aroon_Down", "Return"
];

const App = () => {
  const [features, setFeatures] = useState(Array(FEATURE_NAMES.length).fill(""));
  const [prediction, setPrediction] = useState(null);
  const [stockSymbol, setStockSymbol] = useState("");
  const [searchHistory, setSearchHistory] = useState([]);
  const [optionsPrediction, setOptionsPrediction] = useState(null);

  const handleChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handleStockChange = (event) => {
    setStockSymbol(event.target.value.toUpperCase());
  };

  const handleSearch = () => {
    if (!stockSymbol.trim()) return;
    if (!searchHistory.includes(stockSymbol)) {
      setSearchHistory([stockSymbol, ...searchHistory].slice(0, 5)); // Keep last 5 searches
    }
    fetchYahooFinanceData(stockSymbol);
  };

  const fetchYahooFinanceData = async (symbol) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/fetch_yfinance?symbol=${symbol}`);
      const data = await response.json();
      console.log("Fetched Data for:", symbol, data);
      if (data.error) {
        console.error("Error fetching data:", data.error);
        return;
      }
      const newFeatures = FEATURE_NAMES.map((feature) => data[feature] || 0);
      setFeatures(newFeatures);
    } catch (error) {
      console.error("Error fetching Yahoo Finance data:", error);
    }
  };

  const handlePredict = async () => {
    const featureValues = features.map((val) => parseFloat(val) || 0);
    console.log("Features sent to model:", featureValues);
    const result = await predictMarket(featureValues);
    setPrediction(result);
  };

  const handlePredictLive = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/predict_with_options?symbol=${stockSymbol}`);
      const data = await response.json();
      console.log("API Response for predict_with_options:", data); // Log the response

      if (data.error) {
        console.error("Error fetching prediction:", data.error);
        alert(data.error); // Show error in a pop-up
        return;
      }

      // Update both prediction and optionsPrediction states
      setPrediction(data);
      setOptionsPrediction(data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Failed to fetch prediction. Please try again later."); // Show error in a pop-up
    }
  };

  return (
    <div className="container">
      {/* Title with small logo beside it */}
      <div className="title-container">
        <img src={logo} alt="Logo" className="small-logo" />
        <h1>F&O Market Prediction</h1>
      </div>

      <div className="stock-selector">
        <input
          type="text"
          placeholder="Enter Stock Symbol..."
          value={stockSymbol}
          onChange={handleStockChange}
        />
        <button onClick={handleSearch}>Search</button>
        <button
          onClick={() => {
            setStockSymbol("^NSEI");
            handleSearch("^NSEI");
          }}
        >
          NIFTY50
        </button>

        {searchHistory.length > 0 && (
          <select onChange={(e) => setStockSymbol(e.target.value)}>
            <option value="">Recent Searches</option>
            {searchHistory.map((ticker, index) => (
              <option key={index} value={ticker}>
                {ticker}
              </option>
            ))}
          </select>
        )}
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

      <button onClick={handlePredict}>Predict</button>
      <button onClick={handlePredictLive}>Predict Live</button>

      {/* Prediction and Options Prediction Side by Side */}
      <div style={{ display: "flex", justifyContent: "space-between", gap: "20px" }}>
        {/* Prediction Summary */}
        {prediction && (
          <div style={{ flex: 1, border: "1px solid #ccc", padding: "10px", borderRadius: "5px" }}>
            <h2>Prediction Summary</h2>
            <p>🔹 Prediction: {prediction.prediction}</p>
            <p>🔹 Suggested Action: {prediction.suggested_action || "N/A"}</p>
            <p>🔹 Strike Price: {prediction.strike_price || "N/A"}</p>
            <p>🔹 Stop Loss (Strike): {prediction.stop_loss_strike || "N/A"}</p>
            <p>🔹 Expiry: {prediction.expiry || "N/A"}</p>
            <p>🔹 Confidence: {prediction.confidence ? `${prediction.confidence}%` : "N/A"}</p>
          </div>
        )}

        {/* Options Prediction */}
        {optionsPrediction && optionsPrediction.option_price && (
          <div style={{ flex: 1, border: "1px solid #ccc", padding: "10px", borderRadius: "5px" }}>
            <h2>Options Prediction</h2>
            <p>🔹 Prediction: {optionsPrediction.prediction}</p>
            <p>🔹 Suggested Action: {optionsPrediction.suggested_action || "N/A"}</p>
            <p>🔹 Option Price: {optionsPrediction.option_price || "N/A"}</p>
            <p>🔹 Stop Loss (Option): {optionsPrediction.stop_loss_option || "N/A"}</p>
            <p>🔹 Expiry: {optionsPrediction.expiry || "N/A"}</p>
            <p>🔹 Confidence: {optionsPrediction.confidence ? `${optionsPrediction.confidence}%` : "N/A"}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
