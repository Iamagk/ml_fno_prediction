import React, { useState } from "react";
import { predictMarket } from "./api";
import config from "./config";
import "./index.css";
import logo from "./symbol.jpg"; // Ensure this is correctly placed in 'src'

const FEATURE_NAMES = [
  "STRIKE_PR", "OPEN", "HIGH", "LOW","CLOSE", "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI",
  "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_SIGNAL", "EMA_9", "EMA_21", "EMA_50", "EMA_200",
  "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "MACD_HIST", "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV",
  "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10", "CMF", "PSAR", "AROON_UP", "AROON_DOWN", "RETURN"
];

const App = () => {
  const [features, setFeatures] = useState(Array(FEATURE_NAMES.length).fill(""));
  const [prediction, setPrediction] = useState(null);
  const [stockSymbol, setStockSymbol] = useState("");
  const [searchHistory, setSearchHistory] = useState([]);
  const [optionsPrediction, setOptionsPrediction] = useState(null);
  const [isIndicatorsExpanded, setIsIndicatorsExpanded] = useState(false);
  const [isCurrentDataExpanded, setIsCurrentDataExpanded] = useState(true); // Start expanded for current data

  // Separate basic and technical indicator features
  const basicFeatures = FEATURE_NAMES.slice(0, 10); // First 10 are basic market data
  const technicalIndicators = FEATURE_NAMES.slice(10); // Rest are technical indicators

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
      const response = await fetch(`${config.API_URL}/fetch_yfinance?symbol=${symbol}`);
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
      const response = await fetch(`${config.API_URL}/predict_with_options?symbol=${stockSymbol}`);
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
            fetchYahooFinanceData("^NSEI");
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

      {/* Current Data Section - Collapsible */}
      <div className="current-data-section">
        <div 
          className="current-data-header" 
          onClick={() => setIsCurrentDataExpanded(!isCurrentDataExpanded)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              setIsCurrentDataExpanded(!isCurrentDataExpanded);
            }
          }}
        >
          <h3 className="current-data-title">Current Data</h3>
          <span className={`collapse-icon ${isCurrentDataExpanded ? 'expanded' : ''}`}>
            ▼
          </span>
        </div>
        
        <div className={`current-data-content ${isCurrentDataExpanded ? 'expanded' : 'collapsed'}`}>
          <div className="form-container">
            {basicFeatures.map((name, index) => (
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
        </div>
      </div>

      {/* Technical Indicators Section - Collapsible */}
      <div className="technical-indicators-section">
        <div 
          className="technical-indicators-header" 
          onClick={() => setIsIndicatorsExpanded(!isIndicatorsExpanded)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              setIsIndicatorsExpanded(!isIndicatorsExpanded);
            }
          }}
        >
          <h3 className="technical-indicators-title">Technical Indicators</h3>
          <span className={`collapse-icon ${isIndicatorsExpanded ? 'expanded' : ''}`}>
            ▼
          </span>
        </div>
        
        <div className={`technical-indicators-content ${isIndicatorsExpanded ? 'expanded' : 'collapsed'}`}>
          <div className="form-container">
            {technicalIndicators.map((name, index) => {
              const actualIndex = index + basicFeatures.length; // Adjust index for full features array
              return (
                <div key={actualIndex} className="input-group">
                  <label>{name}</label>
                  <input
                    type="number"
                    value={features[actualIndex]}
                    onChange={(e) => handleChange(actualIndex, e.target.value)}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <button onClick={handlePredict}>Predict</button>
      <button onClick={handlePredictLive}>Predict Live</button>

      {/* Prediction and Options Prediction Side by Side */}
      <div className="prediction-container">
        {/* Prediction Summary */}
        {prediction && (
          <div>
            <h2>Prediction Summary</h2>
            <p><span>🔹 Prediction:</span> <strong>{prediction.prediction}</strong></p>
            <p><span>🔹 Suggested Action:</span> <strong>{prediction.suggested_action || "N/A"}</strong></p>
            <p><span>🔹 Strike Price:</span> <strong>{prediction.strike_price || "N/A"}</strong></p>
            <p><span>🔹 Stop Loss (Strike):</span> <strong>{prediction.stop_loss_strike || "N/A"}</strong></p>
            <p><span>🔹 Exit Price (Strike):</span> <strong>{prediction.exit_price_strike || "N/A"}</strong></p>
            <p><span>🔹 Expiry:</span> <strong>{prediction.expiry || "N/A"}</strong></p>
            <p><span>🔹 Confidence:</span> <strong>{prediction.confidence ? `${prediction.confidence}%` : "N/A"}</strong></p>
          </div>
        )}

        {/* Options Prediction */}
        {optionsPrediction && (
          <div>
            <h2>Options Prediction</h2>
            <p><span>🔹 Prediction:</span> <strong>{optionsPrediction.prediction}</strong></p>
            <p><span>🔹 Suggested Action:</span> <strong>{optionsPrediction.suggested_action || "N/A"}</strong></p>
            <p><span>🔹 Strike Price:</span> <strong>{optionsPrediction.strike_price || "N/A"}</strong></p>
            <p><span>🔹 Option Price:</span> <strong>{optionsPrediction.option_price ? `₹${optionsPrediction.option_price}` : "Calculating..."}</strong></p>
            <p><span>🔹 Stop Loss (Option):</span> <strong>{optionsPrediction.stop_loss_option ? `₹${optionsPrediction.stop_loss_option}` : "N/A"}</strong></p>
            <p><span>🔹 Exit Price (Option):</span> <strong>{optionsPrediction.exit_price_option ? `₹${optionsPrediction.exit_price_option}` : "N/A"}</strong></p>
            <p><span>🔹 Stop Loss (Strike):</span> <strong>{optionsPrediction.stop_loss_strike ? `₹${optionsPrediction.stop_loss_strike}` : "N/A"}</strong></p>
            <p><span>🔹 Exit Price (Strike):</span> <strong>{optionsPrediction.exit_price_strike ? `₹${optionsPrediction.exit_price_strike}` : "N/A"}</strong></p>
            <p><span>🔹 Expiry:</span> <strong>{optionsPrediction.expiry || "N/A"}</strong></p>
            <p><span>🔹 Confidence:</span> <strong>{optionsPrediction.confidence ? `${optionsPrediction.confidence}%` : "N/A"}</strong></p>
            
            {/* Show ensemble pricing details if available */}
            {optionsPrediction.pricing_ensemble && (
              <div className="ensemble-details">
                <h3>🔬 Ensemble Pricing Details</h3>
                <p><span>📊 Agreement Score:</span> <strong>{optionsPrediction.pricing_ensemble.agreement_score ? `${(optionsPrediction.pricing_ensemble.agreement_score * 100).toFixed(1)}%` : "N/A"}</strong></p>
                <p><span>🎯 Model Confidence:</span> <strong>{optionsPrediction.pricing_ensemble.confidence ? `${(optionsPrediction.pricing_ensemble.confidence * 100).toFixed(1)}%` : "N/A"}</strong></p>
                <p><span>📈 Volatility:</span> <strong>{optionsPrediction.pricing_ensemble.volatility ? `${optionsPrediction.pricing_ensemble.volatility}%` : "N/A"}</strong></p>
              </div>
            )}
            
            {/* Show Greeks if available */}
            {optionsPrediction.greeks && Object.keys(optionsPrediction.greeks).length > 0 && (
              <div className="greeks-details">
                <h3>🧮 Option Greeks</h3>
                <p><span>Δ Delta:</span> <strong>{optionsPrediction.greeks.delta ? optionsPrediction.greeks.delta.toFixed(3) : "N/A"}</strong></p>
                <p><span>Γ Gamma:</span> <strong>{optionsPrediction.greeks.gamma ? optionsPrediction.greeks.gamma.toFixed(3) : "N/A"}</strong></p>
                <p><span>Θ Theta:</span> <strong>{optionsPrediction.greeks.theta ? optionsPrediction.greeks.theta.toFixed(3) : "N/A"}</strong></p>
                <p><span>ν Vega:</span> <strong>{optionsPrediction.greeks.vega ? optionsPrediction.greeks.vega.toFixed(3) : "N/A"}</strong></p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
