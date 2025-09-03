import React, { useState } from "react";
import { predictMarket } from "./api";
import config from "./config";
import "./index.css";
import logo from "./symbol.jpg"; // Ensure this is correctly placed in 'src'

// Helper: get a Date object in IST
const toISTDate = (d = new Date()) => new Date(d.toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
// Helper: format YYYY-MM-DD from a Date (in IST)
const formatYMD = (d) => {
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
};
// Compute next Tuesday (IST)
const nextTuesdayIST = (base = new Date()) => {
  const ist = toISTDate(base);
  const day = ist.getDay(); // Sun=0, Mon=1, Tue=2
  let daysAhead = (2 - day + 7) % 7;
  if (daysAhead === 0) daysAhead = 7; // always next week's Tuesday
  const res = new Date(ist);
  res.setDate(ist.getDate() + daysAhead);
  return formatYMD(res);
};
// Ensure a date string is Tuesday (IST). If not, move to the next Tuesday.
const ensureTuesdayExpiry = (expiryStr) => {
  try {
    if (!expiryStr) return nextTuesdayIST();
    // Parse as IST by constructing from parts
    const [y, m, d] = expiryStr.split("-").map((n) => parseInt(n, 10));
    if (!y || !m || !d) return nextTuesdayIST();
    const asUTC = new Date(Date.UTC(y, m - 1, d)); // neutral base
    const ist = toISTDate(asUTC);
    const weekday = ist.getDay(); // Tue = 2
    if (weekday === 2) return formatYMD(ist);
    return nextTuesdayIST(ist);
  } catch {
    return nextTuesdayIST();
  }
};
// Flatten /predict_combined payload into a single object and normalize expiry to Tuesday
const normalizePrediction = (data) => {
  if (!data) return null;
  if (data.summary || data.options) {
    const summary = data.summary ?? {};
    const options = data.options ?? {};
    const flat = {
      prediction: summary.prediction ?? options.prediction,
      suggested_action: summary.suggested_action ?? options.suggested_action,
      strike_price: summary.strike_price ?? options.strike_price,
      option_price: options.option_price,
      stop_loss_strike: options.stop_loss_strike ?? summary.stop_loss,
      exit_price_strike: options.exit_price_strike,
      stop_loss_option: options.stop_loss_option,
      exit_price_option: options.exit_price_option,
      expiry: ensureTuesdayExpiry(summary.expiry ?? options.expiry),
      confidence: summary.confidence ?? options.confidence,
      greeks: data.greeks,
      pricing_ensemble: data.pricing_ensemble,
      // pass-through individual model info if present
      individual_model_predictions: data.individual_model_predictions,
      individual_model_details: data.individual_model_details,
    };
    return flat;
  }
  // For legacy /predict_with_options shape; just normalize expiry (keep extra fields)
  return {
    ...data,
    expiry: ensureTuesdayExpiry(data.expiry),
  };
};

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
      if (!stockSymbol.trim()) {
        alert("Enter a stock symbol first.");
        return;
      }
      // Prefer combined endpoint; fallback to predict_with_options
      let resp = await fetch(`${config.API_URL}/predict_combined?symbol=${stockSymbol}`);
      if (resp.status === 404) {
        resp = await fetch(`${config.API_URL}/predict_with_options?symbol=${stockSymbol}`);
      }
      const raw = await resp.json();
      console.log("API Response (combined/options):", raw);

      if (raw.error) {
        console.error("Error fetching prediction:", raw.error);
        alert(raw.error);
        return;
      }

      // Normalize shape and expiry (force Tuesday in IST)
      let normalized = normalizePrediction(raw);

      // If individual model details are missing, fetch from predict_with_options and merge
      const hasIndiv =
        (normalized && normalized.individual_model_details && normalized.individual_model_details.length > 0) ||
        (normalized && normalized.individual_model_predictions && Object.keys(normalized.individual_model_predictions || {}).length > 0);

      if (!hasIndiv) {
        try {
          const resp2 = await fetch(`${config.API_URL}/predict_with_options?symbol=${stockSymbol}`);
          const raw2 = await resp2.json();
          if (!raw2.error) {
            const norm2 = normalizePrediction(raw2);
            normalized = {
              ...normalized,
              individual_model_predictions: norm2?.individual_model_predictions,
              individual_model_details: norm2?.individual_model_details,
            };
          }
        } catch (e) {
          console.warn("Could not fetch individual model details:", e);
        }
      }

      setPrediction(normalized);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Failed to fetch prediction. Please try again later.");
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

      {/* Prediction - single combined panel */}
      <div className="prediction-container">
        {prediction && (
          <div>
            <h2>Prediction</h2>

            <p><span>🔹 Prediction:</span> <strong>{prediction.prediction ?? "N/A"}</strong></p>
            <p><span>🔹 Suggested Action:</span> <strong>{prediction.suggested_action || "N/A"}</strong></p>

            {/* Strike text (e.g., "24750 CE") */}
            <p><span>🔹 Strike Price:</span> <strong>{prediction.strike_price || "N/A"}</strong></p>
            {/* Strike-based levels */}
            {("stop_loss_strike" in prediction || "exit_price_strike" in prediction) && (
              <>
                <p><span>🔹 Stop Loss (Strike):</span> <strong>{prediction.stop_loss_strike ?? "N/A"}</strong></p>
                <p><span>🔹 Exit Price (Strike):</span> <strong>{prediction.exit_price_strike ?? "N/A"}</strong></p>
              </>
            )}
            {/* Option price (if backend provides) */}
            {"option_price" in prediction && (
              <p><span>🔹 Option Price:</span> <strong>{prediction.option_price != null ? `₹${prediction.option_price}` : "Calculating..."}</strong></p>
            )}
            {/* Option-based levels */}
            {("stop_loss_option" in prediction || "exit_price_option" in prediction) && (
              <>
                <p><span>🔹 Stop Loss (Option):</span> <strong>{prediction.stop_loss_option != null ? `₹${prediction.stop_loss_option}` : "N/A"}</strong></p>
                <p><span>🔹 Exit Price (Option):</span> <strong>{prediction.exit_price_option != null ? `₹${prediction.exit_price_option}` : "N/A"}</strong></p>
              </>
            )}

            <p><span>🔹 Expiry:</span> <strong>{prediction.expiry || "N/A"}</strong></p>
            <p>
              <span>🔹 Confidence:</span>{" "}
              <strong>
                {prediction.confidence != null
                  ? `${(Number(prediction.confidence) > 1 ? Number(prediction.confidence) : Number(prediction.confidence) * 100).toFixed(2)}%`
                  : "N/A"}
              </strong>
            </p>

            {/* Individual Models block removed */}

            {/* Ensemble pricing details if present */}
            {prediction.pricing_ensemble && (
              <div className="ensemble-details">
                <h3>🔬 Ensemble Pricing Details</h3>
                <p>
                  <span>📊 Agreement Score:</span>{" "}
                  <strong>
                    {prediction.pricing_ensemble.agreement_score != null
                      ? `${(prediction.pricing_ensemble.agreement_score * 100).toFixed(1)}%`
                      : "N/A"}
                  </strong>
                </p>
                <p>
                  <span>🎯 Model Confidence:</span>{" "}
                  <strong>
                    {prediction.pricing_ensemble.confidence != null
                      ? `${(prediction.pricing_ensemble.confidence * 100).toFixed(1)}%`
                      : "N/A"}
                  </strong>
                </p>
                <p>
                  <span>📈 Volatility:</span>{" "}
                  <strong>
                    {prediction.pricing_ensemble.volatility != null
                      ? `${prediction.pricing_ensemble.volatility}%`
                      : "N/A"}
                  </strong>
                </p>
              </div>
            )}

            {/* Greeks if present */}
            {prediction.greeks && Object.keys(prediction.greeks).length > 0 && (
              <div className="greeks-details">
                <h3>🧮 Option Greeks</h3>
                <p><span>Δ Delta:</span> <strong>{prediction.greeks.delta != null ? Number(prediction.greeks.delta).toFixed(3) : "N/A"}</strong></p>
                <p><span>Γ Gamma:</span> <strong>{prediction.greeks.gamma != null ? Number(prediction.greeks.gamma).toFixed(3) : "N/A"}</strong></p>
                <p><span>Θ Theta:</span> <strong>{prediction.greeks.theta != null ? Number(prediction.greeks.theta).toFixed(3) : "N/A"}</strong></p>
                <p><span>ν Vega:</span> <strong>{prediction.greeks.vega != null ? Number(prediction.greeks.vega).toFixed(3) : "N/A"}</strong></p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
