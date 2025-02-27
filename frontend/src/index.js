import React, { useState } from "react";
import "./styles.css";
import api from "./api";

export default function App() {
  const [symbol, setSymbol] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    setError("");
    setPrediction(null);

    if (!symbol.trim()) {
      setError("Please enter a stock symbol.");
      return;
    }

    try {
      const response = await api.get(`/predict/${symbol}`);
      setPrediction(response.data);
    } catch (err) {
      setError("Error fetching prediction. Please try again.");
    }
  };

  return (
    <div className="container">
      <h1>F&O Prediction</h1>
      
      <input
        type="text"
        placeholder="Enter stock symbol (e.g., NIFTY)"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
      />

      <button onClick={handlePredict}>Predict</button>

      {error && <p className="error">{error}</p>}

      {prediction && (
        <div className="prediction">
          <h2>Prediction Result</h2>
          <p><strong>Symbol:</strong> {symbol}</p>
          <p><strong>Trend:</strong> {prediction.trend}</p>
          <p><strong>Best Option:</strong> {prediction.option}</p>
          <p><strong>Probability:</strong> {prediction.probability}%</p>
          <p><strong>Expiry:</strong> {prediction.expiry}</p>
          <p><strong>Stop Loss:</strong> {prediction.stop_loss}</p>
        </div>
      )}
    </div>
  );
}