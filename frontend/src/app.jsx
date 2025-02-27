// frontend/src/App.jsx
import { useState } from "react";
import api from "./api";
import "./index.css";

function App() {
  const [prediction, setPrediction] = useState(null);

  const getPrediction = async () => {
    try {
      const response = await api.post("/predict", {
        features: [100.5, 101.2, 99.8, 100.0, 200000, 101.0, 100.2, 55.3, 0.5, -0.1]
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div className="container">
      <h1>F&O Prediction</h1>
      <button onClick={getPrediction}>Get Prediction</button>
      {prediction !== null && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default App;