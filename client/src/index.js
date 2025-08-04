import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx"; // Your real F&O prediction app
import ErrorBoundary from "./ErrorBoundary.jsx";
import "./index.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);