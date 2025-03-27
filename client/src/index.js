import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx"; // Ensure App.jsx exists in the same directory
import "./index.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);