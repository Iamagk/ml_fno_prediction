import React from "react";
import ReactDOM from "react-dom/client";
import DebugApp from "./DebugApp.jsx"; // Debug version to identify Vercel issues
import "./index.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <DebugApp />
  </React.StrictMode>
);