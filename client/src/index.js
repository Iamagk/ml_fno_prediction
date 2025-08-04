import React from "react";
import ReactDOM from "react-dom/client";
import SimpleApp from "./SimpleApp.jsx"; // Test with simple app
import "./index.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <SimpleApp />
  </React.StrictMode>
);