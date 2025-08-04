import React, { useState, useEffect } from "react";

const VercelTestApp = () => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [configStatus, setConfigStatus] = useState("Loading...");

  useEffect(() => {
    console.log("VercelTestApp mounted");
    console.log("Environment:", process.env.NODE_ENV);
    console.log("Location:", window.location.href);
    
    // Test config loading
    import("./config")
      .then(configModule => {
        console.log("Config loaded successfully:", configModule.default);
        setConfigStatus(`Config loaded: ${JSON.stringify(configModule.default)}`);
      })
      .catch(err => {
        console.error("Config loading failed:", err);
        setConfigStatus(`Config error: ${err.message}`);
      });
    
    setIsLoaded(true);
  }, []);

  if (!isLoaded) {
    return <div>Loading...</div>;
  }

  return (
    <div style={{ 
      padding: "20px", 
      fontFamily: "Arial, sans-serif",
      backgroundColor: "#f5f5f5",
      minHeight: "100vh"
    }}>
      <h1 style={{ color: "#2e7d32" }}>🚀 Vercel Deployment Test - SUCCESS!</h1>
      
      <div style={{ 
        backgroundColor: "white", 
        padding: "15px", 
        margin: "20px 0",
        borderRadius: "8px",
        border: "1px solid #ddd"
      }}>
        <h3>Environment Information:</h3>
        <p><strong>Node Environment:</strong> {process.env.NODE_ENV || 'undefined'}</p>
        <p><strong>Current URL:</strong> {window.location.href}</p>
        <p><strong>User Agent:</strong> {navigator.userAgent}</p>
        <p><strong>Config Status:</strong> {configStatus}</p>
      </div>

      <div style={{ 
        backgroundColor: "#e8f5e8", 
        padding: "15px", 
        borderRadius: "8px",
        border: "2px solid #4caf50"
      }}>
        <h3>✅ React is Working on Vercel!</h3>
        <p>If you can see this message, your React app is successfully deployed and running on Vercel.</p>
        <p>Next step: Switch back to your F&O Prediction Dashboard.</p>
      </div>

      <div style={{ marginTop: "20px" }}>
        <button 
          style={{
            padding: "10px 20px",
            backgroundColor: "#2196f3",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer"
          }}
          onClick={() => alert("Button click works!")}
        >
          Test Button Click
        </button>
      </div>
    </div>
  );
};

export default VercelTestApp;
