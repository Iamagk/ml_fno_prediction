import React, { useState, useEffect } from "react";

const DebugApp = () => {
  const [debugInfo, setDebugInfo] = useState({});
  const [configTest, setConfigTest] = useState(null);

  useEffect(() => {
    // Test basic React functionality
    console.log("DebugApp mounted successfully");
    
    // Test environment detection
    const env = process.env.NODE_ENV || 'unknown';
    console.log("Environment:", env);
    
    // Test config import
    try {
      import("./config").then(configModule => {
        console.log("Config loaded:", configModule.default);
        setConfigTest(configModule.default);
      }).catch(err => {
        console.error("Config load error:", err);
        setConfigTest({ error: err.message });
      });
    } catch (err) {
      console.error("Config import error:", err);
      setConfigTest({ error: err.message });
    }

    // Test API availability
    const apiUrl = env === 'production' ? 'https://ml-fno-prediction-server.onrender.com' : 'http://localhost:8000';
    fetch(`${apiUrl}/health`)
      .then(response => response.json())
      .then(data => {
        console.log("API health check:", data);
        setDebugInfo(prev => ({ ...prev, apiHealth: data }));
      })
      .catch(err => {
        console.error("API health check failed:", err);
        setDebugInfo(prev => ({ ...prev, apiError: err.message }));
      });

    setDebugInfo({
      nodeEnv: env,
      userAgent: navigator.userAgent,
      location: window.location.href,
      timestamp: new Date().toISOString()
    });
  }, []);

  return (
    <div style={{ 
      padding: "20px", 
      fontFamily: "monospace", 
      backgroundColor: "#f0f0f0",
      minHeight: "100vh"
    }}>
      <h1 style={{ color: "green" }}>🔧 Debug App - Vercel Deployment Test</h1>
      
      <div style={{ marginBottom: "20px", padding: "10px", backgroundColor: "white", border: "1px solid #ccc" }}>
        <h3>Environment Info:</h3>
        <pre>{JSON.stringify(debugInfo, null, 2)}</pre>
      </div>

      <div style={{ marginBottom: "20px", padding: "10px", backgroundColor: "white", border: "1px solid #ccc" }}>
        <h3>Config Test:</h3>
        <pre>{JSON.stringify(configTest, null, 2)}</pre>
      </div>

      <div style={{ padding: "10px", backgroundColor: "yellow", border: "1px solid orange" }}>
        <strong>If you see this on Vercel, React is working!</strong>
        <br />
        Check the console for any errors.
      </div>
    </div>
  );
};

export default DebugApp;
