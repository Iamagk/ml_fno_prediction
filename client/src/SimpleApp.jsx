import React, { useState } from "react";
import config from "./config";

console.log("SimpleApp loading...", config);

const SimpleApp = () => {
  const [message, setMessage] = useState("Loading...");

  React.useEffect(() => {
    console.log("SimpleApp mounted");
    setMessage("App is working! API URL: " + config.API_URL);
  }, []);

  return (
    <div style={{ padding: "20px", backgroundColor: "#1e1e2f", color: "white", minHeight: "100vh" }}>
      <h1>Simple Test App</h1>
      <p>{message}</p>
      <p>If you can see this, React and the build are working correctly.</p>
      <p>Environment: {process.env.NODE_ENV}</p>
    </div>
  );
};

export default SimpleApp;
