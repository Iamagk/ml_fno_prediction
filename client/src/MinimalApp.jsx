import React from "react";

const MinimalApp = () => {
  return React.createElement("div", {
    style: {
      backgroundColor: "red",
      color: "white", 
      padding: "50px",
      fontSize: "24px",
      textAlign: "center",
      minHeight: "100vh"
    }
  }, "MINIMAL TEST - If you see this, React works!");
};

export default MinimalApp;
