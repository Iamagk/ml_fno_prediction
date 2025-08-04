const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

export const predictMarket = async (features) => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ features }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error calling API:", error);
    return null;
  }
};

export const predictLive = async (symbol) => {
  try {
    const response = await fetch(`${API_URL}/predict_live?symbol=${symbol}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error calling predict_live API:", error);
    return null;
  }
};

export const predictWithOptions = async (symbol) => {
  try {
    const response = await fetch(`${API_URL}/predict_with_options?symbol=${symbol}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error calling predict_with_options API:", error);
    return null;
  }
};

export const fetchYfinanceData = async (symbol) => {
  try {
    const response = await fetch(`${API_URL}/fetch_yfinance?symbol=${symbol}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error calling fetch_yfinance API:", error);
    return null;
  }
};