import yfinance as yf
ticker = yf.Ticker("^NSEI")  # Try "TCS.BO" if needed
data = ticker.history(period="1d")

if data.empty:
    print("No stock data found for TCS")
else:
    print(data)