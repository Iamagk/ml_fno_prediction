import yfinance as yf
ticker = yf.Ticker("^NSEI")  # Try "TCS.BO" if needed
data = ticker.history(period="3y")

if data.empty:
    print("No stock data found for nifty")
else:
    print(data)


