import os
import yfinance as yf
import pandas as pd

# Define where to save data
RAW_DATA_DIR = "../data/raw/"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Choose a stock or index (e.g., NIFTY 50 = ^NSEI)
symbol = "^NSEI"  # Change this to any stock (e.g., "RELIANCE.NS" for Reliance)
start_date = "2020-01-01"  # Adjust the date range as needed
end_date = "2025-06-04"

# Fetch data
print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
data = yf.download(symbol, start=start_date, end=end_date)

# Save data to CSV
csv_path = os.path.join(RAW_DATA_DIR, f"{symbol}_data.csv")
data.to_csv(csv_path)
print(f"Data saved to: {csv_path}")

# Show sample data
print(data.head())