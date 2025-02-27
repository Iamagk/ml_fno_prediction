import os
import pandas as pd
import talib  # For technical indicators (install using: pip install ta-lib)

# Get the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/raw/")
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/processed/")

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Debugging: Check if the path exists
print("Checking directory:", os.path.abspath(RAW_DATA_DIR))
print("Directory exists:", os.path.exists(RAW_DATA_DIR))
print("Files inside:", os.listdir(RAW_DATA_DIR) if os.path.exists(RAW_DATA_DIR) else "Directory missing")

# List CSV files
csv_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")])
if not csv_files:
    print("❌ No data files found. Please run fetch_yfinance.py first.")
    exit()

# Pick the latest file
latest_file = os.path.join(RAW_DATA_DIR, csv_files[-1])
print(f"📄 Processing file: {latest_file}")

# Read CSV (Set first column as index)
df = pd.read_csv(latest_file, skiprows=[1], index_col=0)

# Print the first few rows to check column names
print("🔹 CSV Columns:", df.columns)

# Ensure index is DateTime
df.index = pd.to_datetime(df.index, errors="coerce")
df = df.dropna(subset=["Close"])  # Drop invalid rows

# Drop unnecessary columns
df.drop(columns=["Adj Close"], inplace=True, errors="ignore")

# Add Moving Averages
df["SMA_5"] = df["Close"].rolling(window=5).mean()  # 5-day moving average
df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-day moving average

# Add Relative Strength Index (RSI)
df["RSI_14"] = talib.RSI(df["Close"], timeperiod=14)

# Add Moving Average Convergence Divergence (MACD)
df["MACD"], df["MACD_Signal"], _ = talib.MACD(df["Close"])

# Drop rows with NaN values (from moving averages & RSI calculations)
df.dropna(inplace=True)

# Save processed file
processed_file = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
df.to_csv(processed_file)
print(f"✅ Processed data saved to: {processed_file}")

# Show sample processed data
print(df.head())