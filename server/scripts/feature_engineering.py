import pandas as pd
import numpy as np
import talib
import os

# Load the latest processed data file
data_dir = "data/processed"
files = sorted(os.listdir(data_dir), reverse=True)
latest_file = os.path.join(data_dir, files[0])

df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

# Standardize column names to uppercase (and remove spaces if any)
df.columns = [col.upper().replace(" ", "_") for col in df.columns]

# Now use uppercase column names everywhere
df['SMA_5'] = df['CLOSE'].rolling(window=5).mean()
df['SMA_10'] = df['CLOSE'].rolling(window=10).mean()

df['EMA_9'] = talib.EMA(df['CLOSE'], timeperiod=9)
df['EMA_21'] = talib.EMA(df['CLOSE'], timeperiod=21)
df['EMA_50'] = talib.EMA(df['CLOSE'], timeperiod=50)
df['EMA_200'] = talib.EMA(df['CLOSE'], timeperiod=200)

# Bollinger Bands
df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['CLOSE'], timeperiod=20)

# MACD and MACD Histogram
df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(df['CLOSE'], fastperiod=12, slowperiod=26, signalperiod=9)

# Relative Strength Index (RSI)
df['RSI_14'] = talib.RSI(df['CLOSE'], timeperiod=14)

# Stochastic Oscillator
df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['HIGH'], df['LOW'], df['CLOSE'])

# Average True Range (ATR)
df['ATR'] = talib.ATR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)

# Rate of Change (ROC)
df['ROC_10'] = talib.ROC(df['CLOSE'], timeperiod=10)

# On-Balance Volume (OBV)
df['OBV'] = talib.OBV(df['CLOSE'], df['CONTRACTS'])

# Volume Weighted Average Price (VWAP)
df['VWAP'] = (df['CLOSE'] * df['CONTRACTS']).cumsum() / df['CONTRACTS'].cumsum()

# Additional Indicators
df['ADX'] = talib.ADX(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
df['CCI'] = talib.CCI(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
df['WILLR_14'] = talib.WILLR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
df['MOM_10'] = talib.MOM(df['CLOSE'], timeperiod=10)
df['CMF'] = talib.ADOSC(df['HIGH'], df['LOW'], df['CLOSE'], df['CONTRACTS'], fastperiod=3, slowperiod=10)
df['PSAR'] = talib.SAR(df['HIGH'], df['LOW'], acceleration=0.02, maximum=0.2)
df['AROON_UP'], df['AROON_DOWN'] = talib.AROON(df['HIGH'], df['LOW'], timeperiod=14)

# Target Variable (Return)
df["RETURN"] = df["CLOSE"].pct_change()  # Daily percentage return

# Drop NaN values (especially first row)
df.dropna(inplace=True)

# Check for required column 'CONTRACTS'
if 'CONTRACTS' not in df.columns:
    raise ValueError("CONTRACTS column not found in data. Please check your Bhavcopy file.")

# Save the feature-engineered data
output_file = "data/featured/featured_data.csv"
df.to_csv(output_file, index=False)
print(f"✅ Feature engineering complete. Data saved to: {output_file}")
print(df.columns)