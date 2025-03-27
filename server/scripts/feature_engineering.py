import pandas as pd
import numpy as np
import talib
import os

# Load the latest processed data file
data_dir = "data/processed"
files = sorted(os.listdir(data_dir), reverse=True)
latest_file = os.path.join(data_dir, files[0])

df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

# Ensure numeric data
df = df.apply(pd.to_numeric, errors='coerce')

# Moving Averages
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()

df['EMA_9'] = talib.EMA(df['Close'], timeperiod=9)
df['EMA_21'] = talib.EMA(df['Close'], timeperiod=21)
df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
df['EMA_200'] = talib.EMA(df['Close'], timeperiod=200)

# Bollinger Bands
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)

# MACD and MACD Histogram
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Relative Strength Index (RSI)
df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

# Stochastic Oscillator
df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])

# Average True Range (ATR)
df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

# Rate of Change (ROC)
df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)

# On-Balance Volume (OBV)
df['OBV'] = talib.OBV(df['Close'], df['Volume'])

# Volume Weighted Average Price (VWAP)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

# Additional Indicators
df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
df['WILLR_14'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['MOM_10'] = talib.MOM(df['Close'], timeperiod=10)
df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
df['PSAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['High'], df['Low'], timeperiod=14)

# Target Variable (Return)
df["Return"] = df["Close"].pct_change()  # Daily percentage return

# Drop NaN values (especially first row)
df.dropna(inplace=True)

# Save the feature-engineered data
output_file = "data/featured/featured_data.csv"
df.to_csv(output_file, index=False)
print(f"✅ Feature engineering complete. Data saved to: {output_file}")