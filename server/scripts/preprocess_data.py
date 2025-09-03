import os
import pandas as pd

# Define paths

RAW_DATA_FILE = "data/raw/ALL_FNO_data_latest.csv"
PROCESSED_DATA_DIR = "server/data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

if not os.path.exists(RAW_DATA_FILE):
    print(f"Raw data file not found: {RAW_DATA_FILE}")
    exit()

print(f"Processing file: {RAW_DATA_FILE}")
df = pd.read_csv(RAW_DATA_FILE)
print("Sample Data:")
print(df.head())

cleaned_file = os.path.join(PROCESSED_DATA_DIR, "cleaned_all_fno.csv")
df.to_csv(cleaned_file, index=False)
print(f"Cleaned data saved to: {cleaned_file}")