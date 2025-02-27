import os
import pandas as pd

# Define paths
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Find the latest Bhavcopy file
bhav_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")])
if not bhav_files:
    print("No Bhavcopy files found. Please download data first.")
    exit()

latest_file = os.path.join(RAW_DATA_DIR, bhav_files[-1])  # Use the most recent file
print(f"Processing file: {latest_file}")

# Load CSV file
df = pd.read_csv(latest_file)

# Display first few rows
print("Sample Data:")
print(df.head())

# Save cleaned file
cleaned_file = os.path.join(PROCESSED_DATA_DIR, "cleaned_bhavcopy.csv")
df.to_csv(cleaned_file, index=False)
print(f"Cleaned data saved to: {cleaned_file}")