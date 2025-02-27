import os
import requests
import zipfile
import datetime

# Folder to save Bhavcopy data
RAW_DATA_DIR = "../data/raw/"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Function to get the last available trading day (skip weekends)
def get_last_trading_day():
    today = datetime.datetime.today()
    days_to_subtract = 1  # Start with yesterday

    while True:
        date = today - datetime.timedelta(days=days_to_subtract)
        if date.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
            days_to_subtract += 1
            continue
        return date

# Get last available trading day
last_trading_date = get_last_trading_day()
date_str = last_trading_date.strftime("%d%b%Y").upper()  # e.g., 26FEB2025

# Construct NSE URL
nse_url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{last_trading_date.strftime('%Y')}/{last_trading_date.strftime('%b').upper()}/fo{date_str}bhav.csv.zip"

# Path to save the ZIP file
zip_path = os.path.join(RAW_DATA_DIR, f"fo{date_str}.zip")

# Check if file exists on NSE server
response = requests.head(nse_url)

if response.status_code == 200:
    print(f"Downloading: {nse_url}")
    response = requests.get(nse_url, stream=True)
    with open(zip_path, "wb") as file:
        file.write(response.content)
    print(f"Downloaded successfully: {zip_path}")

    # Extract ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    print(f"Extracted to: {RAW_DATA_DIR}")

    # Delete ZIP file after extraction
    os.remove(zip_path)
    print("ZIP file deleted.")
else:
    print("File not available yet. Try again later.")