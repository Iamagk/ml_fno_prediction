import os
import requests
import zipfile
import datetime
from tqdm import tqdm

# Folder to save Bhavcopy data
RAW_DATA_DIR = "../data/raw/"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
print(f"Saving files to: {os.path.abspath(RAW_DATA_DIR)}")

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + datetime.timedelta(n)

def is_weekday(date):
    return date.weekday() < 5  # 0-4 are Mon-Fri

# Set your date range (last 3 years)
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=5*365)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Referer": "https://www.nseindia.com/",
    "Cookie": "paste_your_cookie_here"
}

for single_date in tqdm(list(daterange(start_date, end_date))):
    if not is_weekday(single_date):
        continue
    day = single_date.strftime("%d")
    month_abbr = single_date.strftime("%b").upper()
    year = single_date.strftime("%Y")
    date_str = f"{day}{month_abbr}{year}"
    nse_url = (
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
        f"{year}/{month_abbr}/fo{date_str}bhav.csv.zip"
    )
    zip_path = os.path.join(RAW_DATA_DIR, f"fo{date_str}.zip")
    csv_path = os.path.join(RAW_DATA_DIR, f"fo{date_str}bhav.csv")
    if os.path.exists(csv_path):
        continue

    try:
        session = requests.Session()
        response = session.head(nse_url, headers=HEADERS, allow_redirects=True, timeout=10)
        print(f"{nse_url} -> {response.status_code}")
        if response.status_code == 200:
            print(f"Downloading: {nse_url}")
            response = session.get(nse_url, stream=True, headers=HEADERS, timeout=30)
            with open(zip_path, "wb") as file:
                file.write(response.content)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            os.remove(zip_path)
        else:
            print(f"Not available: {nse_url}")
            continue
    except Exception as e:
        print(f"Error downloading {nse_url}: {e}")
        continue