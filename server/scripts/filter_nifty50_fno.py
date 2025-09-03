import os
import pandas as pd

RAW_DATA_DIR = "server/data/raw/"
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "NSEI_data_latest.csv")

# Columns to keep
COLUMNS = [
    "INSTRUMENT","SYMBOL","EXPIRY_DT","STRIKE_PR","OPTION_TYP","OPEN","HIGH","LOW","CLOSE",
    "SETTLE_PR","CONTRACTS","VAL_INLAKH","OPEN_INT","CHG_IN_OI","TIMESTAMP"
]

all_dfs = []
for fname in os.listdir(RAW_DATA_DIR):
    if fname.endswith("bhav.csv"):
        fpath = os.path.join(RAW_DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            # Filter for NIFTY50 futures and options only
            df = df[(df["SYMBOL"] == "NIFTY") & (df["INSTRUMENT"].isin(["FUTIDX", "OPTIDX"]))]
            df = df[COLUMNS]
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {fname}: {e}")

if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Filtered NIFTY50 F&O data saved to {OUTPUT_FILE}")
else:
    print("No NIFTY50 F&O data found to merge.")