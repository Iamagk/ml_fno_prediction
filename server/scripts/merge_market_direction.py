
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import joblib

def merge_market_direction(options_csv, futstk_csv, model_path, output_csv):
    """
    Adds market direction prediction as a feature to options data.
    options_csv: path to options (OPTSTK/OPTIDX) data
    futstk_csv: path to FUTSTK data (for features)
    model_path: path to trained market direction model (ensemble)
    output_csv: path to save merged options data with direction feature
    """
    # Load data
    df_opt = pd.read_csv(options_csv)
    df_fut = pd.read_csv(futstk_csv)
    model = joblib.load(model_path)

    # Use same features as in training
    FEATURE_NAMES = [
        "STRIKE_PR", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI",
        "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_SIGNAL", "EMA_9", "EMA_21", "EMA_50", "EMA_200",
        "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "MACD_HIST", "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV",
        "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10", "CMF", "PSAR", "AROON_UP", "AROON_DOWN"
    ]

    # Predict market direction for each FUTSTK row
    X_fut = df_fut[FEATURE_NAMES].astype(float)
    direction_pred = model.predict(X_fut)
    df_fut["MARKET_DIRECTION"] = direction_pred

    # Merge direction into options data by SYMBOL and TIMESTAMP
    merge_cols = ["SYMBOL", "TIMESTAMP"]
    df_opt = df_opt.merge(
        df_fut[merge_cols + ["MARKET_DIRECTION"]],
        on=merge_cols,
        how="left"
    )

    # Save merged file
    df_opt.to_csv(output_csv, index=False)
    print(f"Merged options data with market direction saved to {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge market direction into options data.")
    parser.add_argument("--options_csv", required=True, help="Path to options data CSV (OPTSTK/OPTIDX)")
    parser.add_argument("--futstk_csv", required=True, help="Path to FUTSTK data CSV")
    parser.add_argument("--model_path", required=True, help="Path to trained market direction model (pkl)")
    parser.add_argument("--output_csv", required=True, help="Path to save merged output CSV")
    args = parser.parse_args()
    merge_market_direction(args.options_csv, args.futstk_csv, args.model_path, args.output_csv)
