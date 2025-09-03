#!/bin/bash
# Full pipeline: retrain market direction, merge, then retrain option price model
set -e

# Paths
DATA_DIR="data/featured"
MODEL_DIR="models"
SCRIPTS_DIR="scripts"
FUTSTK_CSV="$DATA_DIR/featured_all_fno.csv"
OPTSTK_CSV="$DATA_DIR/featured_all_fno.csv"
MERGED_CSV="$DATA_DIR/optstk_with_direction.csv"
MARKET_MODEL="$MODEL_DIR/fno_ensemble_model.pkl"

# 1. Retrain market direction model
echo "[1/4] Retraining market direction model..."
rm -f "$MARKET_MODEL"
python $SCRIPTS_DIR/train_model.py

# 2. Extract OPTSTK and FUTSTK rows for merging
echo "[2/4] Preparing data for merge..."
awk -F, 'NR==1 || $1=="OPTSTK"' $OPTSTK_CSV > /tmp/optstk_only.csv
awk -F, 'NR==1 || $1=="FUTSTK"' $FUTSTK_CSV > /tmp/futstk_only.csv

# 3. Merge market direction into options data
echo "[3/4] Merging market direction into options data..."
python $SCRIPTS_DIR/merge_market_direction.py --options_csv /tmp/optstk_only.csv --futstk_csv /tmp/futstk_only.csv --model_path $MARKET_MODEL --output_csv $MERGED_CSV

echo "Merged options data with direction: $MERGED_CSV"

# 4. Retrain option price model using merged data
OPTION_PRICE_OUT="$DATA_DIR/optstk_with_prices.csv"
echo "[4/4] Retraining option price model..."
python $SCRIPTS_DIR/option_price_model.py --input_csv $MERGED_CSV --output_csv $OPTION_PRICE_OUT

echo "Pipeline complete. Option prices saved to: $OPTION_PRICE_OUT"
