import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load feature-engineered data
df = pd.read_csv("data/featured/featured_data.csv")

# Standardize columns to uppercase for consistency
df.columns = [col.upper() for col in df.columns]

# Ensure required columns exist
if "CLOSE" not in df.columns:
    raise ValueError("Column 'CLOSE' is missing from the dataset. Ensure feature engineering is correct.")

# Create 'TARGET' column if not present
if "TARGET" not in df.columns:
    if "RETURN" not in df.columns:
        df["RETURN"] = df["CLOSE"].pct_change()
    df["TARGET"] = (df["RETURN"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

# Save the updated dataset
df.to_csv("data/featured/featured_data.csv", index=False)

# Define features and target, ensuring columns exist
drop_cols = [col for col in ["PRICE", "TARGET", "TIMESTAMP"] if col in df.columns]
X = df.drop(columns=drop_cols, errors='ignore')
y = df["TARGET"]

# Drop non-numeric columns
X = X.select_dtypes(include=[float, int, bool])

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Initialize and train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Print feature names used for training
print("Features used for training:", list(X.columns))

# Feature Importance Plot
importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# Save model
os.makedirs("models", exist_ok=True)
model.save_model("models/fno_xgboost_model.json")
print("✅ Model saved: models/fno_xgboost_model.json")