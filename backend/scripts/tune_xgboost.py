import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load feature-engineered data
df = pd.read_csv("data/featured/featured_data.csv")

# Ensure required columns exist
if "Close" not in df.columns:
    raise ValueError("Column 'Close' is missing from the dataset. Ensure feature engineering is correct.")

# Create 'Return' and 'Target' columns if not present
df["Return"] = df["Close"].pct_change()
df["Target"] = (df["Return"] > 0).astype(int)
df.dropna(inplace=True)  # Drop NaN rows caused by pct_change()

# Save the updated dataset
df.to_csv("data/featured/featured_data.csv", index=False)

# Define features and target, ensuring columns exist
drop_cols = [col for col in ["Price", "Target"] if col in df.columns]
X = df.drop(columns=drop_cols, errors='ignore')  # Drop only if present
y = df["Target"]

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")

# Feature Importance Plot
importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("XGBoost Feature Importance")
plt.show()

# Save model
model.save_model("models/fno_xgboost_model.json")
print("✅ Model saved: models/fno_xgboost_model.json")
