import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the model

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURED_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/featured/")
MODEL_DIR = os.path.join(SCRIPT_DIR, "../models/")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
file_path = os.path.join(FEATURED_DATA_DIR, "featured_data.csv")
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Define target variable (Buy/Sell)
df["Target"] = np.where(df["Return"] > 0, 1, 0)  # 1 = Buy, 0 = Sell

# Drop columns that should not be used as features
X = df.drop(columns=["Target", "Return"])
y = df["Target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model
model_path = os.path.join(MODEL_DIR, "fno_prediction_model.pkl")
joblib.dump(model, model_path)
print(f"🎯 Model saved to: {model_path}")