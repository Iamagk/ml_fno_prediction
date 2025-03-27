import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the feature-engineered dataset
df = pd.read_csv("data/featured/featured_data.csv")

# Define features and target
features = [col for col in df.columns if col not in ["Price", "Target"]]  # Exclude non-numeric columns
target = "Target"

# Drop rows with missing values
df.dropna(inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
)

# Define XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_xgb = grid_search.best_estimator_

# Train the best model on full training data
best_xgb.fit(X_train, y_train)

# Predictions
y_pred = best_xgb.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Plot feature importance
xgb.plot_importance(best_xgb)
plt.title("Feature Importance")
plt.show()

# Save the trained model
model_path = "models/fno_xgboost_model.pkl"
joblib.dump(best_xgb, model_path)
print(f"🎯 Model saved to: {model_path}")