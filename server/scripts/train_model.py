from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
# Add imports for safe parallelism and warning control
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning
import warnings
# Silence convergence warnings from unsuccessful LR candidates
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURED_DATA_DIR = os.path.join(SCRIPT_DIR, "../data/featured/")
MODEL_DIR = os.path.join(SCRIPT_DIR, "../models/")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "fno_ensemble_model.pkl")

if os.path.exists(MODEL_PATH):
    print(f"Model file already exists at {MODEL_PATH}. Skipping training.")
    exit(0)

# Load dataset
file_path = os.path.join(FEATURED_DATA_DIR, "featured_all_fno.csv")
if not os.path.exists(file_path):
    print(f"Data file not found: {file_path}. Skipping model training.")
    exit(0)

df = pd.read_csv(file_path)

# Filter for FUTSTK data only (market direction prediction)
futstk_data = df[df['INSTRUMENT'] == 'FUTSTK'].copy()
print(f"Number of training rows (FUTSTK): {len(futstk_data)}")

if len(futstk_data) == 0:
    print("No FUTSTK data found. Exiting.")
    exit(1)

# Define target variable (market direction: up/down)
futstk_data["Target"] = np.where(futstk_data["CLOSE"] > futstk_data["OPEN"], 1, 0)  # 1 = Up, 0 = Down

# Use available feature columns
FEATURE_NAMES = [
    "STRIKE_PR", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI",
    "SMA_5", "SMA_10", "RSI_14", "MACD", "MACD_SIGNAL", "EMA_9", "EMA_21", "EMA_50", "EMA_200",
    "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "MACD_HIST", "STOCH_K", "STOCH_D", "ATR", "ROC_10", "OBV",
    "VWAP", "ADX", "CCI", "WILLR_14", "MOM_10", "CMF", "PSAR", "AROON_UP", "AROON_DOWN"
]

# Filter for actually available features
available_features = [col for col in FEATURE_NAMES if col in futstk_data.columns]
print(f"Using {len(available_features)} features: {available_features}")

X = futstk_data[available_features].fillna(0).astype(float)
y = futstk_data["Target"]

print(f"Class distribution: {y.value_counts().to_dict()}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models with hyperparameter tuning for best performance...")

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
rf_base = RandomForestClassifier(random_state=42, n_jobs=1)
# Use all cores for GridSearchCV (threaded backend avoids macOS resource tracker issue)
rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Best Random Forest params: {rf_grid.best_params_}")
print(f"Random Forest accuracy: {rf_accuracy:.4f}")

# Hyperparameter tuning for LightGBM
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
lgbm_grid = GridSearchCV(lgbm_base, lgbm_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
lgbm_grid.fit(X_train, y_train)
lgbm_model = lgbm_grid.best_estimator_
lgbm_pred = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_pred)
print(f"Best LightGBM params: {lgbm_grid.best_params_}")
print(f"LightGBM accuracy: {lgbm_accuracy:.4f}")

# Hyperparameter tuning for CatBoost
cat_param_grid = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}
cat_base = CatBoostClassifier(verbose=0, random_state=42)
cat_grid = GridSearchCV(cat_base, cat_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
cat_grid.fit(X_train, y_train)
catboost_model = cat_grid.best_estimator_
cat_pred = catboost_model.predict(X_test)
cat_accuracy = accuracy_score(y_test, cat_pred)
print(f"Best CatBoost params: {cat_grid.best_params_}")
print(f"CatBoost accuracy: {cat_accuracy:.4f}")

# Hyperparameter tuning for Logistic Regression
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
# Increase max_iter to reduce LR warnings during tuning
lr_base = LogisticRegression(max_iter=5000, random_state=42)
lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
lr_grid.fit(X_train_scaled, y_train)
logreg_model = lr_grid.best_estimator_
lr_pred = logreg_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Best Logistic Regression params: {lr_grid.best_params_}")
print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")

# Create ensemble using the best models
ensemble_models = [
    ('rf', rf_model),
    ('lgbm', lgbm_model),
    ('catboost', catboost_model)
]

ensemble_model = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'
)
ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"✅ Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
print(classification_report(y_test, ensemble_pred))

# Enhanced ensemble wrapper class
class OptimizedEnsemble:
    def __init__(self, ensemble_model, scaler, feature_names, individual_models=None):
        self.ensemble_model = ensemble_model
        self.scaler = scaler
        self.feature_names = feature_names
        self.individual_models = individual_models or {}
    
    def predict(self, X):
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X):
        return self.ensemble_model.predict_proba(X)

# Create final optimized ensemble
final_ensemble = OptimizedEnsemble(
    ensemble_model, 
    scaler, 
    available_features,
    individual_models={
        'rf': rf_model,
        'lgbm': lgbm_model,
        'catboost': catboost_model,
        'logreg': logreg_model
    }
)

# Save models
joblib.dump(rf_model, os.path.join(MODEL_DIR, "fno_rf_model.pkl"))
joblib.dump(final_ensemble, os.path.join(MODEL_DIR, "fno_ensemble_model.pkl"))
print(f"🎯 Models saved to: {MODEL_DIR}")
print(f"✅ Training completed successfully with optimized hyperparameters!")