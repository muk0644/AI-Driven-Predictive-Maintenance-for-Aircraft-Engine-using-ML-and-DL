"""
Phase 8: Hyperparameter Tuning

This script performs hyperparameter tuning on the machine learning models
(XGBoost and LightGBM) using RandomizedSearchCV. Hyperparameter tuning
optimizes model parameters to improve prediction accuracy.

Tuning Strategy:
    - Method: RandomizedSearchCV (20 iterations)
    - Cross-Validation: 3-fold CV
    - Scoring Metric: RMSE (Root Mean Square Error)
    - Models: XGBoost and LightGBM

Parameters Tuned:
    - n_estimators: Number of boosting rounds (100-500)
    - max_depth: Maximum tree depth (5-15)
    - learning_rate: Step size shrinkage (0.01-0.2)
    - subsample: Fraction of samples per tree (0.6-1.0)  
    - colsample_bytree: Fraction of features per tree (0.6-1.0)
    - Additional regularization parameters

Usage:
    python 08_hyperparameter_tuning.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
import joblib
from scipy.stats import randint, uniform

print("="*50)
print("PHASE 7: HYPERPARAMETER TUNING")
print("="*50)

# STEP 1: Load processed data
print("\nLoading processed data...")
train_df = pd.read_csv('data/train_processed.csv')
feature_cols = joblib.load('models/feature_columns.pkl')

X = train_df[feature_cols]
y = train_df['RUL']

print(f" Data loaded: {X.shape}")

# STEP 2: Define Scoring Function
def rmse_scorer(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))

rmse_score = make_scorer(rmse_scorer, greater_is_better=True)

# MODEL 1: XGBoost Hyperparameter Tuning
print("\n" + "="*50)
print("TUNING XGBOOST")
print("="*50)

print("\n Original XGBoost Performance:")
original_xgb = pd.read_csv('results/model_comparison.csv')
original_xgb_rmse = original_xgb[original_xgb['Model'] == 'XGBoost']['Val RMSE'].values[0]
print(f"   RMSE: {original_xgb_rmse:.2f} cycles")

# Define parameter space
xgb_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5)
}

print("\n Parameter search space:")
for param, dist in xgb_param_dist.items():
    print(f"   {param}: {dist}")

# Initialize model
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Randomized search
print("\n Starting randomized search (20 iterations)...")
print("This will take 5-10 minutes...\n")

xgb_random = RandomizedSearchCV(
    xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=20,
    scoring=rmse_score,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

xgb_random.fit(X, y)

# Best parameters
print("\n Best XGBoost parameters found:")
for param, value in xgb_random.best_params_.items():
    print(f"   {param}: {value}")

tuned_xgb_rmse = -xgb_random.best_score_
improvement_xgb = ((original_xgb_rmse - tuned_xgb_rmse) / original_xgb_rmse) * 100

print(f"\n XGBoost Results:")
print(f"   Original RMSE: {original_xgb_rmse:.2f} cycles")
print(f"   Tuned RMSE: {tuned_xgb_rmse:.2f} cycles")
print(f"   Improvement: {improvement_xgb:.1f}%")

# Save tuned model
joblib.dump(xgb_random.best_estimator_, 'models/xgboost_tuned.pkl')
print("\n Saved: 'models/xgboost_tuned.pkl'")

# MODEL 2: LightGBM Hyperparameter Tuning
print("\n" + "="*50)
print("TUNING LIGHTGBM")
print("="*50)

print("\n Original LightGBM Performance:")
original_lgb_rmse = original_xgb[original_xgb['Model'] == 'LightGBM']['Val RMSE'].values[0]
print(f"   RMSE: {original_lgb_rmse:.2f} cycles")

# Define parameter space
lgb_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_samples': randint(10, 50),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

print("\n Parameter search space:")
for param, dist in lgb_param_dist.items():
    print(f"   {param}: {dist}")

# Initialize model
lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

# Randomized search
print("\n Starting randomized search (20 iterations)...")
print("This will take 5-10 minutes...\n")

lgb_random = RandomizedSearchCV(
    lgb_model,
    param_distributions=lgb_param_dist,
    n_iter=20,
    scoring=rmse_score,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

lgb_random.fit(X, y)

# Best parameters
print("\n Best LightGBM parameters found:")
for param, value in lgb_random.best_params_.items():
    print(f"   {param}: {value}")

tuned_lgb_rmse = -lgb_random.best_score_
improvement_lgb = ((original_lgb_rmse - tuned_lgb_rmse) / original_lgb_rmse) * 100

print(f"\n LightGBM Results:")
print(f"   Original RMSE: {original_lgb_rmse:.2f} cycles")
print(f"   Tuned RMSE: {tuned_lgb_rmse:.2f} cycles")
print(f"   Improvement: {improvement_lgb:.1f}%")

# Save tuned model
joblib.dump(lgb_random.best_estimator_, 'models/lightgbm_tuned.pkl')
print("\n Saved: 'models/lightgbm_tuned.pkl'")

# STEP 3: Save Tuning Results
print("\n" + "="*50)
print("SAVING TUNING RESULTS")
print("="*50)

tuning_results = pd.DataFrame({
    'Model': ['XGBoost (Original)', 'XGBoost (Tuned)', 'LightGBM (Original)', 'LightGBM (Tuned)'],
    'RMSE': [original_xgb_rmse, tuned_xgb_rmse, original_lgb_rmse, tuned_lgb_rmse],
    'Improvement': [0, improvement_xgb, 0, improvement_lgb]
})

print("\n" + tuning_results.to_string(index=False))

tuning_results.to_csv('results/hyperparameter_tuning_results.csv', index=False)
print("\n Saved: 'results/hyperparameter_tuning_results.csv'")

# Save best parameters
best_params = {
    'XGBoost': xgb_random.best_params_,
    'LightGBM': lgb_random.best_params_
}

import json
with open('results/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f, indent=4, default=str)
print(" Saved: 'results/best_hyperparameters.json'")

# SUMMARY
print("\n" + "="*50)
print("PHASE 7 COMPLETE! ")
print("="*50)
print(f"\n Hyperparameter Tuning Summary:")
print(f"\nXGBoost:")
print(f"  - Original: {original_xgb_rmse:.2f} cycles")
print(f"  - Tuned: {tuned_xgb_rmse:.2f} cycles")
print(f"  - Improvement: {improvement_xgb:.1f}%")
print(f"\nLightGBM:")
print(f"  - Original: {original_lgb_rmse:.2f} cycles")
print(f"  - Tuned: {tuned_lgb_rmse:.2f} cycles")
print(f"  - Improvement: {improvement_lgb:.1f}%")
print(f"\nFiles created:")
print(f"1. models/xgboost_tuned.pkl")
print(f"2. models/lightgbm_tuned.pkl")
print(f"3. results/hyperparameter_tuning_results.csv")
print(f"4. results/best_hyperparameters.json")
print(f"\n Next: Run Phase 8 (Final Comprehensive Comparison)")
print("="*50)