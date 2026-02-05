"""
Phase 3: Machine Learning Baseline Models

This script trains three traditional machine learning models on the preprocessed data:
1. Random Forest - Ensemble of decision trees
2. XGBoost - Gradient boosting with regularization
3. LightGBM - Fast gradient boosting variant

These models serve as baselines to compare against deep learning approaches.

Usage:
    python 03_ml_baseline.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

print("Phase 3: Machine Learning Baseline Models")
print("=" * 50)

# Load preprocessed data
print("\nLoading preprocessed data...")
train_df = pd.read_csv('data/train_processed.csv')
feature_cols = joblib.load('models/feature_columns.pkl')

print(f"Data loaded: {train_df.shape}")
print(f"Features: {len(feature_cols)}")

# Prepare features and target variable
X = train_df[feature_cols]
y = train_df['RUL']

# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")

# Model 1: Random Forest
print("\n" + "=" * 50)
print("Training Model 1: Random Forest")
print("=" * 50)

print("\nRandom Forest is an ensemble of decision trees that reduces overfitting...")
rf_model = RandomForestRegressor(
    n_estimators=100,        # Number of trees in the forest
    max_depth=20,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples in leaf node
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    verbose=1
)

rf_model.fit(X_train, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)

# Metrics
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_r2 = r2_score(y_train, rf_train_pred)

rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
rf_val_mae = mean_absolute_error(y_val, rf_val_pred)
rf_val_r2 = r2_score(y_val, rf_val_pred)

print(f"\n Random Forest Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {rf_train_rmse:.2f} cycles")
print(f"  MAE:  {rf_train_mae:.2f} cycles")
print(f"  R²:   {rf_train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {rf_val_rmse:.2f} cycles")
print(f"  MAE:  {rf_val_mae:.2f} cycles")
print(f"  R²:   {rf_val_r2:.4f}")

# Save model
joblib.dump(rf_model, 'models/random_forest.pkl')
print("\n Model saved: 'models/random_forest.pkl'")

# Model 2:XGBoost
print("\n" + "=" * 50)
print("Training Model 2: XGBoost")
print("=" * 50)

print("\nXGBoost uses gradient boosting with advanced regularization techniques...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,        # Number of boosting rounds
    max_depth=10,            # Maximum tree depth
    learning_rate=0.1,       # Step size shrinkage
    subsample=0.8,           # Fraction of samples per tree
    colsample_bytree=0.8,    # Fraction of features per tree
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# Predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)

# Metrics
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
xgb_train_r2 = r2_score(y_train, xgb_train_pred)

xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
xgb_val_r2 = r2_score(y_val, xgb_val_pred)

print(f"\n XGBoost Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {xgb_train_rmse:.2f} cycles")
print(f"  MAE:  {xgb_train_mae:.2f} cycles")
print(f"  R²:   {xgb_train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {xgb_val_rmse:.2f} cycles")
print(f"  MAE:  {xgb_val_mae:.2f} cycles")
print(f"  R²:   {xgb_val_r2:.4f}")

# Save model
joblib.dump(xgb_model, 'models/xgboost.pkl')
print("\n Model saved: 'models/xgboost.pkl'")

# Model 3: LightGBM
print("\n" + "=" * 50)
print("Training Model 3: LightGBM")
print("=" * 50)

print("\nLightGBM is an efficient gradient boosting framework with fast training...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1                # Suppress training logs
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

# Predictions
lgb_train_pred = lgb_model.predict(X_train)
lgb_val_pred = lgb_model.predict(X_val)

# Metrics
lgb_train_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_pred))
lgb_train_mae = mean_absolute_error(y_train, lgb_train_pred)
lgb_train_r2 = r2_score(y_train, lgb_train_pred)

lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)
lgb_val_r2 = r2_score(y_val, lgb_val_pred)

print(f"\n LightGBM Results:")
print(f"\nTraining Set:")
print(f"  RMSE: {lgb_train_rmse:.2f} cycles")
print(f"  MAE:  {lgb_train_mae:.2f} cycles")
print(f"  R²:   {lgb_train_r2:.4f}")
print(f"\nValidation Set:")
print(f"  RMSE: {lgb_val_rmse:.2f} cycles")
print(f"  MAE:  {lgb_val_mae:.2f} cycles")
print(f"  R²:   {lgb_val_r2:.4f}")

# Save model
joblib.dump(lgb_model, 'models/lightgbm.pkl')
print("\n Model saved: 'models/lightgbm.pkl'")

# Create comparison visualizations
print("\n" + "=" * 50)
print("Creating Visualizations")
print("=" * 50)

# Create figure with 3 subplots for comparing predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Random Forest
axes[0].scatter(y_val, rf_val_pred, alpha=0.5, s=10)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('True RUL (cycles)', fontsize=11)
axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[0].set_title(f'Random Forest\nRMSE: {rf_val_rmse:.2f} | R²: {rf_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# XGBoost
axes[1].scatter(y_val, xgb_val_pred, alpha=0.5, s=10, color='orange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('True RUL (cycles)', fontsize=11)
axes[1].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[1].set_title(f'XGBoost\nRMSE: {xgb_val_rmse:.2f} | R²: {xgb_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# LightGBM
axes[2].scatter(y_val, lgb_val_pred, alpha=0.5, s=10, color='green')
axes[2].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[2].set_xlabel('True RUL (cycles)', fontsize=11)
axes[2].set_ylabel('Predicted RUL (cycles)', fontsize=11)
axes[2].set_title(f'LightGBM\nRMSE: {lgb_val_rmse:.2f} | R²: {lgb_val_r2:.3f}', 
                  fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/ml_predictions.png', dpi=300, bbox_inches='tight')
print("\nSaved: 'results/ml_predictions.png'")
plt.close()

# Create feature importance visualization using XGBoost
print("\nCreating feature importance chart...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Most Important Features (XGBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: 'results/feature_importance.png'")
plt.close()

# Create model comparison table
print("\n" + "=" * 50)
print("Model Comparison Summary")
print("=" * 50)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'Train RMSE': [rf_train_rmse, xgb_train_rmse, lgb_train_rmse],
    'Val RMSE': [rf_val_rmse, xgb_val_rmse, lgb_val_rmse],
    'Train MAE': [rf_train_mae, xgb_train_mae, lgb_train_mae],
    'Val MAE': [rf_val_mae, xgb_val_mae, lgb_val_mae],
    'Val R²': [rf_val_r2, xgb_val_r2, lgb_val_r2]
})

print("\n" + comparison.to_string(index=False))

# Save comparison results to CSV
comparison.to_csv('results/model_comparison.csv', index=False)
print("\nSaved: 'results/model_comparison.csv'")

# Display summary
print("\n" + "=" * 50)
print("Phase 3 Complete!")
print("=" * 50)

best_model = comparison.loc[comparison['Val RMSE'].idxmin(), 'Model']
best_rmse = comparison['Val RMSE'].min()

print(f"\nBest Model: {best_model}")
print(f"Best Validation RMSE: {best_rmse:.2f} cycles")
print(f"\nFiles created:")
print(f"  1. models/random_forest.pkl")
print(f"  2. models/xgboost.pkl")
print(f"  3. models/lightgbm.pkl")
print(f"  4. results/ml_predictions.png")
print(f"  5. results/feature_importance.png")
print(f"  6. results/model_comparison.csv")
print(f"\nNext step: Run 04_deep_learning_lstm.py for deep learning models")
print("=" * 50)