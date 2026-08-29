import sys
import os
import joblib
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, LOCAL_DATA_DIR

MODEL_DIR = root_dir / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns used for model input
FEATURE_COLS = [
    "pm2_5", "pm10", "no2", "so2", "co", "ozone",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "aqi_lag_1", "aqi_lag_24", "aqi_lag_168",
    "aqi_rolling_mean_24", "aqi_rolling_std_24",
    "aqi_rolling_mean_168", "aqi_rolling_std_168",
    "aqi_change_rate_6h"
]

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

def load_dataset():
    """
    Load feature dataset from local storage or feature pipeline.
    """
    local_path = LOCAL_DATA_DIR / "features.parquet"
    if local_path.exists():
        df = pd.read_parquet(local_path)
    else:
        from src.feature_pipeline import run_feature_pipeline
        df = run_feature_pipeline(past_days=14)
    
    # Drop rows with NaN targets or features
    df_clean = df.dropna(subset=TARGET_COLS + FEATURE_COLS).reset_index(drop=True)
    return df_clean

def train_and_evaluate_candidates(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple ML model candidates.
    Returns comparison DataFrame and model dictionary.
    """
    candidates = {}
    
    # 1. Random Forest
    print("\n--- Training Candidate 1: Random Forest ---")
    rf_models = []
    rf_preds = np.zeros_like(y_test)
    for i, col in enumerate(TARGET_COLS):
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train[col])
        rf_preds[:, i] = rf.predict(X_test)
        rf_models.append(rf)
    candidates["RandomForest"] = {"models": rf_models, "preds": rf_preds}

    # 2. Ridge Regression
    print("--- Training Candidate 2: Ridge Regression ---")
    ridge_models = []
    ridge_preds = np.zeros_like(y_test)
    for i, col in enumerate(TARGET_COLS):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train[col])
        ridge_preds[:, i] = ridge.predict(X_test)
        ridge_models.append(ridge)
    candidates["RidgeRegression"] = {"models": ridge_models, "preds": ridge_preds}

    # 3. XGBoost (Optional import)
    try:
        import xgboost as xgb
        print("--- Training Candidate 3: XGBoost ---")
        xgb_models = []
        xgb_preds = np.zeros_like(y_test)
        for i, col in enumerate(TARGET_COLS):
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
            model.fit(X_train, y_train[col])
            xgb_preds[:, i] = model.predict(X_test)
            xgb_models.append(model)
        candidates["XGBoost"] = {"models": xgb_models, "preds": xgb_preds}
    except ImportError:
        print("XGBoost not installed. Skipping XGBoost candidate.")

    # 4. LightGBM (Optional import)
    try:
        import lightgbm as lgb
        print("--- Training Candidate 4: LightGBM ---")
        lgb_models = []
        lgb_preds = np.zeros_like(y_test)
        for i, col in enumerate(TARGET_COLS):
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
            model.fit(X_train, y_train[col])
            lgb_preds[:, i] = model.predict(X_test)
            lgb_models.append(model)
        candidates["LightGBM"] = {"models": lgb_models, "preds": lgb_preds}
    except ImportError:
        print("LightGBM not installed. Skipping LightGBM candidate.")

    # 5. PyTorch (Optional import)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class PyTorchMLP(nn.Module):
            def __init__(self, input_dim, output_dim=3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_dim)
                )
            def forward(self, x):
                return self.net(x)

        print("--- Training Candidate 5: PyTorch Neural Net (MLP) ---")
        X_tr_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train.values, dtype=torch.float32)
        X_te_t = torch.tensor(X_test.values, dtype=torch.float32)
        
        mlp = PyTorchMLP(input_dim=X_train.shape[1], output_dim=3)
        optimizer = optim.Adam(mlp.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        mlp.train()
        for epoch in range(150):
            optimizer.zero_grad()
            out = mlp(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            mlp_preds = mlp(X_te_t).numpy()
        candidates["PyTorchMLP"] = {"models": [mlp], "preds": mlp_preds}
    except ImportError:
        print("PyTorch not installed. Skipping PyTorch candidate.")

    # Compute metrics for each candidate
    results = []
    for name, data in candidates.items():
        preds = data["preds"]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({
            "model_name": name,
            "rmse": round(float(rmse), 3),
            "mae": round(float(mae), 3),
            "r2": round(float(r2), 3),
            "models_data": data["models"]
        })

    results_df = pd.DataFrame(results).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    return results_df

def save_and_register_best_model(results_df, feature_names):
    """
    Save winning model locally and register to Hopsworks Model Registry.
    """
    best_row = results_df.iloc[0]
    best_name = best_row["model_name"]
    best_models = best_row["models_data"]

    metrics = {
        "best_model": best_name,
        "rmse": best_row["rmse"],
        "mae": best_row["mae"],
        "r2": best_row["r2"],
        "feature_names": feature_names,
        "targets": TARGET_COLS
    }

    print("\n==================================================")
    print(f" WINNING MODEL: {best_name}")
    print(f" RMSE: {best_row['rmse']} | MAE: {best_row['mae']} | R2: {best_row['r2']}")
    print("==================================================")

    # Save local artifacts
    artifact_path = MODEL_DIR / "best_model.joblib"
    metrics_path = MODEL_DIR / "metrics.json"

    joblib.dump({"name": best_name, "models": best_models, "features": feature_names}, artifact_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved local model artifacts to {artifact_path}")

    # Register to Hopsworks if API key available
    if HOPSWORKS_API_KEY and HOPSWORKS_API_KEY != "your_hopsworks_api_key_here":
        try:
            import hopsworks
            print("Registering best model to Hopsworks Model Registry...")
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
            mr = project.get_model_registry()

            hw_model = mr.python.create_model(
                name="aether_aqi_predictor",
                metrics={"rmse": best_row["rmse"], "mae": best_row["mae"], "r2": best_row["r2"]},
                description=f"Winning Aether AQI Predictor: {best_name}"
            )
            hw_model.save(str(MODEL_DIR))
            print("Successfully registered model to Hopsworks Model Registry!")
        except Exception as e:
            print(f"Hopsworks Model Registration skipped/failed: {e}")

    return metrics

def run_training_pipeline():
    """
    Main training execution function.
    """
    df = load_dataset()
    print(f"Loaded training dataset with {len(df)} samples.")

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_df = train_and_evaluate_candidates(X_train, X_test, y_train, y_test)
    
    print("\n--- MODEL COMPARISON SUMMARY ---")
    print(results_df[["model_name", "rmse", "mae", "r2"]].to_string(index=False))

    metrics = save_and_register_best_model(results_df, FEATURE_COLS)
    return results_df, metrics

if __name__ == "__main__":
    run_training_pipeline()
