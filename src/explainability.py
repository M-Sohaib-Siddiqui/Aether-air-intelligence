import sys
import joblib
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from src.train import MODEL_DIR, FEATURE_COLS

def get_shap_feature_importance(sample_df: pd.DataFrame = None):
    """
    Compute SHAP feature importance values for the winning model, with graceful fallback.
    Returns sorted list of feature names and mean absolute importance values.
    """
    model_path = MODEL_DIR / "best_model.joblib"
    if not model_path.exists():
        # Default fallback importance if no model is trained yet
        return pd.DataFrame({
            "feature": ["PM2.5", "AQI Lag 24h", "Humidity", "Temperature", "PM10", "Wind Speed"],
            "importance": [0.45, 0.25, 0.12, 0.08, 0.06, 0.04]
        })

    artifact = joblib.load(model_path)
    models = artifact["models"]
    first_model = models[0] if isinstance(models, list) else models

    if sample_df is None:
        local_path = root_dir / "data" / "features.parquet"
        if local_path.exists():
            sample_df = pd.read_parquet(local_path)

    # Try computing SHAP if shap package is installed
    try:
        import shap
        if sample_df is not None:
            X_sample = sample_df[FEATURE_COLS].dropna().head(100)
            explainer = shap.TreeExplainer(first_model)
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values[0])
                
            mean_shap = np.abs(shap_values).mean(axis=0)
        else:
            raise ValueError("No sample df for SHAP")
    except Exception as e:
        # Fallback to feature_importances_ if SHAP is missing or fails
        if hasattr(first_model, "feature_importances_"):
            mean_shap = first_model.feature_importances_
        else:
            mean_shap = np.ones(len(FEATURE_COLS)) / len(FEATURE_COLS)

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": mean_shap
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df

if __name__ == "__main__":
    df_shap = get_shap_feature_importance()
    print("\n--- FEATURE IMPORTANCE ---")
    print(df_shap.to_string(index=False))
