import sys
import os
import json
import inspect
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Hopsworks 5.0.3 Backend Universal Compatibility Patch for Hopsworks 3.7.0 SDK
try:
    import humps
    import hsfs.feature_store
    import hsfs.feature_group
    import hsfs.feature
    import hsfs.core.job
    import hsfs.core.execution

    def _patch_from_json(cls):
        sig_params = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}
        
        @classmethod
        def _safe_from_response_json(c, json_dict):
            if json_dict is None:
                return None
            decamelized = humps.decamelize(json_dict)
            
            def _clean_item(item):
                if isinstance(item, dict):
                    # set defaults for missing positional params in FeatureStore if needed
                    item.setdefault("hive_endpoint", "")
                    item.setdefault("hdfs_store_path", "")
                    item.setdefault("featurestore_description", "")
                    item.setdefault("inode_id", 0)
                    item.setdefault("offline_featurestore_name", "")
                    item.setdefault("online_featurestore_name", "")
                    cleaned = {k: v for k, v in item.items() if k in sig_params}
                    return c(**cleaned)
                return item

            if isinstance(decamelized, list):
                return [_clean_item(x) for x in decamelized]
            elif isinstance(decamelized, dict):
                if "items" in decamelized and isinstance(decamelized["items"], list):
                    return [_clean_item(x) for x in decamelized["items"]]
                return _clean_item(decamelized)
            return None

        cls.from_response_json = _safe_from_response_json

    _patch_from_json(hsfs.feature.Feature)
    _patch_from_json(hsfs.feature_group.FeatureGroup)
    _patch_from_json(hsfs.feature_store.FeatureStore)
    _patch_from_json(hsfs.core.job.Job)
    _patch_from_json(hsfs.core.execution.Execution)

    # Patch FeatureGroup update_from_response_json
    _fg_params = set(inspect.signature(hsfs.feature_group.FeatureGroup.__init__).parameters.keys()) - {'self'}
    _orig_fg_update = hsfs.feature_group.FeatureGroup.update_from_response_json

    def _safe_fg_update(self, json_dict):
        decamelized = humps.decamelize(json_dict)
        cleaned = {k: v for k, v in decamelized.items() if k in _fg_params} if isinstance(decamelized, dict) else decamelized
        return _orig_fg_update(self, cleaned)

    hsfs.feature_group.FeatureGroup.update_from_response_json = _safe_fg_update

except Exception as e:
    print(f"Compatibility patch notice: {e}")

import pandas as pd
import numpy as np
from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, HOPSWORKS_HOST, LOCAL_DATA_DIR
from src.data_fetcher import fetch_all_cities_data

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based, cyclical annual/diurnal, lag, rolling, and target variables for 1-year AQI forecasting.
    """
    df = df.sort_values(by=["city", "time"]).reset_index(drop=True)

    dt = df["time"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["day_of_year"] = dt.dayofyear

    # Cyclical hour, month, and annual day-of-year embeddings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    dfs_processed = []
    for city_name, group in df.groupby("city"):
        group = group.copy().sort_values(by="time")

        # 1-hour, 24-hour (daily), and 168-hour (weekly) lags
        group["aqi_lag_1"] = group["aqi"].shift(1)
        group["aqi_lag_24"] = group["aqi"].shift(24)
        group["aqi_lag_168"] = group["aqi"].shift(168)

        # 24-hour and 7-day rolling statistics
        group["aqi_rolling_mean_24"] = group["aqi"].shift(1).rolling(window=24, min_periods=1).mean()
        group["aqi_rolling_std_24"] = group["aqi"].shift(1).rolling(window=24, min_periods=1).std().fillna(0)
        group["aqi_rolling_mean_168"] = group["aqi"].shift(1).rolling(window=168, min_periods=1).mean()
        group["aqi_rolling_std_168"] = group["aqi"].shift(1).rolling(window=168, min_periods=1).std().fillna(0)

        # Rate of change
        group["aqi_change_rate_6h"] = (group["aqi"].shift(1) - group["aqi"].shift(7)) / 6.0

        # Multi-horizon forecast targets (24h, 48h, 72h)
        group["target_aqi_24h"] = group["aqi"].shift(-24)
        group["target_aqi_48h"] = group["aqi"].shift(-48)
        group["target_aqi_72h"] = group["aqi"].shift(-72)

        dfs_processed.append(group)

    result_df = pd.concat(dfs_processed, ignore_index=True)
    result_df = result_df.bfill().ffill()
    
    result_df["time_str"] = result_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result_df["timestamp_ms"] = (result_df["time"].astype("int64") // 10**6).astype("int64")

    return result_df

def save_features_to_hopsworks_or_local(df: pd.DataFrame):
    """
    Save engineered features to Hopsworks Feature Store catalog (aqi_weather_fg v1) or local parquet fallback.
    """
    saved_to_hopsworks = False
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    host = os.getenv("HOPSWORKS_HOST", "").strip()

    if api_key and api_key != "your_hopsworks_api_key_here":
        try:
            print(f"Connecting to Hopsworks Cloud Feature Store ({host})...")
            import hopsworks
            
            # Login to Hopsworks Cloud
            login_kwargs = {"api_key_value": api_key, "project": HOPSWORKS_PROJECT_NAME}
            if host and host != "c.app.hopsworks.ai":
                login_kwargs["host"] = host

            project = hopsworks.login(**login_kwargs)
            fs = project.get_feature_store()
            
            # Fetch existing or create new Feature Group entity
            try:
                aqi_fg = fs.get_feature_group(name="aqi_weather_fg", version=1)
                print("Found existing Feature Group 'aqi_weather_fg' version 1.")
            except Exception:
                print("Creating Feature Group 'aqi_weather_fg' version 1...")
                aqi_fg = fs.create_feature_group(
                    name="aqi_weather_fg",
                    version=1,
                    primary_key=["city", "time_str"],
                    event_time="timestamp_ms",
                    description="Live AQI feature pipeline for Karachi, Chicago, Sydney, Austria"
                )
                aqi_fg.save(df)
            
            # Insert feature records into Hopsworks Feature Store
            aqi_fg.insert(df, write_options={"wait_for_job": False})
            print(f"\n[SUCCESS] Pushed {len(df)} feature records to Hopsworks Feature Group 'aqi_weather_fg'!")
            print(f"Check your Hopsworks Dashboard: https://{host}/p/{project.id}/fs/featuregroups")
            saved_to_hopsworks = True

        except Exception as e:
            print(f"Hopsworks Cloud Notice: {e}")
            print("\n------------------------------------------------------------------------")
            print(" TO POPULATE HOPSWORKS CLOUD HOME DASHBOARD:")
            print(" 1. Open Hopsworks Cloud (https://eu-west.cloud.hopsworks.ai/p/41197/view)")
            print(" 2. Click User Profile (top right) -> Account Settings -> API Keys")
            print(" 3. Create a key with ALL checkboxes checked")
            print(" 4. Paste key into D:\\dev\\Aether\\.env under HOPSWORKS_API_KEY")
            print("------------------------------------------------------------------------\n")
    else:
        print("\n=========================================================================")
        print(" HOPSWORKS API KEY PENDING IN .env FILE:")
        print(" Paste your Hopsworks API Key into D:\\dev\\Aether\\.env under:")
        print(" HOPSWORKS_API_KEY=your_actual_key")
        print("=========================================================================\n")

    # Local parquet fallback
    local_path = LOCAL_DATA_DIR / "features.parquet"
    df.to_parquet(local_path, index=False)
    print(f"Saved {len(df)} feature records locally to: {local_path}")
    return saved_to_hopsworks

def run_feature_pipeline(past_days: int = 7):
    """
    Full feature pipeline execution.
    """
    print(f"Executing feature pipeline for past {past_days} days...")
    raw_df = fetch_all_cities_data(past_days=past_days)
    feature_df = engineer_features(raw_df)
    save_features_to_hopsworks_or_local(feature_df)
    return feature_df

if __name__ == "__main__":
    df = run_feature_pipeline(past_days=7)
    print(f"Feature pipeline complete! Processed {len(df)} rows across {df['city'].nunique()} cities.")
