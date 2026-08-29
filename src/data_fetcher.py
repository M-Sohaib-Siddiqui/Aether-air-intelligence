import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from src.config import CITIES, AIR_QUALITY_API_URL, WEATHER_API_URL, POLLUTANTS, WEATHER_VARS

HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_air_quality(city_name: str, past_days: int = 7) -> pd.DataFrame:
    """
    Fetch hourly air quality observations from Open-Meteo Air Quality API.
    Supports short windows (7 days) up to 1 full year (365 days).
    """
    if city_name not in CITIES:
        raise ValueError(f"City '{city_name}' not recognized. Choose from {list(CITIES.keys())}")

    city = CITIES[city_name]
    
    if past_days <= 92:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide", "us_aqi"],
            "past_days": past_days,
            "forecast_days": 3,
            "timezone": "UTC"
        }
    else:
        now_utc = datetime.now(timezone.utc)
        start_date = (now_utc - timedelta(days=past_days)).strftime("%Y-%m-%d")
        end_date = now_utc.strftime("%Y-%m-%d")
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide", "us_aqi"],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC"
        }

    response = requests.get(AIR_QUALITY_API_URL, params=params, timeout=25)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    if "time" not in df.columns:
        raise ValueError(f"Air quality data for {city_name} returned empty time series.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["city"] = city_name

    # Rename columns to standard names
    df = df.rename(columns={
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co",
        "us_aqi": "aqi"
    })
    return df

def fetch_weather(city_name: str, past_days: int = 7) -> pd.DataFrame:
    """
    Fetch hourly real-time weather observations from Open-Meteo Forecast/Archive API.
    Supports short windows (7 days) and long-range historical archives (365 days).
    """
    if city_name not in CITIES:
        raise ValueError(f"City '{city_name}' not recognized.")

    city = CITIES[city_name]
    
    if past_days <= 92:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "weather_code"],
            "past_days": past_days,
            "forecast_days": 3,
            "timezone": "UTC"
        }
        response = requests.get(WEATHER_API_URL, params=params, timeout=25)
        response.raise_for_status()
    else:
        now_utc = datetime.now(timezone.utc)
        start_date = (now_utc - timedelta(days=past_days)).strftime("%Y-%m-%d")
        end_date = now_utc.strftime("%Y-%m-%d")
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "weather_code"],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC"
        }
        response = requests.get(HISTORICAL_WEATHER_URL, params=params, timeout=25)
        if response.status_code != 200:
            response = requests.get(WEATHER_API_URL, params=params, timeout=25)
        response.raise_for_status()

    data = response.json()
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    if "time" not in df.columns:
        raise ValueError(f"Weather data for {city_name} returned empty time series.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["city"] = city_name

    df = df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "surface_pressure": "pressure",
        "weather_code": "weather_code"
    })
    return df

def fetch_combined_city_data(city_name: str, past_days: int = 7) -> pd.DataFrame:
    """
    Fetch and merge air quality and weather observations into a single DataFrame.
    """
    aq_df = fetch_air_quality(city_name, past_days=past_days)
    weather_df = fetch_weather(city_name, past_days=past_days)

    merged = pd.merge(aq_df, weather_df, on=["time", "city"], how="inner")
    
    # Fill any missing values using forward/backward fill
    merged = merged.ffill().bfill()
    
    # Standardize AQI calculation if missing or 0
    if "aqi" not in merged.columns or merged["aqi"].isna().all():
        merged["aqi"] = calculate_approx_aqi(merged["pm2_5"])
    else:
        merged["aqi"] = merged["aqi"].fillna(calculate_approx_aqi(merged["pm2_5"]))

    return merged

def calculate_approx_aqi(pm2_5_series: pd.Series) -> pd.Series:
    """
    Approximate US EPA AQI from PM2.5 concentration (ug/m3) fallback.
    """
    def pm2_5_to_aqi(c):
        if pd.isna(c) or c < 0:
            return 0
        if c <= 12.0:
            return (50 / 12.0) * c
        elif c <= 35.4:
            return 51 + ((100 - 51) / (35.4 - 12.1)) * (c - 12.1)
        elif c <= 55.4:
            return 101 + ((150 - 101) / (55.4 - 35.5)) * (c - 35.5)
        elif c <= 150.4:
            return 151 + ((200 - 151) / (150.4 - 55.5)) * (c - 55.5)
        elif c <= 250.4:
            return 201 + ((300 - 201) / (250.4 - 150.5)) * (c - 150.5)
        else:
            return 301 + ((500 - 301) / (500.4 - 250.5)) * (c - 250.5)

    return pm2_5_series.apply(pm2_5_to_aqi).round().astype(int)

def fetch_all_cities_data(past_days: int = 7) -> pd.DataFrame:
    """
    Fetch combined air quality & weather data across all configured target cities.
    """
    dfs = []
    for city in CITIES.keys():
        try:
            df_city = fetch_combined_city_data(city, past_days=past_days)
            dfs.append(df_city)
        except Exception as e:
            print(f"Error fetching data for {city}: {e}")

    if not dfs:
        raise RuntimeError("Failed to fetch data for any target city.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined

if __name__ == "__main__":
    print("Fetching test sample from Open-Meteo...")
    sample_df = fetch_all_cities_data(past_days=2)
    print(f"Fetched {len(sample_df)} rows across {sample_df['city'].nunique()} cities.")
    print(sample_df.head(5))
