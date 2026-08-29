import os
from pathlib import Path
from dotenv import load_dotenv

# Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

# Target Cities Configuration
CITIES = {
    "Karachi": {
        "lat": 24.8607,
        "lon": 67.0011,
        "timezone": "Asia/Karachi",
        "country": "Pakistan"
    },
    "Chicago": {
        "lat": 41.8781,
        "lon": -87.6298,
        "timezone": "America/Chicago",
        "country": "United States"
    },
    "Sydney": {
        "lat": -33.8688,
        "lon": 151.2093,
        "timezone": "Australia/Sydney",
        "country": "Australia"
    },
    "Austria": {
        "lat": 48.2082,
        "lon": 16.3738,
        "timezone": "Europe/Vienna",
        "country": "Austria (Vienna)"
    }
}

# Open-Meteo Endpoints (100% Free & Open Access)
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_QUALITY_URL = AIR_QUALITY_API_URL
OPEN_METEO_WEATHER_URL = WEATHER_API_URL

POLLUTANTS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"]
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]

# Hopsworks Feature Store & Model Registry Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "Aether")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

# Voice Agent & 3D Globe Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_425b65f04d7fbcf101fb54ee11fbfa1fbc8ebccf2b1d2afa")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "P8NfsqD6Mj2lTFzuAccu")
CESIUM_ION_ACCESS_TOKEN = os.getenv("CESIUM_ION_ACCESS_TOKEN", os.getenv("CESIUM_ION_TOKEN", ""))

# Local Storage Directory Fallback
LOCAL_DATA_DIR = ROOT_DIR / "data"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Air Quality Categories (US EPA AQI Standard)
AQI_CATEGORIES = [
    (0, 50, "Good", "#10B981"),
    (51, 100, "Moderate", "#F59E0B"),
    (101, 150, "Unhealthy for Sensitive Groups", "#F97316"),
    (151, 200, "Unhealthy", "#EF4444"),
    (201, 300, "Very Unhealthy", "#8B5CF6"),
    (301, 500, "Hazardous", "#6B21A8")
]
