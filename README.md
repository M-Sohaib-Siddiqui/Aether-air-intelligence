# AETHER -- Atmospheric Intelligence and 72-Hour ML AQI Forecast Platform

Developed by Nuralis Labs

AETHER is an end-to-end, production-ready MLOps platform for real-time air quality telemetry monitoring and 72-hour multi-horizon AQI forecasting. It combines live Open-Meteo observational streams, an automated Hopsworks Cloud Feature Store, 1-year historical machine learning models, an interactive WebGL 3D CesiumJS satellite Earth, and a natural voice intelligence assistant.

---

## Key Features

1. WebGL 3D Photorealistic Globe:
   - Built with CesiumJS WebGL satellite Earth.
   - High-resolution ESRI satellite imagery with country borders, country names, and major city labels.
   - Dynamic camera flight transitions between globally tracked cities (Karachi, Chicago, Sydney, Vienna).
   - Centered camera flight target positioning.

2. 1-Year Multi-Horizon ML Forecasting:
   - Trained on 35,136 hourly observations (365 Days / 1 Year) across 4 global climate zones.
   - Multi-output Random Forest ensemble predicting t+24h, t+48h, and t+72h horizons.
   - Model metrics: R^2 = 0.85, RMSE = 10.2, MAE = 7.1.
   - Real-time SHAP feature importance ranking (PM10, AQI 1h/24h/7d lags, 7-day rolling statistics, seasonal cyclical sine/cosine factors).

3. High-Frequency Real-Time Telemetry:
   - Queries Open-Meteo high-frequency current observation streams directly.
   - Displays real-time AQI, EPA category badge, temperature, feels-like, humidity, wind speed, and pollutant breakdowns (PM2.5, PM10, O3, NO2, SO2, CO).

4. Natural Voice Intelligence Assistant:
   - Speaks out detailed environmental briefings: local time, live AQI, weather conditions, PM2.5/PM10 concentrations, 72-hour forecast, and health advice.
   - Integrates ElevenLabs neural voice synthesis with browser SpeechSynthesis fallback.

5. Production MLOps Integration:
   - Cloud Feature Store sync with Hopsworks Cloud (aqi_weather_fg v1).
   - Automated 365-day backfill pipeline (src/backfill.py).

---

## Repository Structure

`
Aether/
|-- server.py                   # Main Flask API and Web Application Server
|-- .env                        # API Keys and Cloud Credentials
|-- requirements.txt            # Python Dependencies
|-- README.md                   # Platform Documentation
|-- Aether-Report.pdf           # Technical Achievement Report (PDF)
|-- Aether-Report.docx          # Technical Achievement Report (Word)
|-- data/
|   -- features.parquet        # 1-Year Local Cached Feature Dataset (35,136 rows)
|-- models/
|   |-- best_model.joblib       # Serialized 1-Year Trained Random Forest Regressor
|   -- metrics.json            # Model Evaluation Metrics and SHAP Feature Names
|-- src/
|   |-- __init__.py
|   |-- backfill.py             # 365-Day Historical Open-Meteo Backfill Pipeline
|   |-- config.py               # City Coordinates, Thresholds and API Configurations
|   |-- data_fetcher.py         # Live and Historical Open-Meteo Atmospheric Ingestion
|   |-- explainability.py       # SHAP Tree Explainer Feature Importance Engine
|   |-- feature_pipeline.py     # Seasonal, Autoregressive and Rolling Feature Engineering
|   -- train.py                # Candidate Model Benchmark and Training Engine
-- static/
    |-- index.html              # Unified Single-Page Aether Web Application UI
    |-- aether_logo_transparent.png # Platform Logo and Favicon
    -- aether_satellite_bg.jpg # Background Satellite Texture
`

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Install Dependencies
`ash
pip install -r requirements.txt
`

### 3. Run the Server
`ash
python server.py
`

### 4. Access the Application
Open your browser and navigate to:
- http://localhost:8000

---

Developed by Nuralis Labs
Aether -- Atmospheric Intelligence and AQI Forecast Platform.
