"""
Central configuration settings for the PM10 prediction project.
"""

import os
from datetime import datetime

# === Project paths ===
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_PATH)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# === Timestamp ===
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Logging ===
LOG_FILE = os.path.join(LOG_DIR, f"pm10_prediction_{RUN_TIMESTAMP}.log")

# === Date settings ===
START_DATE = datetime(2020, 1, 1).date()
END_DATE = datetime(2025, 1, 1).date()

# === Location settings ===
DEBRECEN_LAT = 47.5317
DEBRECEN_LON = 21.6244
DEBRECEN_ELEVATION = 122

# === API keys ===
THINGSPEAK_CHANNEL_ID = "2312381"
THINGSPEAK_READ_API_KEY = "4LBD6CEVGGNDKQE2"
OW_API_KEY = os.getenv("OW_API_KEY", "83d50870977893b61ac48149455cf65a")
TT_API_KEY = os.getenv("TT_API_KEY", "")

# === API request limit ===
API_REQUEST_LIMIT = 3000

# === Spark configuration ===
SPARK_CONFIG = {
    "app_name": "Debrecen_PM10_Prediction",
    "shuffle_partitions": "48",
    "arrow_enabled": "true",
    "driver_java_options": "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "spark.sql.warehouse.dir": "file:///C:/temp/spark-warehouse",
    "spark.driver.extraLibraryPath": "C:\\hadoop\\bin"
}

# === Model parameters ===
MODEL_PARAMS = {
    "max_depth_rf": 3,
    "num_trees": 50,
    "subsampling_rate": 0.8,
    "seed": 42,
    "minInstancesPerNode": 5,
    "featureSubsetStrategy": "sqrt"
}

# === Database configuration ===
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "172.25.221.213"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "sensors"),
    "user": os.getenv("DB_USER", "username"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "table_name": "historical_2024"
}

# === Feature columns ===
FEATURE_COLUMNS = [
    # Core measurements
    "pm10", "pm2_5", "temperature", "humidity", "pressure",
    "wind_speed", "wind_dir", "elevation", "is_urban",

    # Temporal flags & cyclical encoding
    "month", "hour", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos", "hour_weekend",

    # Multi‑horizon lags
    "pm10_lag12", "pm10_lag24", "pm10_lag48",
    "pm10_lag72", "pm10_lag168",

    # Rolling averages & volatilities
    "12h_pm10_avg", "24h_pm10_avg", "48h_pm10_avg",
    "72h_pm10_avg", "weekly_pm10_avg",
    "12h_pm10_std", "24h_pm10_std", "weekly_pm10_std",
    "pm10_volatility_24h",

    # Differences & acceleration
    "pm10_diff_12h", "pm10_diff_24h", "pm10_diff_48h",
    "pm10_acceleration",

    # Weather trends & interactions
    "pressure_12h_trend", "pressure_24h_trend",
    "temp_12h_trend", "temp_24h_trend",
    "humidity_temp_index", "temp_wind_index",
    "pressure_change_velocity",

    # Wind stability
    "wind_dir_stability",

    # Pollution ratios/load
    "pm_ratio", "pollution_load",
    "pm10_12h_avg_sq", "avg12h_times_diff12h",

    # Future‐flag
    "is_future"
]

