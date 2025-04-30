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

THINGSPEAK_KEYS = {
    "board_inside": {
        "channel_id": "2312381",
        "read_key": "4LBD6CEVGGNDKQE2",
        "field_names": ["pm10"]
    },
    "board_outside": {
        "channel_id": "2934032",
        "read_key": "NW6NJ5E711EPC8EY",
        "field_names": ["pm10"]
    }
}

OW_API_KEY = os.getenv("OW_API_KEY", "83d50870977893b61ac48149455cf65a")

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
    "table_name": ["historical", "board_inside", "board_outside", "hourly_2025_inside", "hourly_2025_outside"]
}

FEATURE_COLUMNS = [
    # Core measurements
    "pm10", "temperature", "humidity", "pressure", "wind_speed", "precipitation",

    # Temporal & cyclical
    "hour_sin", "hour_cos", "week_sin", "week_cos", "month_sin", "month_cos",
    "winter_indicator", "spring_indicator", "summer_indicator", "fall_indicator",

    # Lag features
    "pm10_lag3", "pm10_lag12", "pm10_lag24", "pm10_lag168",

    # Rolling statistics
    "3h_pm10_avg",  "24h_pm10_avg",
    "3h_pm10_std",  "24h_pm10_std",
    "3h_pressure_avg",
    "3h_wind_speed_avg", "24h_wind_speed_avg",
    "rolling_max_pm10_24h", "rolling_min_pm10_24h",

    # Weather features
    "3h_wind_speed_avg", "24h_wind_speed_avg",

    # Weather and air quality interactions
    "pm10_volatility_3h", "pm10_diff_3h", "pm10_diff_12h", "pm10_diff_24h", 

    # Precipitation and dew point
    "is_precipitation", "precipitation_intensity", "dew_point",

    # Debrecen-specific
    "temp_24h_trend", "pollution_load",
    "temp_lag24", "temp_diff_24h",
    "stagnation_index", "heating_effect", "dry_spell_days"
    
]