"""
Data fetching functions for the PM10 prediction project.
"""

import pandas as pd
import pytz
import requests
import logging
from datetime import datetime, timedelta

import openmeteo_requests
import requests_cache
from retry_requests import retry
from pyspark.sql.functions import lit, col

from src.config import (
    DEBRECEN_LAT, DEBRECEN_LON, DEBRECEN_ELEVATION,
    THINGSPEAK_KEYS,
    OW_API_KEY
)

from src.db.db_operations import db_data_transaction
from src.dataframe.assemble_dataframe import assemble_dataframe
from src.dataframe.optimize_dataframe import optimize_dataframe
from src.model.data_processing import add_unified_features

logger = logging.getLogger(__name__)

# Configure Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=30)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def fetch_open_meteo_data(endpoint, params):
    """
    Generic function to fetch data from Open-Meteo APIs.

    Parameters:
        endpoint (str): Open-Meteo API endpoint URL.
        params (dict): Parameters for the API request.

    Returns:
        dict: Response data or None if failed.
    """
    try:
        response = openmeteo.weather_api(endpoint, params=params)
        return response[0]
    except Exception as e:
        logger.error(f"Error fetching data from {endpoint}: {str(e)}")
        return None

def historical_air_quality(start_date, end_date):
    aq_response = fetch_open_meteo_data(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "hourly": ["pm10"],
            "start_date": start_date,
            "end_date": end_date
        }
    )
    if not aq_response:
        return None

    try:
        aq_hourly = aq_response.Hourly()
        # Validate required methods exist
        if not (callable(aq_hourly.Time) and callable(aq_hourly.TimeEnd) and callable(aq_hourly.Interval)):
            logger.error("Invalid hourly data structure in air quality response")
            return None

        time_start = pd.to_datetime(aq_hourly.Time(), unit="s", utc=True)
        time_end = pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True)
        interval = aq_hourly.Interval()

        aq_data = {
            "datetime": pd.date_range(
                start=time_start,
                end=time_end,
                freq=pd.Timedelta(seconds=interval),
                inclusive="left"
            ).tz_localize(None),
            "pm10": aq_hourly.Variables(0).ValuesAsNumpy()
        }
        return pd.DataFrame(aq_data)
    except AttributeError as e:
        logger.error(f"Invalid air quality API response structure: {str(e)}")
        return None

def historical_weather(start_date, end_date):
    weather_response = fetch_open_meteo_data(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m",
                "wind_direction_10m", "precipitation"
            ]
            }
        )
    if not weather_response:
        return None

    weather_hourly = weather_response.Hourly()
    weather_data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=weather_hourly.Interval()),
            inclusive="left"
        ).tz_localize(None),
        "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
        "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
        "pressure": weather_hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed": weather_hourly.Variables(3).ValuesAsNumpy(),
        "wind_dir": weather_hourly.Variables(4).ValuesAsNumpy(),
        "precipitation": weather_hourly.Variables(5).ValuesAsNumpy()
    }
    weather_df = pd.DataFrame(weather_data)
    return weather_df

def future_weather(start_date, end_date):
    weather_response = fetch_open_meteo_data(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m",
                "wind_direction_10m", "precipitation"
                ],
                "start_date": start_date,
                "end_date": end_date
            }
        )
    if not weather_response:
        return None

    weather_hourly = weather_response.Hourly()
    weather_data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=weather_hourly.Interval()),
            inclusive="left"
        ).tz_localize(None),
        "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
        "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
        "pressure": weather_hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed": weather_hourly.Variables(3).ValuesAsNumpy(),
        "wind_dir": weather_hourly.Variables(4).ValuesAsNumpy(),
        "precipitation": weather_hourly.Variables(5).ValuesAsNumpy()
    }
    weather_df = pd.DataFrame(weather_data)
    return weather_df

def assemble_and_pass(spark, start_date, end_date,
                      is_future=False,
                      redownload='n',
                      db_operation='',
                      table=''):
    """
    Fetch historical or future air quality and weather data, merge with hourly continuity, enrich,
    and optionally save or return as Spark DataFrame.

    Returns Spark DataFrame with columns:
      ['datetime','pm10','temperature','humidity','pressure',
       'wind_speed','wind_dir','precipitation','elevation','is_urban','is_future']
    """
    try:
        now = datetime.now().replace(minute=0, second=0, microsecond=0)

        if not is_future:
            # Historical path
            aq_df      = historical_air_quality(start_date.isoformat(), end_date.isoformat())
            weather_df = historical_weather   (start_date.isoformat(), end_date.isoformat())
                               
            if aq_df is None or weather_df is None:
                logger.error("assemble_and_pass: missing historical dataframes")
                return None

            extra_cols = {
                "elevation": DEBRECEN_ELEVATION,
                "is_urban":  True,
                "is_future": False
            }
            merged = assemble_dataframe(
                spark,
                [aq_df, weather_df],
                join_key="datetime",
                how="inner",
                extra_columns=extra_cols
            ).dropDuplicates(["datetime"])

            final_df = optimize_dataframe(
                merged,
                partition_cols="datetime",
                partition_count=168
            )

        else:
            # Future path
            aq_end     = max(end_date, now.date())
            aq_df      = historical_air_quality(start_date.isoformat(), aq_end.isoformat())
            weather_df = future_weather(start_date.isoformat(), end_date.isoformat())
            if aq_df is None or weather_df is None:
                logger.error("assemble_and_pass: missing future dataframes")
                return None

            # full hourly index
            all_hours = pd.DataFrame({
                "datetime": pd.date_range(start=start_date, end=end_date, freq="h")
            })
            # merge weather + AQ
            df = all_hours.merge(weather_df, on="datetime", how="left") \
                          .merge(aq_df[["datetime","pm10"]], on="datetime", how="left")

            # inject constants
            df["elevation"]  = DEBRECEN_ELEVATION
            df["is_urban"]   = True
            df["is_future"] = df["datetime"] >= now

            print(f"[DEBUG] Weather range: {weather_df['datetime'].min()} → {weather_df['datetime'].max()}, rows: {len(weather_df)}")


            merged = spark.createDataFrame(df)
            final_df = optimize_dataframe(
                merged,
                partition_cols="datetime",
                partition_count=168
            )

        # final select
        db_cols = [
            "datetime","pm10","temperature","humidity","pressure",
            "wind_speed","wind_dir","precipitation",
            "elevation","is_urban","is_future"
        ]
        out_df = final_df.select(*db_cols)

        if redownload == "y":
            ok = db_data_transaction(spark, db_operation, table, data=out_df)
            if ok:
                logger.info("Data successfully saved to table.")
            return ok

        logger.info("Data successfully extracted.")
        return out_df

    except Exception as e:
        logger.error(f"Error in assemble_and_pass: {e}", exc_info=True)
        raise

def fetch_current_data(spark, field_id2=2, results=1):
    """
    Fetch and merge current sensor data from ThingSpeak and weather data from OpenWeather.

    Parameters:
        spark: SparkSession object.
        field_id2 (int): ThingSpeak field ID for pm10.
        results (int): Number of data points to retrieve.

    Returns:
        Spark DataFrame with merged data matching the historical_2024 schema.
    """
    logger.info("Fetching current sensor and weather data...")
    try:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Fetch ThingSpeak sensor data
        logger.info("Fetching ThingSpeak sensor data...")
        ts_url = f"https://api.thingspeak.com/channels/{THINGSPEAK_KEYS}/feeds.json?api_key={THINGSPEAK_KEYS}&results={results}"
        response = requests.get(ts_url)
        response.raise_for_status()
        ts_data = response.json()
        feeds = ts_data.get("feeds", [])
        records = []
        logger.debug(f"Feeds received: {feeds}")
        for entry in feeds:
            dt = pd.to_datetime(entry.get("created_at"), errors='coerce')
            if dt is None or pd.isna(dt):
                dt = datetime.now(pytz.UTC)
            else:
                dt = dt.tz_localize(None)
            if entry.get(f"field{field_id2}"):
                try:
                    pm10_val = float(entry[f"field{field_id2}"])
                except (ValueError, TypeError):
                    continue
                records.append({
                    "datetime": dt,
                    f"field{field_id2}": pm10_val
                })
        if not records:
            logger.warning("No valid ThingSpeak records found")
            return None
        ts_pd = pd.DataFrame(records)
        # Fetch OpenWeather data
        logger.info("Fetching current weather data...")
        if not OW_API_KEY:
            logger.error("OpenWeather API key missing")
            return None
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": DEBRECEN_LAT,
            "lon": DEBRECEN_LON,
            "appid": OW_API_KEY,
            "units": "metric"
        }
        weather_response = requests.get(weather_url, params=params)
        if weather_response.status_code != 200:
            logger.error(f"OpenWeather API failed: {weather_response.status_code}")
            return None
        weather_json = weather_response.json()
        weather_data = {
            "datetime": datetime.fromtimestamp(weather_json["dt"], pytz.UTC),
            "temperature": weather_json["main"]["temp"],
            "humidity": weather_json["main"]["humidity"],
            "pressure": weather_json["main"]["pressure"],
            "wind_speed": weather_json["wind"]["speed"],
            "wind_dir": weather_json["wind"]["deg"]
        }
        weather_pd = pd.DataFrame([weather_data])
        current_db_time = datetime.now(pytz.UTC)
        ts_pd["datetime"] = current_db_time
        weather_pd["datetime"] = current_db_time
        logger.info("Merging sensor and weather data...")
        merged_df = assemble_dataframe(
            spark,
            [ts_pd, weather_pd],
            join_key="datetime",
            how="outer",
            extra_columns={
                "elevation": DEBRECEN_ELEVATION,
                "is_urban": True,
                "data_timestamp": current_db_time.isoformat()
            }
        )
        merged_df = merged_df.dropDuplicates(["datetime"])
        merged_df = merged_df.withColumn("datetime", col("datetime").cast("timestamp"))
        db_columns = ["datetime", "pm10", "temperature", "humidity", 
                      "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"]
        field_map = {
            f"field{field_id2}": "pm10"
        }
        for field, col_name in field_map.items():
            if field in merged_df.columns:
                merged_df = merged_df.withColumnRenamed(field, col_name)
        for col_name in db_columns:
            if col_name not in merged_df.columns:
                merged_df = merged_df.withColumn(col_name, lit(None))
        final_df = merged_df.select(*db_columns).dropDuplicates(["datetime"])
        optimized_df = optimize_dataframe(final_df, partition_cols="datetime", partition_count=8)
        save_success = db_data_transaction(spark, "save", "current", data=optimized_df)
        if save_success:
            logger.info(f"Successfully saved {optimized_df.count()} current records")
            return optimized_df
        else:
            logger.error("Failed to save current data")
            return None
    except Exception as e:
        logger.error(f"Error fetching current data: {str(e)}")
        return None

def get_prediction_input_data(spark):
    try:
        # parameters
        today   = datetime.today().date() - timedelta(days=2)
        past_90 = today - timedelta(days=90)
        fut_start = today - timedelta(days=2)
        fut_end   = today + timedelta(days=7)

        # fetch
        hist_spark   = assemble_and_pass(spark, past_90, today,    is_future=False)
        future_spark = assemble_and_pass(spark, fut_start, fut_end, is_future=True)
        logger.info(f"Hist rows: {hist_spark.count()}, Future rows: {future_spark.count()}")

        # union, dedupe, sort
        combo = (
            hist_spark
              .unionByName(future_spark)
              .dropDuplicates(["datetime"])
              .sort("datetime")
        )
        
        print("\n=== FIRST FUTURE RECORDS ===")
        future_spark.orderBy("datetime").show(200, truncate=False)
        
        # split
        hist_final = combo.filter((col("is_future") == False) & col("pm10").isNotNull())
        fut_final  = combo.filter(col("is_future") == True) \
                          .select(
                              "datetime","temperature","humidity","pressure",
                              "wind_speed","wind_dir","precipitation",
                              "elevation","is_urban","is_future"
                          )

        return hist_final, fut_final

    except Exception as e:
        logger.error(f"Error in get_prediction_input_data: {e}", exc_info=True)
        return None
