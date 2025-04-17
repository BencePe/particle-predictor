"""
Data fetching functions for the PM10 prediction project.
"""

import pandas as pd
import pytz
import requests
import time
import json
import logging
from datetime import datetime, timedelta

import openmeteo_requests
import requests_cache
from retry_requests import retry
from pyspark.sql.functions import lit, col, date_trunc, avg
from pyspark.sql.types import TimestampType

from config import (
    DEBRECEN_LAT, DEBRECEN_LON, DEBRECEN_ELEVATION, 
    START_DATE, END_DATE,
    THINGSPEAK_CHANNEL_ID, THINGSPEAK_READ_API_KEY,
    OW_API_KEY
)
from db.db_operations import db_data_transaction
from dataframe.assemble_dataframe import assemble_dataframe
from dataframe.normalize_timestamp import normalize_timestamp
from dataframe.optimize_dataframe import optimize_dataframe

logger = logging.getLogger(__name__)

# Configure Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
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
        "hourly": ["pm10", "pm2_5"],
        "start_date": start_date,
        "end_date": end_date
        }
    )
    if not aq_response:
        return None

    aq_hourly = aq_response.Hourly()
    aq_data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=aq_hourly.Interval()),
            inclusive="left"
        ).tz_localize(None),
        "pm10": aq_hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": aq_hourly.Variables(1).ValuesAsNumpy()
    }
    aq_df = pd.DataFrame(aq_data)
    return aq_df

def historical_weather(start_date, end_date):
    weather_response = fetch_open_meteo_data(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m",
                "wind_direction_10m"
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
        "wind_dir": weather_hourly.Variables(4).ValuesAsNumpy()
    }
    weather_df = pd.DataFrame(weather_data)
    return weather_df

def future_weather():
    weather_response = fetch_open_meteo_data(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m",
                "wind_direction_10m"
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
        "wind_dir": weather_hourly.Variables(4).ValuesAsNumpy()
    }
    weather_df = pd.DataFrame(weather_data)
    return weather_df

def assemble_and_pass(spark, redownload, db_operation, table):
    """
    Fetch historical air quality and weather data for model building or prediction (default is model building).

    Parameters:
        spark: SparkSession object.

    Returns:
        Spark DataFrame with merged data.
    """
    try:
        aq_df = historical_air_quality(START_DATE.isoformat(),END_DATE.isoformat())
        weather_df = historical_weather(START_DATE.isoformat(),END_DATE.isoformat())

        # Merge datasets
        extra_cols = {"elevation": DEBRECEN_ELEVATION, "is_urban": True}
        merged_spark_df = assemble_dataframe(spark, [aq_df, weather_df], join_key="datetime", how="inner", extra_columns=extra_cols)
        merged_spark_df = merged_spark_df.dropDuplicates(['datetime'])
        final_df = optimize_dataframe(merged_spark_df, partition_cols="datetime", partition_count=48)
        db_columns = ["datetime", "pm10", "pm2_5", "temperature", "humidity", 
                      "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"]
        db_df = final_df.select(*db_columns)
        

        if redownload == 'n':
            logger.info("Data successfully extracted.")
            return db_df
        else:
            success = db_data_transaction(spark, db_operation, table, data=db_df)
            if success:
                logger.info("Data successfully saved to table.")
            return success
            
    except Exception as e:
        logger.error(f"Error fetching Debrecen data: {str(e)}")
        raise

def fetch_current_data(spark, field_id2=2, field_id4=4, results=1):
    """
    Fetch and merge current sensor data from ThingSpeak and weather data from OpenWeather.

    Parameters:
        spark: SparkSession object.
        field_id2 (int): ThingSpeak field ID for pm10.
        field_id4 (int): ThingSpeak field ID for pm2_5.
        results (int): Number of data points to retrieve.

    Returns:
        Spark DataFrame with merged data matching the historical_2024 schema.
    """
    logger.info("Fetching current sensor and weather data...")
    try:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Fetch ThingSpeak sensor data
        logger.info("Fetching ThingSpeak sensor data...")
        ts_url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API_KEY}&results={results}"
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
            if entry.get(f"field{field_id2}") and entry.get(f"field{field_id4}"):
                try:
                    pm10_val = float(entry[f"field{field_id2}"])
                    pm2_5_val = float(entry[f"field{field_id4}"])
                except (ValueError, TypeError):
                    continue
                records.append({
                    "datetime": dt,
                    f"field{field_id2}": pm10_val,
                    f"field{field_id4}": pm2_5_val
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
        db_columns = ["datetime", "pm10", "pm2_5", "temperature", "humidity", 
                      "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"]
        field_map = {
            f"field{field_id2}": "pm10",
            f"field{field_id4}": "pm2_5"
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
    """
    Fetches historical data and prepares future data for prediction.
    """
    try:
        # Define date ranges
        today = datetime.today()
        past_date = today - timedelta(days=31)
        future_date = today + timedelta(days=7)
        
        # Format as strings for API calls
        today_str = today.strftime('%Y-%m-%d')
        past_str = past_date.strftime('%Y-%m-%d')
        future_str = future_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching historical data from {past_str} to {today_str}")
        logger.info(f"Preparing future data from {today_str} to {future_str}")
        
        # Fetch historical weather and air quality data
        hist_weather_df = historical_weather(past_str, today_str)
        hist_aq_df = historical_air_quality(past_str, today_str)
        
        # Fetch future weather data (next 7 days)
        future_weather_df = future_weather()
        
        # Create merged historical dataframe
        if not hist_weather_df.empty and not hist_aq_df.empty:
            historical_df = pd.merge(hist_weather_df, hist_aq_df, on='datetime', how='inner')
            historical_df['is_future'] = False
            historical_df['is_urban'] = True
            historical_df['elevation'] = DEBRECEN_ELEVATION
        else:
            logger.error("Failed to fetch historical data")
            return None
            
        # Prepare future dataframe with weather data only
        if not future_weather_df.empty:
            future_df = future_weather_df.copy()
            
            # Add placeholder PM columns
            pm_columns = [col for col in hist_aq_df.columns if col != 'datetime']
            for col in pm_columns:
                future_df[col] = None    
            
            future_df['elevation'] = DEBRECEN_ELEVATION
            future_df['is_urban'] = True
            future_df['is_future'] = True

        else:
            logger.error("Failed to fetch future weather data")
            return None
            
        # Combine datasets
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        
        # Convert to Spark DataFrame
        input_df = spark.createDataFrame(combined_df)
        
        return input_df
        
    except Exception as e:
        logger.error(f"Error fetching prediction input data: {str(e)}")
        return None