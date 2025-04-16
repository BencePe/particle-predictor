"""
Data fetching functions for the PM10 prediction project.
"""

import os
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
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, date_trunc, avg, to_timestamp
from pyspark.sql.types import TimestampType
from pyspark import StorageLevel
from pyspark.ml import PipelineModel

from config import (
    DEBRECEN_LAT, DEBRECEN_LON, DEBRECEN_ELEVATION, 
    START_DATE, END_DATE, 
    THINGSPEAK_CHANNEL_ID, THINGSPEAK_READ_API_KEY,
    API_REQUEST_LIMIT,
    OW_API_KEY,
    DATA_DIR
)
from db_operations import db_data_transaction

logger = logging.getLogger(__name__)

# Configure Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def normalize_timestamp(spark_df, timestamp_col="datetime"):
    """
    Normalize timestamps to ensure consistent format without timezone info
    for TimescaleDB compatibility.

    Parameters:
        spark_df: Spark DataFrame with timestamp column.
        timestamp_col (str): Name of the timestamp column.

    Returns:
        Spark DataFrame with normalized timestamp.
    """
    if spark_df is None:
        return None
    normalized_df = spark_df.withColumn(
        timestamp_col,
        to_timestamp(col(timestamp_col).cast("string"), "yyyy-MM-dd HH:mm:ss")
    )
    return normalized_df


def assemble_dataframe(spark, df_list, join_key="datetime", how="inner", extra_columns=None, deduplicate=True):
    """
    Assemble a merged Spark DataFrame from a list of Pandas DataFrames.

    Parameters:
        spark: SparkSession object.
        df_list (list): List of Pandas DataFrames to merge.
        join_key (str): Column name to join on (default "datetime").
        how (str): Merge type, e.g., "inner", "left" (default "inner").
        extra_columns (dict): A dictionary of {column_name: value} to add as extra columns.
        deduplicate (bool): If True, drop duplicates on the join key.

    Returns:
        Spark DataFrame: The merged and enriched DataFrame.
    """
    if not df_list or len(df_list) < 1:
        logger.error("No dataframes provided for assembly.")
        return None

    merged_pd_df = df_list[0]
    for df in df_list[1:]:
        merged_pd_df = pd.merge(merged_pd_df, df, on=join_key, how=how)

    if deduplicate:
        merged_pd_df = merged_pd_df.drop_duplicates(subset=join_key)

    spark_df = spark.createDataFrame(merged_pd_df)
    spark_df = normalize_timestamp(spark_df, timestamp_col=join_key)

    if extra_columns is not None:
        for col_name, col_value in extra_columns.items():
            spark_df = spark_df.withColumn(col_name, lit(col_value))
    return spark_df


def optimize_dataframe(df, partition_cols=None, partition_count=None, cache=True, storage_level=StorageLevel.MEMORY_AND_DISK):
    """
    Optimize a Spark DataFrame for performance.

    Parameters:
        df: Spark DataFrame to optimize.
        partition_cols (list or str): Column(s) to partition by.
        partition_count (int): Number of partitions.
        cache (bool): Whether to cache the DataFrame.
        storage_level: PySpark StorageLevel to use if caching.

    Returns:
        Spark DataFrame: Optimized DataFrame.
    """
    if df is None:
        return None

    result_df = df
    if partition_cols and partition_count:
        result_df = result_df.repartition(partition_count, partition_cols)
    elif partition_cols:
        result_df = result_df.repartition(partition_cols)
    elif partition_count:
        result_df = result_df.repartition(partition_count)
    if cache:
        result_df = result_df.persist(storage_level)
    return result_df.dropna()


def save_dataframe(df, name_prefix, mode="overwrite"):
    """
    Save a DataFrame as a Parquet file with timestamp.

    Parameters:
        df: Spark DataFrame to save.
        name_prefix (str): Prefix for the filename.
        mode (str): Write mode ("overwrite", "append", etc.)

    Returns:
        str: Path where the data was saved.
    """
    if df is None:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(DATA_DIR, f"{name_prefix}_{timestamp}.parquet")
    df.write.parquet(save_path, mode=mode)
    logger.info(f"Data saved to {save_path}")
    return save_path


def rate_limit_wait(start_time, request_count):
    """
    Simple rate limiter to respect API rate limits.

    Parameters:
        start_time (float): Time when requests started.
        request_count (int): Number of requests made.
    """
    elapsed_time = time.time() - start_time
    requests_per_second = API_REQUEST_LIMIT / 5760  # Convert per hour to per second
    if request_count / elapsed_time > requests_per_second:
        wait_time = request_count / requests_per_second - elapsed_time
        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)


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


def fetch_debrecen_data(spark):
    """
    Fetch historical air quality and weather data for Debrecen.

    Parameters:
        spark: SparkSession object.

    Returns:
        Spark DataFrame with merged data.
    """
    logger.info(f"Fetching data for Debrecen from {START_DATE} to {END_DATE}")
    try:
        # Fetch air quality data
        aq_response = fetch_open_meteo_data(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": DEBRECEN_LAT,
                "longitude": DEBRECEN_LON,
                "hourly": ["pm10", "pm2_5"],
                "start_date": START_DATE.isoformat(),
                "end_date": END_DATE.isoformat()
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

        # Fetch weather data
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
                "start_date": START_DATE.isoformat(),
                "end_date": END_DATE.isoformat()
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

        # Merge datasets
        extra_cols = {"elevation": DEBRECEN_ELEVATION, "is_urban": True}
        merged_spark_df = assemble_dataframe(spark, [aq_df, weather_df], join_key="datetime", how="inner", extra_columns=extra_cols)
        merged_spark_df = merged_spark_df.dropDuplicates(['datetime'])
        final_df = optimize_dataframe(merged_spark_df, partition_cols="datetime", partition_count=48)
        db_columns = ["datetime", "pm10", "pm2_5", "temperature", "humidity", 
                      "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"]
        db_df = final_df.select(*db_columns)
        
        # Save to database
        success = db_data_transaction(spark, "save", "historical_2024", data=db_df)
        if success:
            logger.info("Data successfully saved to historical_2024 table")
        return final_df
        
    except Exception as e:
        logger.error(f"Error fetching Debrecen data: {str(e)}")
        raise


def fetch_current_data(spark, field_id2=2, field_id4=4, results=1000):
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


def compute_and_store_hourly_mean(spark):
    """
    Compute the hourly mean from data in the 'current' table, store it in 'air_quality_2025',
    and wipe the 'current' table clean in db.

    Parameters:
        spark: SparkSession object.

    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info("Computing hourly mean from current sensor data...")
        current_data = db_data_transaction(spark, "load", "current")
        if current_data is None or current_data.count() == 0:
            logger.warning("No current data found to compute hourly mean.")
            return False
        current_data = current_data.withColumn("hour", date_trunc("hour", col("datetime")))
        hourly_means = current_data.groupBy("hour") \
            .agg(
                avg("pm10").alias("pm10"),
                avg("pm2_5").alias("pm2_5"),
                avg("temperature").alias("temperature"),
                avg("humidity").alias("humidity"),
                avg("pressure").alias("pressure"),
                avg("wind_speed").alias("wind_speed"),
                avg("wind_dir").alias("wind_dir"),
                lit(DEBRECEN_ELEVATION).alias("elevation"),
                lit(True).alias("is_urban")
            ) \
            .withColumnRenamed("hour", "datetime")
        hourly_means = normalize_timestamp(hourly_means)
        hourly_means = hourly_means.select(
            "datetime", "pm10", "pm2_5", "temperature", "humidity", 
            "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"
        )
        success = db_data_transaction(spark, "save", "air_quality_2025", data=hourly_means)
        if not success:
            logger.error("Failed to save hourly means to air_quality_2025.")
            return False
        connection_properties = {
            "url": f"jdbc:postgresql://{spark.conf.get('spark.jdbc.db.host')}:{spark.conf.get('spark.jdbc.db.port')}/{spark.conf.get('spark.jdbc.db.name')}",
            "user": spark.conf.get("spark.jdbc.db.user"),
            "password": spark.conf.get("spark.jdbc.db.password"),
            "driver": "org.postgresql.Driver"
        }
        spark.read.format("jdbc") \
            .options(**connection_properties) \
            .option("dbtable", "(DELETE FROM current) AS deleted_rows") \
            .load()
        logger.info("Successfully computed hourly means and saved to air_quality_2025.")
        logger.info("Cleared current sensor data table.")
        return True
    except Exception as e:
        logger.error(f"Error computing hourly mean: {str(e)}")
        return False


# --- New Functions for Prediction Input Data and Forecast Prediction ---

def get_prediction_input_data(spark):
    """
    Build the input dataset for prediction by retrieving:
      - Historical data for the past 31 days (with both weather and measured air quality).
      - Forecast data for the next 7 days (with only weather predictors).
      
    A new column 'is_future' is added to indicate forecast rows (True) and historical rows (False).
    
    Returns:
        Spark DataFrame: Combined dataset for prediction.
    """
    try:
        # Historical data from past 31 days using the past_days parameter
        hist_params = {
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "past_days": 31,
            "hourly": [
                "pm10", "pm2_5",
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m", "wind_direction_10m"
            ]
        }
        hist_endpoint = "https://air-quality-api.open-meteo.com/v1/air-quality"
        hist_response = fetch_open_meteo_data(hist_endpoint, params=hist_params)
        if not hist_response:
            logger.error("Failed to fetch historical (past 31 days) data.")
            return None
        hist_hourly = hist_response.Hourly()
        hist_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hist_hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hist_hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hist_hourly.Interval()),
                inclusive="left"
            ).tz_localize(None),
            "pm10": hist_hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hist_hourly.Variables(1).ValuesAsNumpy(),
            "temperature": hist_hourly.Variables(2).ValuesAsNumpy(),
            "humidity": hist_hourly.Variables(3).ValuesAsNumpy(),
            "pressure": hist_hourly.Variables(4).ValuesAsNumpy(),
            "wind_speed": hist_hourly.Variables(5).ValuesAsNumpy(),
            "wind_dir": hist_hourly.Variables(6).ValuesAsNumpy()
        }
        
        hist_df = pd.DataFrame(hist_data)
        hist_df["is_future"] = False  # historical data flag

        # Forecast data for next 7 days (weather predictors only)
        forecast_params = {
            "latitude": DEBRECEN_LAT,
            "longitude": DEBRECEN_LON,
            "forecast_days": 7,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m", "wind_direction_10m"
            ]
        }
        forecast_endpoint = "https://api.open-meteo.com/v1/forecast"
        forecast_response = fetch_open_meteo_data(forecast_endpoint, params=forecast_params)
        if not forecast_response:
            logger.error("Failed to fetch forecast data for the next 7 days.")
            return None
        forecast_hourly = forecast_response.Hourly()
        forecast_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(forecast_hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(forecast_hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=forecast_hourly.Interval()),
                inclusive="left"
            ).tz_localize(None),
            "temperature": forecast_hourly.Variables(0).ValuesAsNumpy(),
            "humidity": forecast_hourly.Variables(1).ValuesAsNumpy(),
            "pressure": forecast_hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed": forecast_hourly.Variables(3).ValuesAsNumpy(),
            "wind_dir": forecast_hourly.Variables(4).ValuesAsNumpy()
        }
        forecast_df = pd.DataFrame(forecast_data)
        # Since API doesn't forecast PM values, set them to None.
        forecast_df["pm10"] = None
        forecast_df["pm2_5"] = None
        forecast_df["is_future"] = True  # forecast data flag

        # Combine historical and forecast data
        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
        extra_cols = {"elevation": DEBRECEN_ELEVATION, "is_urban": True}  # extra metadata
        combined_spark_df = assemble_dataframe(spark, [combined_df], join_key="datetime", how="inner", extra_columns=extra_cols)
        combined_spark_df = combined_spark_df.dropDuplicates(["datetime"])
        return combined_spark_df

    except Exception as e:
        logger.error(f"Error fetching prediction input data: {str(e)}")
        return None

def predict_future_air_quality(spark: SparkSession, model: PipelineModel):
    """
    Use the provided Spark ML model to forecast PM10 and PM2.5 levels.
    The model is applied to a combined dataset of historical air quality data and
    forecasted weather data for the next 7 days.
    
    Only rows where 'is_future' == True are returned as future predictions.

    Parameters:
        spark (SparkSession): Active Spark session.
        model (PipelineModel): Trained Spark ML model.

    Returns:
        DataFrame: Predictions for future timestamps.
    """
    try:
        # Fetch and prepare the input data
        input_df = get_prediction_input_data(spark)
        if input_df is None or input_df.rdd.isEmpty():
            logger.error("No input data available for prediction.")
            return None

        # Generate predictions
        full_predictions = model.transform(input_df)
        logger.info("All predictions generated successfully.")
        full_predictions.show(10, truncate=False)

        # Filter for future predictions only
        future_predictions = full_predictions.filter(col("is_future") == True)
        logger.info("Future air quality predictions extracted successfully.")
        future_predictions.show(10, truncate=False)

        return future_predictions

    except Exception as e:
        logger.error(f"Error predicting future air quality: {str(e)}")
        return None
