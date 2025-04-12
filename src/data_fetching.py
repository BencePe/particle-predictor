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
        spark_df: Spark DataFrame with timestamp column
        timestamp_col (str): Name of the timestamp column
        
    Returns:
        Spark DataFrame with normalized timestamp
    """
    if spark_df is None:
        return None
        
    # Convert to string and back to timestamp to remove timezone info
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
        extra_columns (dict): A dictionary of {column_name: value} to add as extra columns 
                              to the final Spark DataFrame (default None).

    Returns:
        Spark DataFrame: The merged and enriched DataFrame.
    """
    if not df_list or len(df_list) < 1:
        logger.error("No dataframes provided for assembly.")
        return None

    # Start with the first DataFrame and merge successively with the rest
    merged_pd_df = df_list[0]
    for df in df_list[1:]:
        merged_pd_df = pd.merge(merged_pd_df, df, on=join_key, how=how)

    # Drop duplicates in Pandas before converting to Spark
    if deduplicate:
        merged_pd_df = merged_pd_df.drop_duplicates(subset=join_key)

    # Convert the merged Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(merged_pd_df)
    
    # Normalize timestamp format to ensure consistency
    spark_df = normalize_timestamp(spark_df, timestamp_col=join_key)

    # Optionally, add extra columns (e.g. location or constant metadata)
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
        
    # Start with the original DataFrame
    result_df = df
    
    # Repartition if specified
    if partition_cols and partition_count:
        result_df = result_df.repartition(partition_count, partition_cols)
    elif partition_cols:
        result_df = result_df.repartition(partition_cols)
    elif partition_count:
        result_df = result_df.repartition(partition_count)
    
    # Cache if requested
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
        spark: SparkSession object
        
    Returns:
        DataFrame: Spark DataFrame with merged data (Open-Meteo air quality and weather data).
    """
    logger.info(f"Fetching data for Debrecen from {START_DATE} to {END_DATE}")

    try:
        # --- Fetch air quality data ---
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
            ).tz_localize(None),  # Remove timezone info
            "pm10": aq_hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": aq_hourly.Variables(1).ValuesAsNumpy()
        }
        aq_df = pd.DataFrame(aq_data)

        # --- Fetch weather data ---
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
            ).tz_localize(None),  # Remove timezone info
            "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
            "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
            "pressure": weather_hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed": weather_hourly.Variables(3).ValuesAsNumpy(),
            "wind_dir": weather_hourly.Variables(4).ValuesAsNumpy()
        }
        weather_df = pd.DataFrame(weather_data)

        # --- Assemble the merged DataFrame using the helper function ---
        # Extra columns to add: elevation and is_urban.
        extra_cols = {"elevation": DEBRECEN_ELEVATION, "is_urban": True}
        merged_spark_df = assemble_dataframe(spark, [aq_df, weather_df], join_key="datetime", how="inner", extra_columns=extra_cols)

        # Ensure no duplicate timestamps
        merged_spark_df = merged_spark_df.dropDuplicates(['datetime'])

        # Optimize and save the DataFrame
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
        spark: SparkSession object
        field_id2 (int): ThingSpeak field ID for pm10 (default 2)
        field_id4 (int): ThingSpeak field ID for pm2_5 (default 4)
        results (int): Number of data points to retrieve from ThingSpeak (default 1000)
        
    Returns:
        Spark DataFrame: Merged data matching historical_2024 schema, with 'datetime'
                         in timestamptz (UTC) format.
    """
    logger.info("Fetching current sensor and weather data...")
    
    try:
        # Set legacy time parser policy to handle Spark datetime parsing as before Spark 3.0.
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        
        # 1. Fetch ThingSpeak sensor data
        logger.info("Fetching ThingSpeak sensor data...")
        ts_url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API_KEY}&results={results}"
        response = requests.get(ts_url)
        response.raise_for_status()
        ts_data = response.json()
        
        feeds = ts_data.get("feeds", [])
        records = []
        logger.debug(f"Feeds received: {feeds}")
        
        # Process each record from ThingSpeak feeds.
        # If 'created_at' is missing or unparseable, use the current time.
        for entry in feeds:
            dt = pd.to_datetime(entry.get("created_at"), errors='coerce')
            if dt is None or pd.isna(dt):
                dt = datetime.now(pytz.UTC)
            else:
                # Remove any attached timezone (if present) for consistency; we reassign UTC below.
                dt = dt.tz_localize(None)
            # Check that both fields exist and can be converted to float.
            if entry.get(f"field{field_id2}") and entry.get(f"field{field_id4}"):
                try:
                    pm10_val = float(entry[f"field{field_id2}"])
                    pm2_5_val = float(entry[f"field{field_id4}"])
                except (ValueError, TypeError):
                    continue
                records.append({
                    "datetime": dt,  # initially naive, we will assign UTC below
                    f"field{field_id2}": pm10_val,
                    f"field{field_id4}": pm2_5_val
                })
        
        if not records:
            logger.warning("No valid ThingSpeak records found")
            return None
            
        ts_pd = pd.DataFrame(records)
        
        # 2. Fetch OpenWeather data
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
        
        # 3. Use the current time (in UTC) for database consistency.
        current_db_time = datetime.now(pytz.UTC)
        ts_pd["datetime"] = current_db_time
        weather_pd["datetime"] = current_db_time
        
        # 4. Merge datasets using your helper function
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
        
        # Drop any duplicate records based on datetime.
        merged_df = merged_df.dropDuplicates(["datetime"])
        
        # Convert datetime to timestamp using legacy policy.
        # Since the underlying value is timezone aware (UTC), the PostgreSQL driver
        # will interpret it as timestamptz.
        merged_df = merged_df.withColumn("datetime", col("datetime").cast("timestamp"))
        
        # 5. Map to the table schema
        db_columns = ["datetime", "pm10", "pm2_5", "temperature", "humidity", 
                      "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"]
        field_map = {
            f"field{field_id2}": "pm10",
            f"field{field_id4}": "pm2_5"
        }
        for field, col_name in field_map.items():
            if field in merged_df.columns:
                merged_df = merged_df.withColumnRenamed(field, col_name)
        
        # Ensure all schema columns exist; add missing columns as null.
        for col_name in db_columns:
            if col_name not in merged_df.columns:
                merged_df = merged_df.withColumn(col_name, lit(None))
                
        final_df = merged_df.select(*db_columns).dropDuplicates(["datetime"])
        
        # 6. Optimize and save the DataFrame
        optimized_df = optimize_dataframe(final_df, partition_cols="datetime", partition_count=8)
        
        # Save to the "current" table in your database.
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
        spark: SparkSession object
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        logger.info("Computing hourly mean from current sensor data...")
        
        # Load data from the 'current' table
        current_data = db_data_transaction(spark, "load", "current")
        
        if current_data is None or current_data.count() == 0:
            logger.warning("No current data found to compute hourly mean.")
            return False
        
        # Truncate datetime to hours for grouping
        current_data = current_data.withColumn("hour", date_trunc("hour", col("datetime")))
        
        # Group by hour and compute averages
        hourly_means = current_data.groupBy("hour") \
            .agg(
                avg("pm10").alias("pm10"),
                avg("pm2_5").alias("pm2_5"),
                avg("temperature").alias("temperature"),
                avg("humidity").alias("humidity"),
                avg("pressure").alias("pressure"),
                avg("wind_speed").alias("wind_speed"),
                avg("wind_dir").alias("wind_dir"),
                # Keep the first value for non-numeric columns
                lit(DEBRECEN_ELEVATION).alias("elevation"),
                lit(True).alias("is_urban")
            ) \
            .withColumnRenamed("hour", "datetime")
        
        # Normalize timestamp format to ensure consistency
        hourly_means = normalize_timestamp(hourly_means)
        
        # Ensure we have all the required columns with the right names
        hourly_means = hourly_means.select(
            "datetime", "pm10", "pm2_5", "temperature", "humidity", 
            "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"
        )
        
        # Save hourly means to air_quality_2025 table
        success = db_data_transaction(spark, "save", "air_quality_2025", data=hourly_means)
        
        if not success:
            logger.error("Failed to save hourly means to air_quality_2025.")
            return False
            
        # Clear the current table by executing a delete query
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