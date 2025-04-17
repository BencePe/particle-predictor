"""Data fetching functions for the PM10 prediction project."""

import logging
import json
import typing
from datetime import datetime, timedelta

import openmeteo_requests
import pandas as pd
import pytz
import requests
import requests_cache
from requests.exceptions import RequestException
from pyspark.sql.functions import col, lit

from src.config.config_manager import get_config
from src.dataframe.assemble_dataframe import assemble_dataframe
from src.dataframe.optimize_dataframe import optimize_dataframe
from src.db.db_operations import db_data_transaction

# logger = logging.getLogger(__name__)
# Configure Open-Meteo API client
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
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
    except (RequestException, ValueError, TypeError) as e:
        logger.error(f"Error fetching data from {endpoint} with params {params}: {str(e)}")
        return None


def _fetch_data_from_open_meteo(endpoint, params, hourly_vars, logger):
    """
    Internal function to fetch and format data from Open-Meteo APIs. If the response
    is None or the data is empty, returns None.
    """
    response = fetch_open_meteo_data(endpoint, params)
    if not response:
        return None
    hourly = response.Hourly()
    data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_localize(None)
    }
    for i, var in enumerate(hourly_vars):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    result = pd.DataFrame(data)
    if result.empty:
        logger.error(f"No data fetched from {endpoint} with params {params}")
        return None
    return result


def historical_air_quality(start_date, end_date, logger=logging.getLogger(__name__)) -> typing.Optional[pd.DataFrame]:
    config = get_config()
    return _fetch_data_from_open_meteo("https://air-quality-api.open-meteo.com/v1/air-quality", {"latitude": config.DEBRECEN_LAT, "longitude": config.DEBRECEN_LON, "hourly": ["pm10", "pm2_5"], "start_date": start_date, "end_date": end_date}, ["pm10", "pm2_5"], logger)

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


def historical_weather(start_date, end_date, logger=logging.getLogger(__name__)) -> typing.Optional[pd.DataFrame]:
    config = get_config()
    return _fetch_data_from_open_meteo("https://archive-api.open-meteo.com/v1/archive", {"latitude": config.DEBRECEN_LAT, "longitude": config.DEBRECEN_LON, "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"], "start_date": start_date, "end_date": end_date}, ["temperature", "humidity", "pressure", "wind_speed", "wind_dir"], logger)

        "datetime": pd.date_range(
            start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=weather_hourly.Interval()),
            inclusive="left"
        ).tz_localize(None),
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


def future_weather(logger=logging.getLogger(__name__)) -> typing.Optional[pd.DataFrame]:
    config = get_config()
    return _fetch_data_from_open_meteo("https://api.open-meteo.com/v1/forecast", {"latitude": config.DEBRECEN_LAT, "longitude": config.DEBRECEN_LON, "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m"]}, ["temperature", "humidity", "pressure", "wind_speed", "wind_dir"], logger)


        
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


def assemble_and_pass(spark, redownload, db_operation, table, join_key="datetime", how="inner", extra_columns=None, deduplicate=True, partition_cols="datetime", partition_count=48, db_columns=["datetime", "pm10", "pm2_5", "temperature", "humidity", "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"], logger=logger):
    """
    Fetch historical air quality and weather data for model building or prediction (default is model building).

    Parameters:
        spark: SparkSession object.
        redownload (str): Whether or not to redownload the data.
        db_operation (str): Database operation (save or load).
        table (str): Name of the table.
        join_key (str): Key to join the dataframes.
        how (str): Type of join.
        extra_columns (dict): Extra columns.
        deduplicate (bool): Whether or not to deduplicate.
        partition_cols (str): Columns to partition.
        partition_count (int): Number of partitions.
        db_columns (list): Columns to save to the database.
        logger (Logger): Logger instance.

    Returns:
        Spark DataFrame with merged data.
    """
    try:
        config = get_config()
        aq_df = historical_air_quality(config.START_DATE.isoformat(), config.END_DATE.isoformat(), logger)
        if aq_df is None:
            return None

        weather_df = historical_weather(config.START_DATE.isoformat(), config.END_DATE.isoformat(), logger)
        if weather_df is None:
            return None

        merged_spark_df = assemble_dataframe(spark, [aq_df, weather_df], join_key=join_key, how=how, extra_columns=extra_columns, deduplicate=deduplicate)
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
    except (ValueError, TypeError) as e:
        logger.error(f"Data processing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in assemble_and_pass: {str(e)}")
        raise

def fetch_current_data(spark, field_id2=2, field_id4=4, results=1, join_key="datetime", how="outer", extra_columns=None, partition_cols="datetime", partition_count=8, db_columns=["datetime", "pm10", "pm2_5", "temperature", "humidity", "pressure", "wind_speed", "wind_dir", "elevation", "is_urban"], logger=None):
    """
    Fetch and merge current sensor data from ThingSpeak and weather data from OpenWeather.

    Parameters:
        spark: SparkSession object.
        field_id2 (int): ThingSpeak field ID for pm10.
        field_id4 (int): ThingSpeak field ID for pm2_5.
        join_key (str): Key to join the dataframes.
        how (str): Type of join.
        extra_columns (dict): Extra columns.
        deduplicate (bool): Whether or not to deduplicate.
        partition_cols (str): Columns to partition.
        partition_count (int): Number of partitions.
        db_columns (list): Columns to save to the database.
        logger (Logger): Logger instance.
        results (int): Number of data points to retrieve.


    Returns:
        Spark DataFrame with merged data matching the historical_2024 schema.
    """
    logger.info("Fetching current sensor and weather data...")
    try:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Fetch ThingSpeak sensor data
        try:
            config = get_config()
            logger.info("Fetching ThingSpeak sensor data...")

            ts_url = f"https://api.thingspeak.com/channels/{config.THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={config.THINGSPEAK_READ_API_KEY}&results={results}"

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
        except (RequestException, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching ThingSpeak data: {str(e)}")
            return None

        # Fetch OpenWeather data
        try:

            logger.info("Fetching current weather data...")
            if not config.OW_API_KEY:
                logger.error("OpenWeather API key missing")
                return None
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": config.DEBRECEN_LAT,
                "lon": config.DEBRECEN_LON,
                "appid": config.OW_API_KEY,
                "units": "metric"
            }
            weather_response = requests.get(weather_url, params=params)
            if weather_response.status_code != 200:
                logger.error(f"OpenWeather API failed: {weather_response.status_code}")
                return None
            weather_json = weather_response.json()
            weather_data = {
                "datetime": datetime.fromtimestamp(weather_json["dt"], pytz.UTC),
                "temperature": weather_json["main"][ "temp"],
                "humidity": weather_json["main"]["humidity"],
                "pressure": weather_json["main"]["pressure"],
                "wind_speed": weather_json["wind"]["speed"],
                "wind_dir": weather_json["wind"]["deg"]
            }
            weather_pd = pd.DataFrame([weather_data])
        except (RequestException, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return None
        current_db_time = datetime.now(pytz.UTC)
        ts_pd["datetime"] = current_db_time
        weather_pd["datetime"] = current_db_time
        logger.info("Merging sensor and weather data...")
        merged_df = assemble_dataframe(spark, [ts_pd, weather_pd], join_key=join_key, how=how, extra_columns={
            **extra_columns,
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
        optimized_df = optimize_dataframe(final_df, partition_cols=partition_cols, partition_count=partition_count)
        save_success = db_data_transaction(spark, "save", "current", data=optimized_df)
        if save_success:
            logger.info(f"Successfully saved {optimized_df.count()} current records")
            return optimized_df
        else:
            logger.error("Failed to save current data")
            return optimized_df
    except (ValueError, TypeError) as e:
        logger.error(f"Data processing error: {e}")
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
        config = get_config()
        future_date = today + timedelta(days=7)

        # Format as strings for API calls
        today_str = today.strftime("%Y-%m-%d")
        past_str = past_date.strftime("%Y-%m-%d")
        future_str = future_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching historical data from {past_str} to {today_str}")
        logger.info(f"Preparing future data from {today_str} to {future_str}")

        # Fetch historical weather and air quality data
        hist_weather_df = historical_weather(past_str, today_str, logger)
        if hist_weather_df is None:
            logger.error("No data fetched for historical weather")
            return None
        hist_aq_df = historical_air_quality(past_str, today_str, logger)
        if hist_aq_df is None:
            logger.error("No data fetched for historical air quality")
            return None

        # Fetch future weather data (next 7 days)
        future_weather_df = future_weather(logger)
        if future_weather_df is None:
            logger.error("No data fetched for future weather")
            return None

        # Create merged historical dataframe
        historical_df = pd.merge(hist_weather_df, hist_aq_df, on="datetime", how="inner")
        historical_df["is_future"] = False
        historical_df["is_urban"] = True
        historical_df["elevation"] = config.DEBRECEN_ELEVATION

        # Prepare future dataframe with weather data only
        future_df = future_weather_df.copy()

        # Add placeholder PM columns
        pm_columns = [col for col in hist_aq_df.columns if col != "datetime"]
        for col in pm_columns:
            future_df[col] = None

        future_df["elevation"] = config.DEBRECEN_ELEVATION
        future_df["is_urban"] = True
        future_df["is_future"] = True

        # Combine datasets
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)

        # Convert to Spark DataFrame
        input_df = spark.createDataFrame(combined_df)

        if input_df.rdd.isEmpty():
            logger.warning("The dataframe is empty")
            return None
        return input_df
    except (ValueError, TypeError) as e:
        logger.error(f"Data processing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching prediction input data: {str(e)}")
        return None
        else:
            logger.error("Failed to fetch historical data")
            return pd.DataFrame()
            
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
            return pd.DataFrame()
            
        # Combine datasets
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        
        # Convert to Spark DataFrame
        input_df = spark.createDataFrame(combined_df)
        
        if input_df.rdd.isEmpty():
            logger.warning("The dataframe is empty")
        return input_df
    except (ValueError, TypeError) as e:
        logger.error(f"Data processing error: {e}")
        return None
    except Exception as e :
        logger.error(f"Error fetching prediction input data: {str(e)}")
        return None