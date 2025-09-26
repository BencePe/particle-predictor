"""
Data fetching functions for the PM10 prediction project.
"""

from datetime import datetime, date, timedelta
from pyspark.sql import SparkSession
import pandas as pd
import pytz
import requests
import logging
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

cache_session = requests_cache.CachedSession('.cache', expire_after=30)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_open_meteo_data(endpoint, params):
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
        if not (callable(aq_hourly.Time) and callable(aq_hourly.TimeEnd) and callable(aq_hourly.Interval)):
            logger.error("Invalid hourly data structure in air quality response")
            return None

        time_start = pd.to_datetime(aq_hourly.Time(), unit="s", utc=True)
        time_end   = pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True)
        interval   = aq_hourly.Interval()

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
        "humidity":    weather_hourly.Variables(1).ValuesAsNumpy(),
        "pressure":    weather_hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed":  weather_hourly.Variables(3).ValuesAsNumpy(),
        "wind_dir":    weather_hourly.Variables(4).ValuesAsNumpy(),
        "precipitation": weather_hourly.Variables(5).ValuesAsNumpy()
    }
    return pd.DataFrame(weather_data)


def future_weather(start_date=None, end_date=None, past_days=None, forecast_days=None):
    params = {
        "latitude": DEBRECEN_LAT,
        "longitude": DEBRECEN_LON,
        "hourly": [
            "temperature_2m","relative_humidity_2m",
            "surface_pressure","wind_speed_10m",
            "wind_direction_10m","precipitation"
        ]
    }

    if past_days is not None or forecast_days is not None:
        if start_date or end_date:
            raise ValueError("Cannot combine start/end with past_days/forecast_days")
        if past_days is not None:
            params["past_days"] = past_days
        if forecast_days is not None:
            params["forecast_days"] = forecast_days
    else:
        params["start_date"] = start_date
        params["end_date"]   = end_date

    weather_response = fetch_open_meteo_data(
        "https://api.open-meteo.com/v1/forecast",
        params=params
    )
    if not weather_response:
        return None

    wh = weather_response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(wh.Time(),    unit="s", utc=True),
        end=  pd.to_datetime(wh.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=wh.Interval()),
        inclusive="left"
    ).tz_localize(None)

    return pd.DataFrame({
        "datetime":     times,
        "temperature":  wh.Variables(0).ValuesAsNumpy(),
        "humidity":     wh.Variables(1).ValuesAsNumpy(),
        "pressure":     wh.Variables(2).ValuesAsNumpy(),
        "wind_speed":   wh.Variables(3).ValuesAsNumpy(),
        "wind_dir":     wh.Variables(4).ValuesAsNumpy(),
        "precipitation":wh.Variables(5).ValuesAsNumpy()
    })


def assemble_and_pass(
    spark, start_date, end_date,
    is_future=False,
    redownload='n',
    db_operation='',
    table=''
):
    try:
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        today = now.date()
        API_MAX = date(2025, 5, 5)
        weather_up = min(end_date, API_MAX)
        
        if end_date > API_MAX:
            logger.warning(f"Clamping weather end_date {end_date} → {weather_up}")
        
        if not is_future:
            aq_df = historical_air_quality(start_date.isoformat(), end_date.isoformat())
            weather_df = historical_weather(start_date.isoformat(), end_date.isoformat())
                               
            if aq_df is None or weather_df is None:
                logger.error("assemble_and_pass: missing historical dataframes")
                return None

            extra_cols = {
                "elevation": DEBRECEN_ELEVATION,
                "is_urban": True,
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
            aq_df = historical_air_quality(start_date.isoformat(), now.date().isoformat())
            if aq_df is None:
                logger.error("assemble_and_pass: missing AQ up to now")
                return None

            weather_df = future_weather(
                past_days=90,
                forecast_days=7
            )
            if weather_df is None:
                logger.error("assemble_and_pass: forecast call failed")
                return None

            hours = pd.DataFrame({
                "datetime": pd.date_range(
                    start=start_date, end=end_date, freq="h"
                )
            })

            df = (
                hours
                .merge(weather_df, on="datetime", how="left")
                .merge(aq_df[["datetime","pm10"]], on="datetime", how="left")
            )

            df["elevation"] = DEBRECEN_ELEVATION
            df["is_urban"] = True
            df["is_future"] = df["datetime"] > now
            df.loc[df["is_future"], "pm10"] = 0.0

            sdf = spark.createDataFrame(df)
            final_df = optimize_dataframe(
                sdf, partition_cols="datetime", partition_count=168
            )

        db_cols = [
            "datetime", "pm10", "temperature", "humidity", "pressure",
            "wind_speed", "wind_dir", "precipitation",
            "elevation", "is_urban", "is_future"
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


def get_prediction_input_data(spark):
    now     = datetime.now().replace(minute=0, second=0, microsecond=0)
    today   = now.date()
    past_90 = today - timedelta(days=90)
    fut_end = today + timedelta(days=7)

    combo = assemble_and_pass(spark, past_90, fut_end, is_future=True)
    if combo is None:
        logger.error("get_prediction_input_data: no DF returned")
        return None

    hist_spark = combo \
        .filter((col("is_future")==False) & col("pm10").isNotNull())

    future_spark = combo \
        .filter(col("is_future")==True) \
        .select(
            "datetime","pm10","temperature","humidity",
            "pressure","wind_speed","wind_dir",
            "precipitation","elevation","is_urban","is_future"
        )

    return hist_spark, future_spark, now


def fetch_current_data(spark, thingspeak_keys=THINGSPEAK_KEYS):
    logger.info("Fetching current sensor and weather data for multiple boards...")
    results = {}
    
    try:
        spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        
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
        weather_pd["datetime"] = current_db_time
        
        for board_name, board_config in thingspeak_keys.items():
            logger.info(f"Processing board: {board_name}")
            channel_id = board_config.get("channel_id")
            read_key = board_config.get("read_key")
            field_names = board_config.get("field_names", [])
            
            if not channel_id or not read_key:
                logger.error(f"Missing configuration for board {board_name}")
                continue
                
            logger.info(f"Fetching ThingSpeak data for {board_name}...")
            ts_url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={read_key}&results=1"
            response = requests.get(ts_url)
            
            if response.status_code != 200:
                logger.error(f"ThingSpeak API failed for {board_name}: {response.status_code}")
                continue
                
            ts_data = response.json()
            feeds = ts_data.get("feeds", [])
            records = []
            
            for entry in feeds:
                dt = pd.to_datetime(entry.get("created_at"), errors='coerce')
                if dt is None or pd.isna(dt):
                    dt = current_db_time
                else:
                    dt = dt.tz_localize(None)
                
                record = {"datetime": dt}
                valid_record = False
                
                for i, field_name in enumerate(field_names, 1):
                    field_key = f"field{i}"
                    if field_key in entry and entry[field_key]:
                        try:
                            record[field_name] = float(entry[field_key])
                            valid_record = True
                        except (ValueError, TypeError):
                            record[field_name] = None
                
                if valid_record:
                    records.append(record)
            
            if not records:
                logger.warning(f"No valid ThingSpeak records found for {board_name}")
                continue
                
            ts_pd = pd.DataFrame(records)
            ts_pd["datetime"] = current_db_time
            
            logger.info(f"Merging sensor and weather data for {board_name}...")
            merged_df = assemble_dataframe(
                spark,
                [ts_pd, weather_pd],
                join_key="datetime",
                how="outer",
                extra_columns={
                    "elevation": DEBRECEN_ELEVATION,
                    "is_urban": True,
                    "data_timestamp": current_db_time.isoformat(),
                    "board_name": board_name
                }
            )
            
            merged_df = merged_df.dropDuplicates(["datetime"])
            merged_df = merged_df.withColumn("datetime", col("datetime").cast("timestamp"))
            
            db_columns = ["datetime", "pm10", "temperature", "humidity", 
                        "pressure", "wind_speed", "wind_dir", "elevation", 
                        "is_urban", "board_name"]
            
            for col_name in db_columns:
                if col_name not in merged_df.columns:
                    merged_df = merged_df.withColumn(col_name, lit(None))
            
            final_df = merged_df.select(*db_columns).dropDuplicates(["datetime"])
            optimized_df = optimize_dataframe(final_df, partition_cols="datetime", partition_count=8)
            
            table_name = f"current_{board_name}"
            save_success = db_data_transaction(spark, "save", table_name, data=optimized_df)
            
            if save_success:
                logger.info(f"Successfully saved {optimized_df.count()} records for {board_name}")
                results[board_name] = optimized_df
            else:
                logger.error(f"Failed to save data for {board_name}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error fetching current data: {str(e)}")
        return None