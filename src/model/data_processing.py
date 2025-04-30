"""
Data processing and feature engineering functions.
"""

import logging
from pyspark.sql.functions import (
    year, month, hour, dayofweek, dayofyear, datediff,
    when, col, lag, avg, coalesce, lit,
    stddev, sin, cos, count, unix_timestamp, exp,
    min, abs, pow, sum as spark_sum, dayofmonth, max as spark_max, min as spark_min
)
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from src.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def add_temporal_features(df):
    return (
        df
        .withColumn("year", year("datetime"))
        .withColumn("month", month("datetime"))
        .withColumn("day", dayofmonth("datetime"))
        .withColumn("hour", hour("datetime"))
        .withColumn("day_of_week", dayofweek("datetime"))
        .withColumn("is_rush_hour", when(
            ((col("hour") >= 6) & (col("hour") <= 10)) |
            ((col("hour") >= 16) & (col("hour") <= 22)), 1
        ).otherwise(0))
        .withColumn("hour_sin", sin(2 * 3.1415926535 * col("hour") / lit(24)))
        .withColumn("hour_cos", cos(2 * 3.1415926535 * col("hour") / lit(24)))
        .withColumn("week_sin", sin(2 * 3.1415926535 * col("day_of_week") / lit(7)))
        .withColumn("week_cos", cos(2 * 3.1415926535 * col("day_of_week") / lit(7)))
        .withColumn("month_sin", sin(2 * 3.1415926535 * col("month") / lit(12)))
        .withColumn("month_cos", cos(2 * 3.1415926535 * col("month") / lit(12)))
        .withColumn("spring_indicator", when((col("month") >= 3) & (col("month") <= 5), 1).otherwise(0))
        .withColumn("summer_indicator", when((col("month") >= 6) & (col("month") <= 8), 1).otherwise(0))
        .withColumn("fall_indicator",   when((col("month") >= 9) & (col("month") <= 11),1).otherwise(0))
        .withColumn("winter_indicator", when((col("month") == 12)|(col("month") <= 2), 1).otherwise(0))
    )

def add_lag_features(df):
    w = Window.orderBy("datetime")
    return (
        df
        .withColumn("pm10_lag3", coalesce(lag("pm10", 3).over(w)))
        .withColumn("pm10_lag12",coalesce(lag("pm10",12).over(w)))
        .withColumn("pm10_lag24",coalesce(lag("pm10",24).over(w)))
        .withColumn("pm10_lag168",coalesce(lag("pm10",168).over(w)))
    )

def add_rolling_features(df):
    
    w_month = Window.partitionBy("month").orderBy("datetime")
    
    windows = {
        'lag3':   w_month.rowsBetween(-3,0),
        'lag24':  w_month.rowsBetween(-24,0),
    }
    return (
        df
        .withColumn("3h_pm10_avg",   coalesce(avg("pm10").over(windows['lag3'])))
        .withColumn("24h_pm10_avg",  coalesce(avg("pm10").over(windows['lag24'])))
        .withColumn("3h_pm10_std",   coalesce(stddev("pm10").over(windows['lag3'])))
        .withColumn("24h_pm10_std",  coalesce(stddev("pm10").over(windows['lag24'])))
        .withColumn("3h_pressure_avg", coalesce(avg("pressure").over(windows['lag3'])))
        .withColumn("3h_wind_speed_avg",coalesce(avg("wind_speed").over(windows['lag3'])))
        .withColumn("24h_wind_speed_avg",coalesce(avg("wind_speed").over(windows['lag24'])))
        .withColumn("rolling_max_pm10_24h", coalesce(spark_max("pm10").over(windows['lag24'])))
        .withColumn("rolling_min_pm10_24h", coalesce(spark_min("pm10").over(windows['lag24'])))
    )

def add_diff_and_volatility_features(df):
    w = Window.partitionBy("month").orderBy("datetime")
    return (
        df
        .withColumn("pm10_volatility_3h",  col("3h_pm10_std")  / (col("3h_pm10_avg")  + lit(1e-5)))
        .withColumn("pm10_diff_3h",  coalesce(col("pm10") - col("pm10_lag3"), lit(0.0)))
        .withColumn("pm10_diff_12h", coalesce(col("pm10") - col("pm10_lag12"),lit(0.0)))
        .withColumn("pm10_diff_24h", coalesce(col("pm10") - col("pm10_lag24"),lit(0.0)))
    )

def add_precipitation_features(df):
    w = Window.partitionBy("month").orderBy("datetime")
    w24 = w.rowsBetween(-24,0)
    base = df
    
    if "precipitation" in df.columns:
        base = (
            df
            .withColumn("is_precipitation", when(col("precipitation") > 0.1, 1).otherwise(0))
            .withColumn("precipitation_intensity",
                        when(col("precipitation") < 0.5, 0)
                       .when(col("precipitation") < 5,   1)
                       .otherwise(2)))
    else:
        for c in [
            "is_precipitation", "precipitation_intensity"
        ]:
            base = base.withColumn(c, lit(0))
    return base

def add_interaction_and_trend_features(df):
    w_month = Window.partitionBy("month").orderBy("datetime")
    
    windows = {
        'lag3':  w_month.rowsBetween(-3, 0),
        'lag24': w_month.rowsBetween(-24,0)
    }
    return (
        df
        .withColumn("dew_point", coalesce(col("temperature") - ((lit(100) - col("humidity")) / lit(5)), lit(0.0)))
        .withColumn("temp_24h_trend",    coalesce(avg("temperature").over(windows['lag24'])  - col("temperature"), lit(0.0)))
        .withColumn("pollution_load", coalesce(col("pm10") * col("wind_speed")))
    )
    
def add_weather_lag_features(df):
    w = Window.partitionBy("month").orderBy("datetime")
    return (
        df
        .withColumn("temp_lag24", coalesce(lag("temperature", 24).over(w)))
    )

def add_weather_rate_of_change(df):
    w = Window.partitionBy("month").orderBy("datetime")
    return (
        df
        .withColumn("temp_diff_24h", coalesce(col("temperature") - col("temp_lag24"), lit(0.0)))
    )

def add_debrecen_specific_features(df):
    """Adds features specific to Debrecen's environment"""
    w_month = Window.partitionBy("month").orderBy("datetime")
    w72 = w_month.rowsBetween(-72, 0)

    return (
        df
        .withColumn("stagnation_index", 
            when(
                (col("wind_speed") < 2.5) & 
                (abs(col("pressure") - lag("pressure", 12).over(w_month)) < 1.0) & 
                (col("humidity") > 70), 
                1
            ).otherwise(0)
        )
        .withColumn(
            "heating_effect", 
            when(
                (col("winter_indicator") == 1) & 
                (col("temperature") < 2) & 
                (col("humidity") > 80), 
                1
            ).otherwise(0)
        )
        .withColumn("dry_spell_hours", 
        spark_sum(when(col("precipitation") == 0, 1).otherwise(0)).over(w72))
        .withColumn("dry_spell_days",(col("dry_spell_hours") / 24).cast("int")
)
    )
    
def add_urban_features(df):
    logger.info("Adding enhanced urban features...")
    try:        
        df = add_temporal_features(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_weather_lag_features(df)
        df = add_weather_rate_of_change(df)
        df = add_diff_and_volatility_features(df)
        df = add_precipitation_features(df)
        df = add_interaction_and_trend_features(df)
        df = add_debrecen_specific_features(df)
        
        feature_cols = [c for c in df.columns if c in FEATURE_COLUMNS]
        df = df.na.fill(0.0, subset=feature_cols)
        df = df.drop("year", "month", "day", "hour", "day_of_week")
        logger.info("Enhanced urban features added.")
        return df

    except Exception as e:
        logger.error(f"Error in add_urban_features: {e}")
        raise

def add_unified_features(df):
    logger.info("Adding unified features...")
    try:
        hist = df.filter(col("is_future") == False)
        fut  = df.filter(col("is_future")  == True)

        if hist.count() > 0:
            hist = add_urban_features(hist)
        if fut.count()  > 0:
            fut  = add_urban_features(fut)

        if hist.count()>0 and fut.count()>0:
            return hist.unionByName(fut)
        elif hist.count()>0:
            return hist
        else:
            return fut

    except Exception as e:
        logger.error(f"Error in add_unified_features: {e}")
        raise

def validate_data(df):
    logger.info("Validating data...")
    
    required_columns = {"datetime", "temperature", "humidity", 
                       "pressure", "wind_speed", "is_future"}
    
    if df.filter(col("is_future") == False).count() > 0:
        required_columns.add("pm10")
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.debug("Current schema:")
    df.printSchema()
    
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    null_data = null_counts.collect()[0]
    
    logger.info("Null counts per column:")
    null_counts.show()
    
    total = df.count()
    if total < 100:
        logger.warning(f"Low record count: {total} samples")
        return False
    
    critical_columns = ["temperature", "humidity", "pressure", "wind_speed"]
    if "pm10" in df.columns:
        critical_columns.append("pm10")
    
    validation_passed = True
    for col_name in critical_columns:
        null_count = getattr(null_data, col_name)
        null_pct = (null_count / total) * 100
        
        if null_pct > 5:
            logger.warning(f"High null percentage in {col_name}: {null_pct:.1f}%")
            validation_passed = False
        elif null_pct > 0:
            logger.info(f"{col_name} has {null_pct:.1f}% nulls")
                
    if validation_passed:
        logger.info("Data validation passed")
    else:
        logger.error("Data validation failed")
        
    return validation_passed