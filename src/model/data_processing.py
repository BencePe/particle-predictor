"""
Data processing and feature engineering functions.
"""

import logging
from pyspark.sql.functions import (
    year, month, hour, dayofweek, dayofyear, datediff,
    when, col, lag, avg, coalesce, lit,
    stddev, sin, cos, count, unix_timestamp, exp,
    min, abs, pow, sum as spark_sum
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
        .withColumn("hour", hour("datetime"))
        .withColumn("day_of_week", dayofweek("datetime"))
        .withColumn("day_of_year", dayofyear("datetime"))
        .withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))
        .withColumn("is_rush_hour", when(
            ((col("hour") >= 6) & (col("hour") <= 10)) |
            ((col("hour") >= 16) & (col("hour") <= 22)), 1
        ).otherwise(0))
        .withColumn("hour_sin", sin(2 * 3.1415926535 * col("hour") / lit(24)))
        .withColumn("hour_cos", cos(2 * 3.1415926535 * col("hour") / lit(24)))
        .withColumn("month_sin", sin(2 * 3.1415926535 * col("month") / lit(12)))
        .withColumn("month_cos", cos(2 * 3.1415926535 * col("month") / lit(12)))
        .withColumn("day_of_year_sin", sin(2 * 3.1415926535 * col("day_of_year") / lit(365)))
        .withColumn("day_of_year_cos", cos(2 * 3.1415926535 * col("day_of_year") / lit(365)))
        .withColumn("hour_weekend", col("hour") * col("is_weekend"))
        .withColumn("spring_indicator", when((col("month") >= 3) & (col("month") <= 5), 1).otherwise(0))
        .withColumn("summer_indicator", when((col("month") >= 6) & (col("month") <= 8), 1).otherwise(0))
        .withColumn("fall_indicator",   when((col("month") >= 9) & (col("month") <= 11),1).otherwise(0))
        .withColumn("winter_indicator", when((col("month") == 12)|(col("month") <= 2), 1).otherwise(0))
    )

def add_lag_features(df):
    w = Window.partitionBy("year", "month").orderBy("datetime")
    return (
        df
        .withColumn("pm10_lag3", lag("pm10", 3).over(w))
        .withColumn("pm10_lag6", lag("pm10", 6).over(w))
        .withColumn("pm10_lag12",lag("pm10",12).over(w))
        .withColumn("pm10_lag24",lag("pm10",24).over(w))
        .withColumn("pm10_lag48",lag("pm10",48).over(w))
        .withColumn("pm10_lag72",lag("pm10",72).over(w))
        .withColumn("pm10_lag168",lag("pm10",168).over(w))
    )

def add_rolling_features(df):
    windows = {
        'lag3':   Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-3,0),
        'lag6':   Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-6,0),
        'lag12':  Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-12,0),
        'lag24':  Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-24,0),
        'lag48':  Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-48,0),
        'lag72':  Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-72,0),
        'lag168': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-168,0)
    }
    return (
        df
        .withColumn("3h_pm10_avg",   avg("pm10").over(windows['lag3']))
        .withColumn("6h_pm10_avg",   avg("pm10").over(windows['lag6']))
        .withColumn("12h_pm10_avg",  avg("pm10").over(windows['lag12']))
        .withColumn("24h_pm10_avg",  avg("pm10").over(windows['lag24']))
        .withColumn("48h_pm10_avg",  avg("pm10").over(windows['lag48']))
        .withColumn("72h_pm10_avg",  avg("pm10").over(windows['lag72']))
        .withColumn("weekly_pm10_avg",avg("pm10").over(windows['lag168']))
        .withColumn("3h_pm10_std",   stddev("pm10").over(windows['lag3']))
        .withColumn("6h_pm10_std",   stddev("pm10").over(windows['lag6']))
        .withColumn("12h_pm10_std",  stddev("pm10").over(windows['lag12']))
        .withColumn("24h_pm10_std",  stddev("pm10").over(windows['lag24']))
        .withColumn("weekly_pm10_std",stddev("pm10").over(windows['lag168']))
        .withColumn("3h_temp_avg",   avg("temperature").over(windows['lag3']))
        .withColumn("12h_temp_avg",  avg("temperature").over(windows['lag12']))
        .withColumn("3h_humidity_avg",avg("humidity").over(windows['lag3']))
        .withColumn("12h_humidity_avg",avg("humidity").over(windows['lag12']))
        .withColumn("3h_pressure_avg", avg("pressure").over(windows['lag3']))
        .withColumn("12h_pressure_avg",avg("pressure").over(windows['lag12']))
        .withColumn("3h_wind_speed_avg",avg("wind_speed").over(windows['lag3']))
        .withColumn("12h_wind_speed_avg",avg("wind_speed").over(windows['lag12']))
    )

def add_diff_and_volatility_features(df):
    w = Window.partitionBy("year","month").orderBy("datetime")
    return (
        df
        .withColumn("pm10_volatility_3h",  col("3h_pm10_std")  / (col("3h_pm10_avg")  + lit(1e-5)))
        .withColumn("pm10_volatility_6h",  col("6h_pm10_std")  / (col("6h_pm10_avg")  + lit(1e-5)))
        .withColumn("pm10_volatility_12h", col("12h_pm10_std") / (col("12h_pm10_avg") + lit(1e-5)))
        .withColumn("pm10_volatility_24h", col("24h_pm10_std") / (col("24h_pm10_avg") + lit(1e-5)))
        .withColumn("pm10_diff_3h",  coalesce(col("pm10") - col("pm10_lag3"), lit(0.0)))
        .withColumn("pm10_diff_6h",  coalesce(col("pm10") - col("pm10_lag6"), lit(0.0)))
        .withColumn("pm10_diff_12h", coalesce(col("pm10") - col("pm10_lag12"),lit(0.0)))
        .withColumn("pm10_diff_24h", coalesce(col("pm10") - col("pm10_lag24"),lit(0.0)))
        .withColumn("pm10_diff_48h", coalesce(col("pm10") - col("pm10_lag48"),lit(0.0)))
        .withColumn("pm10_acceleration_3h",  coalesce(col("pm10_diff_3h")  - lag("pm10_diff_3h",3).over(w), lit(0.0)))
        .withColumn("pm10_acceleration_12h", coalesce(col("pm10_diff_12h") - lag("pm10_diff_12h",12).over(w),lit(0.0)))
        .withColumn("pm10_acceleration",     coalesce(col("pm10_diff_24h") - lag("pm10_diff_24h",24).over(w),lit(0.0)))
        .withColumn("pm10_rate_of_change_3h", col("pm10_diff_3h")  / (col("pm10_lag3")  + lit(1e-5)))
        .withColumn("pm10_rate_of_change_12h",col("pm10_diff_12h") / (col("pm10_lag12") + lit(1e-5)))
    )

def add_precipitation_features(df):
    w      = Window.partitionBy("year","month").orderBy("datetime")
    w24    = w.rowsBetween(-24,0)
    base   = df
    if "precipitation" in df.columns:
        base = (
            df
            .withColumn("is_precipitation", when(col("precipitation") > 0.1, 1).otherwise(0))
            .withColumn("precipitation_intensity",
                        when(col("precipitation") < 0.5, 0)
                       .when(col("precipitation") < 5,   1)
                       .otherwise(2))
            .withColumn("recent_rain",       coalesce(lag("is_precipitation",1).over(w), lit(0)))
            .withColumn("rain_last_6h",     when(spark_sum("is_precipitation").over(w.rowsBetween(-6,0)) > 0, 1).otherwise(0))
            .withColumn("rain_last_12h",    when(spark_sum("is_precipitation").over(w.rowsBetween(-12,0))> 0, 1).otherwise(0))
            .withColumn("rain_last_24h",    when(spark_sum("is_precipitation").over(w24)              > 0, 1).otherwise(0))
            .withColumn("cumulative_24h_precip", spark_sum(coalesce(col("precipitation"),lit(0))).over(w24))
        )
    else:
        # no precipitation → all zeros
        for c in [
            "is_precipitation", "precipitation_intensity",
            "recent_rain", "rain_last_6h", "rain_last_12h", "rain_last_24h",
            "cumulative_24h_precip"
        ]:
            base = base.withColumn(c, lit(0))
    return base

def add_interaction_and_trend_features(df):
    windows = {
        'lag3':  Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-3, 0),
        'lag6':  Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-6, 0),
        'lag12': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-12,0),
        'lag24': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-24,0)
    }
    w = Window.partitionBy("year","month").orderBy("datetime")
    return (
        df
        .withColumn("wind_speed_humidity", col("wind_speed") * col("humidity"))
        .withColumn("temp_pressure",         col("temperature") * col("pressure"))
        .withColumn("dew_point",             col("temperature") - (lit(100) - col("humidity"))/lit(5))
        .withColumn("pm10_dew_point",        col("pm10") * col("dew_point"))
        .withColumn("pollution_drift_week",  (col("pm10") - col("weekly_pm10_avg"))/(col("weekly_pm10_avg") + lit(1e-5)))
        .withColumn("pressure_3h_trend", coalesce(avg("pressure").over(windows['lag3'])  - col("pressure"), lit(0.0)))
        .withColumn("pressure_6h_trend", coalesce(avg("pressure").over(windows['lag6'])  - col("pressure"), lit(0.0)))
        .withColumn("pressure_12h_trend",coalesce(avg("pressure").over(windows['lag12']) - col("pressure"), lit(0.0)))
        .withColumn("pressure_24h_trend",coalesce(avg("pressure").over(windows['lag24']) - col("pressure"), lit(0.0)))
        .withColumn("temp_3h_trend",     coalesce(avg("temperature").over(windows['lag3'])   - col("temperature"), lit(0.0)))
        .withColumn("temp_6h_trend",     coalesce(avg("temperature").over(windows['lag6'])   - col("temperature"), lit(0.0)))
        .withColumn("temp_12h_trend",    coalesce(avg("temperature").over(windows['lag12'])  - col("temperature"), lit(0.0)))
        .withColumn("temp_24h_trend",    coalesce(avg("temperature").over(windows['lag24'])  - col("temperature"), lit(0.0)))
        .withColumn("humidity_3h_trend", coalesce(avg("humidity").over(windows['lag3'])      - col("humidity"), lit(0.0)))
        .withColumn("humidity_12h_trend",coalesce(avg("humidity").over(windows['lag12'])     - col("humidity"), lit(0.0)))
        .withColumn("humidity_temp_index", col("humidity") * col("temperature"))
        .withColumn("humidity_pressure_index", col("humidity") * col("pressure") / lit(1000))
        .withColumn("temp_wind_index",      col("temperature") * col("wind_speed"))
        .withColumn("pressure_change_velocity", (col("pressure") - lag("pressure",3).over(w)) / lit(3.0))
        .withColumn("rapid_pressure_change", when(abs(col("pressure") - lag("pressure",6).over(w)) > 5, 1).otherwise(0))
        .withColumn("rapid_humidity_increase", when((col("humidity") - lag("humidity",3).over(w)) > 15,1).otherwise(0))
        .withColumn("wind_dir_stability", stddev(col("wind_dir")).over(windows['lag24']) / lit(180.0))
        .withColumn("wind_dir_8", ((col("wind_dir") + 22.5) % 360 / 45).cast("int"))
        .withColumn("wind_speed_cat", when(col("wind_speed")<2,0)
                                     .when(col("wind_speed")<5,1)
                                     .when(col("wind_speed")<10,2)
                                     .otherwise(3))
        .withColumn("pollution_load", col("pm10") * col("wind_speed"))
        .withColumn("pm10_12h_avg_sq", pow(col("12h_pm10_avg"),2))
        .withColumn("avg12h_times_diff12h", col("12h_pm10_avg") * col("pm10_diff_12h"))
        .withColumn("temp_humidity_interaction",      col("temperature") * col("humidity")  / lit(100))
        .withColumn("temp_pressure_interaction",      col("temperature") * col("pressure")  / lit(1000))
        .withColumn("wind_temp_cross", when((col("temperature")>15)&(col("wind_speed")>5),1).otherwise(0))
    )

def add_seasonal_indicators(df):
    return df.withColumn(
        "spring_indicator",
        when((month("datetime") >= 3) & (month("datetime") <= 5), 1).otherwise(0)
    ).withColumn(
        "summer_indicator",
        when((month("datetime") >= 6) & (month("datetime") <= 8), 1).otherwise(0)
    ).withColumn(
        "fall_indicator",
        when((month("datetime") >= 9) & (month("datetime") <= 11), 1).otherwise(0)
    ).withColumn(
        "winter_indicator",
        when((month("datetime") == 12) | (month("datetime") <= 2), 1).otherwise(0)
    )

def add_weather_change_features(df):
    w = Window.partitionBy("year","month").orderBy("datetime")
    return (
        df
        .withColumn("weather_change_index",
                    abs(col("pressure_6h_trend"))/lit(5) +
                    abs(col("temp_6h_trend"))/lit(3) +
                    abs(col("humidity_3h_trend"))/lit(10))
        .withColumn("atmospheric_instability",
                    abs(col("pressure_change_velocity")) *
                    (col("3h_pm10_std") / (col("3h_pm10_avg") + lit(1))))
        .withColumn("weather_system_change",
            when((abs(col("pressure_24h_trend")) > 10) |
                 (abs(col("temp_24h_trend"))     > 5) |
                 (col("rapid_pressure_change")    == 1), 1).otherwise(0))
    )

def add_weather_lag_features(df):
    w = Window.partitionBy("year", "month").orderBy("datetime")
    return (
        df
        .withColumn("temp_lag3", lag("temperature", 3).over(w))
        .withColumn("temp_lag6", lag("temperature", 6).over(w))
        .withColumn("temp_lag12", lag("temperature", 12).over(w))
        .withColumn("temp_lag24", lag("temperature", 24).over(w))
        .withColumn("humidity_lag3", lag("humidity", 3).over(w))
        .withColumn("humidity_lag6", lag("humidity", 6).over(w))
        .withColumn("humidity_lag12", lag("humidity", 12).over(w))
        .withColumn("humidity_lag24", lag("humidity", 24).over(w))
        .withColumn("pressure_lag3", lag("pressure", 3).over(w))
        .withColumn("pressure_lag6", lag("pressure", 6).over(w))
        .withColumn("pressure_lag12", lag("pressure", 12).over(w))
        .withColumn("pressure_lag24", lag("pressure", 24).over(w))
        .withColumn("wind_speed_lag3", lag("wind_speed", 3).over(w))
        .withColumn("wind_speed_lag6", lag("wind_speed", 6).over(w))
        .withColumn("wind_speed_lag12", lag("wind_speed", 12).over(w))
        .withColumn("wind_speed_lag24", lag("wind_speed", 24).over(w))
    )

def add_weather_rate_of_change(df):
    w = Window.partitionBy("year", "month").orderBy("datetime")
    return (
        df
        .withColumn("temp_diff_3h", coalesce(col("temperature") - col("temp_lag3"), lit(0.0)))
        .withColumn("temp_diff_6h", coalesce(col("temperature") - col("temp_lag6"), lit(0.0)))
        .withColumn("temp_diff_12h", coalesce(col("temperature") - col("temp_lag12"), lit(0.0)))
        .withColumn("temp_diff_24h", coalesce(col("temperature") - col("temp_lag24"), lit(0.0)))
        .withColumn("humidity_diff_3h", coalesce(col("humidity") - col("humidity_lag3"), lit(0.0)))
        .withColumn("humidity_diff_6h", coalesce(col("humidity") - col("humidity_lag6"), lit(0.0)))
        .withColumn("humidity_diff_12h", coalesce(col("humidity") - col("humidity_lag12"), lit(0.0)))
        .withColumn("pressure_diff_3h", coalesce(col("pressure") - col("pressure_lag3"), lit(0.0)))
        .withColumn("pressure_diff_6h", coalesce(col("pressure") - col("pressure_lag6"), lit(0.0)))
        .withColumn("pressure_diff_12h", coalesce(col("pressure") - col("pressure_lag12"), lit(0.0)))
        .withColumn("wind_speed_diff_3h", coalesce(col("wind_speed") - col("wind_speed_lag3"), lit(0.0)))
        .withColumn("wind_speed_diff_6h", coalesce(col("wind_speed") - col("wind_speed_lag6"), lit(0.0)))
        .withColumn("wind_speed_diff_12h", coalesce(col("wind_speed") - col("wind_speed_lag12"), lit(0.0)))
        .withColumn("temp_rate_3h", col("temp_diff_3h") / (coalesce(col("temp_lag3"), lit(1.0)) + lit(1e-5)))
        .withColumn("pressure_rate_3h", col("pressure_diff_3h") / (coalesce(col("pressure_lag3"), lit(1000.0)) + lit(1e-5)))
        .withColumn("humidity_rate_3h", col("humidity_diff_3h") / (coalesce(col("humidity_lag3"), lit(50.0)) + lit(1e-5)))
    )

def add_weather_regime_indicators(df):
    return (
        df
        .withColumn("heat_wave", when(
            (col("temperature") > 30) & 
            (col("temp_lag24") > 28) & 
            (col("humidity") < 40), 
            1).otherwise(0))
        .withColumn("cold_front", when(
            (col("temp_diff_12h") < -5) & 
            (col("pressure_diff_12h") > 5),
            1).otherwise(0))
        .withColumn("warm_front", when(
            (col("temp_diff_12h") > 5) & 
            (col("pressure_diff_12h") < -5),
            1).otherwise(0))
        .withColumn("approaching_storm", when(
            (col("pressure_diff_6h") < -2) &
            (col("wind_speed") > 5) &
            (col("humidity") > 70),
            1).otherwise(0))
        .withColumn("stagnant_air", when(
            (col("wind_speed") < 2) &
            (col("wind_speed_lag3") < 2) &
            (abs(col("pressure_diff_12h")) < 1),
            1).otherwise(0))
        .withColumn("air_mass_replacement", when(
            (abs(col("temp_diff_12h")) > 3) &
            (abs(col("humidity_diff_12h")) > 15) &
            (col("wind_speed") > 3),
            1).otherwise(0))
    )

def add_urban_features(df):
    """
    Enhanced urban features for PM10 prediction with consistent feature set.
    """
    logger.info("Adding enhanced urban features...")
    try:
        df = add_temporal_features(df)
        df = add_weather_lag_features(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_diff_and_volatility_features(df)
        df = add_precipitation_features(df)
        df = add_interaction_and_trend_features(df)
        df = add_weather_change_features(df)
        df = add_seasonal_indicators(df)
        df = add_weather_rate_of_change(df)
        df = add_weather_regime_indicators(df)
        
        
        feature_cols = [c for c in df.columns if c in FEATURE_COLUMNS]
        df = df.na.fill(0.0, subset=feature_cols)
        df = df.drop("year", "day_of_week", "day_of_year")
        logger.info("Enhanced urban features added.")
        return df

    except Exception as e:
        logger.error(f"Error in add_urban_features: {e}")
        raise

def add_unified_features(df):
    """
    Apply urban features consistently to historical & future data.
    """
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
    """
    Validate data quality, completeness, and required column presence.
    
    Enhanced with column existence checks and improved validation logic
    """
    logger.info("Validating data...")
    
    # 1. Check for required column presence
    required_columns = {"datetime", "temperature", "humidity", 
                       "pressure", "wind_speed", "is_future"}
    
    if df.filter(col("is_future") == False).count() > 0:
        required_columns.add("pm10")
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # 2. Schema validation
    logger.debug("Current schema:")
    df.printSchema()
    
    # 3. Null value analysis
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    null_data = null_counts.collect()[0]
    
    logger.info("Null counts per column:")
    null_counts.show()
    
    # 4. Record count check
    total = df.count()
    if total < 100:
        logger.warning(f"Low record count: {total} samples")
        return False
    
    # 5. Critical column null percentage check
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

def prepare_training_data(df, test_ratio=0.2): 
    """
    Time-based split for training and test.
    """
    logger.info(f"Splitting {int((1-test_ratio)*100)}% train / {int(test_ratio*100)}% test by datetime")
    df = df.orderBy("datetime")
    total = df.count()
    split = int(total*(1-test_ratio))
    train = df.limit(split).cache()
    test  = df.subtract(train).cache()
    logger.info(f"Train: {train.count()} rows, Test: {test.count()} rows")
    return train, test