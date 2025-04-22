"""
Data processing and feature engineering functions.
"""

import logging
from pyspark.sql.functions import (
    year, month, hour, dayofweek, dayofyear, datediff,
    when, col, lag, avg, coalesce, lit,
    stddev, sin, cos, count, unix_timestamp, exp,
    min
)
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from src.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def add_urban_features(df):
    """
    Enhanced urban features for PM10 prediction with consistent feature set for historical & future.
    """
    logger.info("Adding enhanced urban features...")
    try:
        # Temporal features
        df = (
            df
            .withColumn("year", year("datetime"))
            .withColumn("month", month("datetime"))
            .withColumn("hour", hour("datetime"))
            .withColumn("day_of_week", dayofweek("datetime"))
            .withColumn("day_of_year", dayofyear("datetime"))
            .withColumn(
                "is_weekend",
                when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0)
            )
            .withColumn(
                "is_rush_hour",
                when(((col("hour") >= 7) & (col("hour") <= 9)) |
                     ((col("hour") >= 16) & (col("hour") <= 19)), 1).otherwise(0)
            )
        )

        # Cyclical encodings
        df = (
            df
            .withColumn("hour_sin", sin(2 * 3.1415926535 * col("hour") / lit(24)))
            .withColumn("hour_cos", cos(2 * 3.1415926535 * col("hour") / lit(24)))
            .withColumn("month_sin", sin(2 * 3.1415926535 * col("month") / lit(12)))
            .withColumn("month_cos", cos(2 * 3.1415926535 * col("month") / lit(12)))
            .withColumn("day_of_year_sin", sin(2 * 3.1415926535 * col("day_of_year") / lit(365)))
            .withColumn("day_of_year_cos", cos(2 * 3.1415926535 * col("day_of_year") / lit(365)))
            .withColumn("hour_weekend", col("hour") * col("is_weekend"))
        )

        # Window specifications
        windows = { 
            'lag12': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-12, 0),
            'lag24': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-24, 0),
            'lag48': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-48, 0),
            'lag72': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-72, 0),
            'lag168': Window.partitionBy("year","month").orderBy("datetime").rowsBetween(-168, 0)
        }
        w = Window.partitionBy("year","month").orderBy("datetime")

        # Lag features
        df = (
            df
            .withColumn("pm10_lag12", lag("pm10", 12).over(w))
            .withColumn("pm10_lag24", lag("pm10", 24).over(w))
            .withColumn("pm10_lag48", lag("pm10", 48).over(w))
            .withColumn("pm10_lag72", lag("pm10", 72).over(w))
            .withColumn("pm10_lag168", lag("pm10", 168).over(w))
        )

        # Rolling stats
        df = (
            df
            .withColumn("12h_pm10_avg", avg("pm10").over(windows['lag12']))
            .withColumn("24h_pm10_avg", avg("pm10").over(windows['lag24']))
            .withColumn("48h_pm10_avg", avg("pm10").over(windows['lag48']))
            .withColumn("72h_pm10_avg", avg("pm10").over(windows['lag72']))
            .withColumn("weekly_pm10_avg", avg("pm10").over(windows['lag168']))
            .withColumn("12h_pm10_std", stddev("pm10").over(windows['lag12']))
            .withColumn("24h_pm10_std", stddev("pm10").over(windows['lag24']))
            .withColumn("weekly_pm10_std", stddev("pm10").over(windows['lag168']))
        )

        # Volatility, diffs, acceleration
        df = (
            df
            .withColumn("pm10_volatility_24h", col("24h_pm10_std")/(col("24h_pm10_avg")+lit(1e-5)))
            .withColumn("pm10_diff_12h", coalesce(col("pm10")-col("pm10_lag12"), lit(0.0)))
            .withColumn("pm10_diff_24h", coalesce(col("pm10")-col("pm10_lag24"), lit(0.0)))
            .withColumn("pm10_diff_48h", coalesce(col("pm10")-col("pm10_lag48"), lit(0.0)))
            .withColumn("pm10_acceleration", coalesce(col("pm10_diff_24h")-lag("pm10_diff_24h",24).over(w), lit(0.0)))
        )

        # Trends & interactions
        df = (
            df
            .withColumn("pressure_12h_trend", coalesce(avg("pressure").over(windows['lag12'])-col("pressure"), lit(0.0)))
            .withColumn("pressure_24h_trend", coalesce(avg("pressure").over(windows['lag24'])-col("pressure"), lit(0.0)))
            .withColumn("temp_12h_trend", coalesce(avg("temperature").over(windows['lag12'])-col("temperature"), lit(0.0)))
            .withColumn("temp_24h_trend", coalesce(avg("temperature").over(windows['lag24'])-col("temperature"), lit(0.0)))
            .withColumn("humidity_temp_index", col("humidity")*col("temperature"))
            .withColumn("temp_wind_index", col("temperature")*col("wind_speed"))
            .withColumn("pressure_change_velocity", (col("pressure")-lag("pressure",3).over(w))/lit(3.0))
        )

        # Wind & pollution
        df = (
            df
            .withColumn("wind_dir_stability", stddev(col("wind_dir")).over(windows['lag24'])/lit(180.0))
            .withColumn("wind_dir_8", ((col("wind_dir")+22.5)%360/45).cast("int"))
            .withColumn("wind_speed_cat",
                        when(col("wind_speed")<2,0)
                        .when(col("wind_speed")<5,1)
                        .when(col("wind_speed")<10,2)
                        .otherwise(3))
            .withColumn("pm_ratio", col("pm2_5")/(col("pm10")+lit(1e-7)))
            .withColumn("pollution_load", col("pm10")*col("wind_speed"))
            .withColumn("pm10_12h_avg_sq", pow(col("12h_pm10_avg"), 2))
            .withColumn("avg12h_times_diff12h", col("12h_pm10_avg") * col("pm10_diff_12h"))
        )

                # Fill any remaining nulls in feature columns
        feature_cols = [c for c in df.columns if c in FEATURE_COLUMNS]
        df = df.na.fill(0.0, subset=feature_cols)

        # Cleanup
        df = df.drop("year","day_of_week","day_of_year")
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
        hist = df.filter(col("is_future")==False)
        fut  = df.filter(col("is_future")==True)

        if hist.count()>0:
            hist = add_urban_features(hist)
        if fut.count()>0:
            # start with raw future
            fut = add_urban_features(fut)
            # optionally apply decay logic here (using add_default_features or custom)

        # Combine
        if hist.count()>0 and fut.count()>0:
            return hist.unionByName(fut)
        elif hist.count()>0:
            return hist
        else:
            return fut
    except Exception as e:
        logger.error(f"Error in add_unified_features: {e}")
        raise

def add_default_features(df):
    """Defaults if no history present"""
    return (
        df
        .withColumn("pm10_lag12", lit(0.0))
        .withColumn("pm10_lag24", lit(0.0))
        .withColumn("pm10_lag48", lit(0.0))
        .withColumn("pm10_lag72", lit(0.0))
        .withColumn("pm10_lag168", lit(0.0))
        .withColumn("12h_pm10_avg", lit(0.0))
        .withColumn("24h_pm10_avg", lit(0.0))
        .withColumn("48h_pm10_avg", lit(0.0))
        .withColumn("72h_pm10_avg", lit(0.0))
        .withColumn("weekly_pm10_avg", lit(0.0))
        .withColumn("12h_pm10_std", lit(0.0))
        .withColumn("24h_pm10_std", lit(0.0))
        .withColumn("weekly_pm10_std", lit(0.0))
        .withColumn("pm10_volatility_24h", lit(0.1))
        .withColumn("pm10_diff_12h", lit(0.0))
        .withColumn("pm10_diff_24h", lit(0.0))
        .withColumn("pm10_diff_48h", lit(0.0))
        .withColumn("pm10_acceleration", lit(0.0))
        .withColumn("pressure_12h_trend", lit(0.0))
        .withColumn("pressure_24h_trend", lit(0.0))
        .withColumn("temp_12h_trend", lit(0.0))
        .withColumn("temp_24h_trend", lit(0.0))
        .withColumn("humidity_temp_index", lit(0.0))
        .withColumn("temp_wind_index", lit(0.0))
        .withColumn("pressure_change_velocity", lit(0.0))
        .withColumn("wind_dir_stability", lit(0.0))
        .withColumn("wind_dir_8", lit(0))
        .withColumn("wind_speed_cat", lit(0))
        .withColumn("pm_ratio", lit(0.5))
        .withColumn("pollution_load", lit(0.0))
    )

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
