from config import FEATURE_COLUMNS

"""
Data processing and feature engineering functions.
"""

import logging
from pyspark.sql.functions import *
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)



def add_urban_features(df):
    """
    Add engineered features to the DataFrame for urban PM10 prediction (historical data only).
    Parameters:
        df: Spark DataFrame with raw data (must include pm10, pm2_5, temperature, etc.)
    Returns:
        DataFrame: Enhanced DataFrame with additional features for historical records
    """
    logger.info("Adding urban features and calculating derived metrics...")
    try:
        # Temporal features
        df = df.withColumn("year", year("datetime")) \
               .withColumn("month", month("datetime")) \
               .withColumn("hour", hour("datetime")) \
               .withColumn("day_of_week", dayofweek("datetime")) \
               .withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))

        # Window specs for time-based aggregations
        window_lag_24h = Window.partitionBy("year", "month").orderBy("datetime")
        window_24h_avg = Window.partitionBy("year", "month").orderBy("datetime").rowsBetween(-24, 0)
        window_7d_avg = Window.partitionBy("year", "month").orderBy("datetime").rowsBetween(-168, 0)

        # Window-based features
        df = df.withColumn("pm10_lag24", lag("pm10", 24).over(window_lag_24h)) \
               .withColumn("weekly_pm10_avg", avg("pm10").over(window_7d_avg)) \
               .withColumn("pressure_trend", avg("pressure").over(window_24h_avg) - col("pressure"))

        # Wind features
        df = df.withColumn("wind_dir_8", ((col("wind_dir") + 22.5) % 360 / 45).cast("int")) \
               .withColumn("wind_speed_cat", 
                           when(col("wind_speed") < 2, 0)
                           .when(col("wind_speed") < 5, 1)
                           .when(col("wind_speed") < 10, 2)
                           .otherwise(3))

        # Pollution features
        df = df.withColumn("pm_ratio", col("pm2_5")/(col("pm10") + 1e-7)) \
               .withColumn("pollution_load", col("pm10") * col("wind_speed"))

        # Fill missing values and drop unnecessary columns
        df = df.withColumn("pm10_lag24", coalesce(col("pm10_lag24"), lit(0.0))) \
               .withColumn("weekly_pm10_avg", coalesce(col("weekly_pm10_avg"), col("pm10"))) \
               .drop("year", "day_of_week")  # Keep month/hour/is_weekend for modeling

        logger.info("Urban feature engineering completed successfully")
        return df

    except Exception as e:
        logger.error(f"Error during urban feature engineering: {e}")
        raise


def add_unified_features(df):
    """
    Enhanced feature engineering that handles both historical and future data.
    Splits on `is_future` flag and applies the appropriate feature set.

    Parameters:
        df: Spark DataFrame with raw data and boolean `is_future` column

    Returns:
        DataFrame: Enhanced DataFrame with full feature set for both historical and future rows
    """
    logger.info("Adding unified features for both historical and future data...")
    try:
        historical_df = df.filter(col("is_future") == False)
        future_df = df.filter(col("is_future") == True)

        # Process historical data (requires PM columns)
        if historical_df.count() > 0:
            logger.info("Processing historical data with full urban feature set")
            historical_df = add_urban_features(historical_df)

        # Process future data (limited features)
        if future_df.count() > 0:
            logger.info("Processing future data with limited feature set")
            # Temporal features
            future_df = future_df.withColumn("year", year("datetime")) \
                                 .withColumn("month", month("datetime")) \
                                 .withColumn("hour", hour("datetime")) \
                                 .withColumn("day_of_week", dayofweek("datetime")) \
                                 .withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))

            # Wind features
            future_df = future_df.withColumn("wind_dir_8", ((col("wind_dir") + 22.5) % 360 / 45).cast("int")) \
                                 .withColumn("wind_speed_cat", 
                                            when(col("wind_speed") < 2, 0)
                                            .when(col("wind_speed") < 5, 1)
                                            .when(col("wind_speed") < 10, 2)
                                            .otherwise(3))

            # Pressure trend
            window_24h_avg = Window.partitionBy("year", "month").orderBy("datetime").rowsBetween(-24, 0)
            future_df = future_df.withColumn("pressure_trend", avg("pressure").over(window_24h_avg) - col("pressure"))

            # Fill PM-dependent features from latest historical snapshot
            if historical_df.count() > 0:
                latest = historical_df.orderBy(desc("datetime")).limit(1).collect()[0].asDict()
                lp10 = latest["pm10"]
                lag24 = latest.get("pm10_lag24", lp10)
                week_avg = latest.get("weekly_pm10_avg", lp10)
                p25 = latest.get("pm2_5", 0)

                future_df = future_df.withColumn("pm10_lag24", lit(lag24)) \
                                     .withColumn("weekly_pm10_avg", lit(week_avg)) \
                                     .withColumn("pm_ratio", lit(p25/(lp10 + 1e-7))) \
                                     .withColumn("pollution_load", lit(lp10) * col("wind_speed"))
            else:
                # default placeholders
                future_df = future_df.withColumn("pm10_lag24", lit(0.0)) \
                                     .withColumn("weekly_pm10_avg", lit(0.0)) \
                                     .withColumn("pm_ratio", lit(0.5)) \
                                     .withColumn("pollution_load", lit(0.0))

            future_df = future_df.drop("year", "day_of_week")

        # Combine
        if historical_df.count() > 0 and future_df.count() > 0:
            result_df = historical_df.unionByName(future_df)
        elif historical_df.count() > 0:
            result_df = historical_df
        else:
            result_df = future_df

        logger.info("Unified feature engineering completed successfully")
        return result_df

    except Exception as e:
        logger.error(f"Error during unified feature engineering: {e}")
        raise

def validate_data(df):
    """
    Validate data quality and completeness.
    
    Parameters:
        df: Spark DataFrame to validate
        
    Returns:
        bool: True if data passes validation
    """
    logger.info("Validating data quality...")
    
    # Check for missing values
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    
    # Log schema and null counts
    df.printSchema()
    null_counts.show()
    
    # Get total count of rows
    total_count = df.count()
    
    # Check if there are enough records
    if total_count < 100:
        logger.warning(f"Very few records ({total_count}) in dataset - may be insufficient for training")
        return False
        
    # Check if critical columns have few nulls
    critical_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    
    # Only check PM columns if we're not dealing with future data
    if "is_future" not in df.columns or df.filter(col("is_future") == False).count() > 0:
        critical_cols.append("pm10")
    
    critical_nulls = null_counts.select(*critical_cols).collect()[0]
    
    for col_name in critical_cols:
        null_percent = 100 * getattr(critical_nulls, col_name) / total_count
        if null_percent > 5:
            logger.warning(f"Critical column {col_name} has {null_percent:.2f}% null values")
            return False
    
    logger.info("Data validation passed")
    return True

def prepare_training_data(df, test_ratio=0.2, random_seed=42):
    """
    Prepare training and testing datasets.
    
    Parameters:
        df: Spark DataFrame with processed features
        test_ratio (float): Ratio of data to use for testing
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    logger.info(f"Splitting data into training ({1-test_ratio:.0%}) and test ({test_ratio:.0%}) sets")
    
    # Split the data
    train_df, test_df = df.randomSplit([1-test_ratio, test_ratio], seed=random_seed)
    
    # Cache for performance
    train_df = train_df.cache()
    test_df = test_df.cache()
    
    logger.info(f"Training set: {train_df.count()} records")
    logger.info(f"Test set: {test_df.count()} records")
    
    return train_df, test_df