"""
Data processing and feature engineering functions.
"""

import logging
from pyspark.sql.functions import *
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

def add_urban_features(df):
    """
    Add engineered features to the DataFrame for urban PM10 prediction.
    
    Parameters:
        df: Spark DataFrame with raw data
        
    Returns:
        DataFrame: Enhanced DataFrame with additional features
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
                
        logger.info("Feature engineering completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
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
    critical_cols = ["pm10", "temperature", "humidity", "pressure", "wind_speed"]
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