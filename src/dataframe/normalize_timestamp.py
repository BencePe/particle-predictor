import logging
from pyspark.sql.functions import col, to_timestamp, lit
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame


def normalize_timestamp(spark_df: DataFrame, timestamp_col: str = "datetime") -> DataFrame:
    """Normalize timestamps to a consistent format.

    Ensures timestamps are in 'yyyy-MM-dd HH:mm:ss' format without timezone
    information, which is useful for compatibility with databases like
    TimescaleDB. If the timestamp column cannot be converted, an error column
    is added to the dataframe indicating the failure.

    Args:
        spark_df (DataFrame): The input Spark DataFrame containing a timestamp column.
        timestamp_col (str, optional): The name of the timestamp column.
            Defaults to "datetime".

    Returns:
        DataFrame: A new Spark DataFrame with the timestamp column normalized.
            If the normalization fails, a new column named "{timestamp_col}_error"
            is added with the value "Failed to normalize".

    Raises:
        pyspark.sql.utils.AnalysisException: If the timestamp column has an
            incompatible type or format.
        Exception: If an unexpected error occurs during the normalization.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("NormalizeTimestamp").getOrCreate()
        >>> data = [("2023-01-01 10:00:00",), ("2023-01-02 11:00:00",)]
        >>> df = spark.createDataFrame(data, ["datetime"])
        >>> normalized_df = normalize_timestamp(df)
        >>> normalized_df.show()
    """
    logger = logging.getLogger(__name__)
    try:
        if spark_df is None:
            return None
        normalized_df = spark_df.withColumn(
            timestamp_col,
            to_timestamp(col(timestamp_col).cast("string"), "yyyy-MM-dd HH:mm:ss")
        )
        return normalized_df
    except Exception as e:
        if "AnalysisException" in str(e):
            logger.error(f"Error normalizing timestamp column '{timestamp_col}': Incorrect format or incompatible type. Error: {e}")
            return spark_df.withColumn(f"{timestamp_col}_error", lit("Failed to normalize"))
        else:
            logger.error(f"An unexpected error occurred during timestamp normalization: {e}")
            return spark_df.withColumn(f"{timestamp_col}_error", lit("Failed to normalize"))
