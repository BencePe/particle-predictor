from pyspark.sql.functions import col, to_timestamp


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