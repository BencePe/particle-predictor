from pyspark.sql.functions import col, to_timestamp


def normalize_timestamp(spark_df, timestamp_col="datetime"):
    if spark_df is None:
        return None
    normalized_df = spark_df.withColumn(
        timestamp_col,
        to_timestamp(col(timestamp_col).cast("string"), "yyyy-MM-dd HH:mm:ss")
    )
    return normalized_df