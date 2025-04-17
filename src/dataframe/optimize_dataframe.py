from pyspark import StorageLevel


def optimize_dataframe(df, partition_cols=None, partition_count=None, cache=True, storage_level=StorageLevel.MEMORY_AND_DISK):
    """
    Optimize a Spark DataFrame for performance.

    Parameters:
        df: Spark DataFrame to optimize.
        partition_cols (list or str): Column(s) to partition by.
        partition_count (int): Number of partitions.
        cache (bool): Whether to cache the DataFrame.
        storage_level: PySpark StorageLevel to use if caching.

    Returns:
        Spark DataFrame: Optimized DataFrame.
    """
    if df is None:
        return None

    result_df = df
    if partition_cols and partition_count:
        result_df = result_df.repartition(partition_count, partition_cols)
    elif partition_cols:
        result_df = result_df.repartition(partition_cols)
    elif partition_count:
        result_df = result_df.repartition(partition_count)
    if cache:
        result_df = result_df.persist(storage_level)
    return result_df.dropna()