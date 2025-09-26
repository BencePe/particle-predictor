from pyspark import StorageLevel


def optimize_dataframe(df, partition_cols=None, partition_count=None, cache=True, storage_level=StorageLevel.MEMORY_AND_DISK):
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