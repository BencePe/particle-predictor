from pyspark import StorageLevel
from pyspark.sql import DataFrame
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


def optimize_dataframe(df: DataFrame, partition_cols: Union[str, List[str]] = None, partition_count: int = None, cache: bool = True, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK) -> Union[DataFrame, None]:
    """Optimize a Spark DataFrame for performance by partitioning and/or caching.

    This function can repartition the DataFrame, cache it, or both,
    to improve the efficiency of subsequent operations. It also drops rows
    with null values after optimization.

    Args:
        df (DataFrame): The Spark DataFrame to optimize.
        partition_cols (Union[str, List[str]], optional): Column(s) to partition by.
            If a string is provided, it will be treated as a single column name.
            If a list of strings is provided, it will partition by multiple columns.
            Defaults to None.
        partition_count (int, optional): The number of partitions to use.
            If not provided, Spark's default will be used. Defaults to None.
        cache (bool, optional): Whether to cache the DataFrame. Caching can
            significantly improve the performance of operations that access
            the DataFrame multiple times. Defaults to True.
        storage_level (StorageLevel, optional): The storage level to use for
            caching. Common levels include ``StorageLevel.MEMORY_ONLY``,
            ``StorageLevel.MEMORY_AND_DISK``, and ``StorageLevel.DISK_ONLY``.
            Defaults to StorageLevel.MEMORY_AND_DISK.

    Returns:
        Union[DataFrame, None]: The optimized DataFrame with no null values, or None
            if the input DataFrame is None or an error occurs.

    Raises:
        TypeError: If the input arguments are not of the expected types.
        ValueError: If the input arguments have invalid values.
    """
    try:
        if df is None:
            logger.warning("optimize_dataframe received a None dataframe.")
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
    except (TypeError, ValueError) as e:
        logger.error(f"Error optimizing dataframe: {e}")
        return None