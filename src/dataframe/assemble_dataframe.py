
import logging
from dataframe.normalize_timestamp import normalize_timestamp


import pandas as pd
from pyspark.sql.functions import lit

logger = logging.getLogger(__name__)

def assemble_dataframe(spark, df_list, join_key="datetime", how="inner", extra_columns=None, deduplicate=True):
    """
    Assemble a merged Spark DataFrame from a list of Pandas DataFrames.

    Parameters:
        spark: SparkSession object.
        df_list (list): List of Pandas DataFrames to merge.
        join_key (str): Column name to join on (default "datetime").
        how (str): Merge type, e.g., "inner", "left" (default "inner").
        extra_columns (dict): A dictionary of {column_name: value} to add as extra columns.
        deduplicate (bool): If True, drop duplicates on the join key.

    Returns:
        Spark DataFrame: The merged and enriched DataFrame.
    """
    if not df_list or len(df_list) < 1:
        logger.error("No dataframes provided for assembly.")
        return None

    merged_pd_df = df_list[0]
    for df in df_list[1:]:
        merged_pd_df = pd.merge(merged_pd_df, df, on=join_key, how=how)

    if deduplicate:
        merged_pd_df = merged_pd_df.drop_duplicates(subset=join_key)

    spark_df = spark.createDataFrame(merged_pd_df)
    spark_df = normalize_timestamp(spark_df, timestamp_col=join_key)

    if extra_columns is not None:
        for col_name, col_value in extra_columns.items():
            spark_df = spark_df.withColumn(col_name, lit(col_value))
    return spark_df