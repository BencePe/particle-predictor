"""Module to assemble multiple dataframes"""
import logging
from src.dataframe.normalize_timestamp import normalize_timestamp


from typing import List
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql import DataFrame as SparkDataFrame

logger = logging.getLogger(__name__)

def assemble_dataframe(spark, df_list, join_key:str="datetime", how:str="inner", extra_columns:dict=None, deduplicate:bool=True) -> SparkDataFrame:
    """Assemble a merged Spark DataFrame from a list of Pandas DataFrames.

    This function merges a list of Pandas DataFrames into a single Spark DataFrame,
    normalizes the timestamp column, and optionally adds extra columns and
    removes duplicate rows.

    Parameters:
        spark: The SparkSession object used to create the Spark DataFrame.
        df_list: A list of Pandas DataFrames to be merged.
        join_key: The name of the column to use as the join key.
          Defaults to "datetime".
        how: The type of join to perform. Can be any valid Pandas merge type
          (e.g., "inner", "left", "right", "outer"). Defaults to "inner".
        extra_columns: A dictionary where keys are new column names and values
          are the values to be assigned to these columns in each row of the
          resulting DataFrame. Defaults to None.
        deduplicate: Whether to remove duplicate rows based on the join key
          after the merge operation. Defaults to True.

    Returns:
        A Spark DataFrame that is the result of merging all input Pandas
        DataFrames. Returns None if the input list is empty or if any error
        occurs during the process.

    Raises:
        TypeError: If `df_list` does not contain only Pandas DataFrames.
        ValueError: If the `join_key` is not found in the dataframes.
        Exception: If an unexpected error occurs during assembly.
    """
    try:
        if not df_list or len(df_list) < 1:
            logger.error("No dataframes provided for assembly.")
            return None
        if not all(isinstance(df, pd.DataFrame) for df in df_list):
            raise TypeError("All elements in df_list must be Pandas DataFrames.")
        
        merged_pd_df = df_list[0]
        for df in df_list[1:]:
            if not join_key in df.columns or not join_key in merged_pd_df.columns:
                raise ValueError(f"Join key '{join_key}' not found in one or more dataframes.")
            merged_pd_df = pd.merge(merged_pd_df, df, on=join_key, how=how)

        if deduplicate:
            merged_pd_df = merged_pd_df.drop_duplicates(subset=join_key)

        spark_df = spark.createDataFrame(merged_pd_df)
        spark_df = normalize_timestamp(spark_df, timestamp_col=join_key)

        if extra_columns is not None:
            for col_name, col_value in extra_columns.items():
                spark_df = spark_df.withColumn(col_name, lit(col_value))
        return spark_df
    except (TypeError, ValueError) as e:
        logger.error(f"Error during dataframe assembly: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during dataframe assembly: {e}")
        return None