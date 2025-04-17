import os
from datetime import datetime
from pyspark.sql.utils import AnalysisException
from src.config.config_manager import get_config
from src.utils import logger as global_logger
config = get_config()

def save_dataframe(df, name_prefix, mode="overwrite", data_dir=settings.DATA_DIR, logger=global_logger):
    """Save a DataFrame as a Parquet file with a timestamp in the filename.

    Args:
        df (DataFrame): The Spark DataFrame to save.
        name_prefix (str): A prefix for the filename.
        mode (str, optional): The write mode for the Parquet file.
            Defaults to "overwrite".
            Other options include "append", "ignore", "error", etc.
        data_dir (str, optional): The directory where the data will be saved.
            Defaults to the value of DATA_DIR in settings.
        logger: (Logger, optional): logger to use.
            Defaults to global_logger.

    Returns:
        str or None: The path where the data was saved if successful, None otherwise.

    Raises:
        AnalysisException: If there is an error writing the DataFrame to Parquet.

    Example:
        >>> path = save_dataframe(my_df, "my_data", mode="append", data_dir="/tmp/data")
        >>> if path:
        ...     print(f"Data successfully saved to {path}")
    """
    if df is None:
        return None
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(data_dir, f"{name_prefix}_{timestamp}.parquet")
        df.write.parquet(save_path, mode=mode)
        logger.info(f"Data saved to {save_path}")
        return save_path
    except AnalysisException as e:
        logger.error(f"Error writing dataframe to Parquet: {e}")
        return None
