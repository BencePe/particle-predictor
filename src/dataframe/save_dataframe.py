from src.config import DATA_DIR
from src.fetching.data_fetching import logger
from datetime import datetime
import os



def save_dataframe(df, name_prefix, mode="overwrite"):
    """
    Save a DataFrame as a Parquet file with timestamp.

    Parameters:
        df: Spark DataFrame to save.
        name_prefix (str): Prefix for the filename.
        mode (str): Write mode ("overwrite", "append", etc.)

    Returns:
        str: Path where the data was saved.
    """
    if df is None:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(DATA_DIR, f"{name_prefix}_{timestamp}.parquet")
    df.write.parquet(save_path, mode=mode)
    logger.info(f"Data saved to {save_path}")
    return save_path