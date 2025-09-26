from src.config import DATA_DIR
from src.fetching.data_fetching import logger
from datetime import datetime
import os



def save_dataframe(df, name_prefix, mode="overwrite"):
    if df is None:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(DATA_DIR, f"{name_prefix}_{timestamp}.parquet")
    df.write.parquet(save_path, mode=mode)
    logger.info(f"Data saved to {save_path}")
    return save_path