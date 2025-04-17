from fetching.data_fetching import get_prediction_input_data
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.utils import AnalysisException
from src.utils import get_logger


def predict_future_air_quality(spark: SparkSession, model: PipelineModel, logger=None):
    """
    """
    if logger is None:
        logger = get_logger(__name__)
    
    Use the provided Spark ML model to forecast PM10 and PM2.5 levels.
    The model is applied to a combined dataset of historical air quality data and
    forecasted weather data for the next 7 days.

    Only rows where `is_future` == True are returned as future predictions.

    Parameters:
        spark (SparkSession): Active Spark session.
        model (PipelineModel): Trained Spark ML model.

    Returns:
        DataFrame: Predictions for future timestamps.
    """
    try:
        # Fetch and prepare the input data
        input_df = get_prediction_input_data(spark)
        if input_df is None:
            logger.error("No input data available for prediction.")
            return None

        # Generate predictions
        full_predictions = model.transform(input_df)
        logger.info("All predictions generated successfully.")

        # Filter for future predictions only
        future_predictions = full_predictions.filter(col("is_future") == True)
        logger.info("Future air quality predictions extracted successfully.")

        return future_predictions

    except AnalysisException as ae:
        logger.error(f"Spark Analysis error during prediction: {ae}")
        return None

    except Exception as ex:
        logger.error(f"An unexpected error occurred during prediction: {ex}")
        return None