from src.fetching.data_fetching import get_prediction_input_data, logger
from src.model.model_building import plot_predictions

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def predict_future_air_quality(spark: SparkSession, model: PipelineModel):
    """
    Use the provided Spark ML model to forecast PM10 and PM2.5 levels.
    The model is applied to a combined dataset of historical air quality data and
    forecasted weather data for the next 7 days.

    Only rows where 'is_future' == True are returned as future predictions.

    Parameters:
        spark (SparkSession): Active Spark session.
        model (PipelineModel): Trained Spark ML model.

    Returns:
        DataFrame: Predictions for future timestamps.
    """
    try:
        # Fetch and prepare the input data
        input_df = get_prediction_input_data(spark)
        if input_df is None or input_df.rdd.isEmpty():
            logger.error("No input data available for prediction.")
            return None

        # Generate predictions
        full_predictions = model.transform(input_df)
        logger.info("All predictions generated successfully.")
        plot_predictions(full_predictions)

        # Filter for future predictions only
        future_predictions = full_predictions.filter(col("is_future") == True)
        logger.info("Future air quality predictions extracted successfully.")
        plot_predictions(future_predictions)
        return future_predictions

    except Exception as e:
        logger.error(f"Error predicting future air quality: {str(e)}")
        return None