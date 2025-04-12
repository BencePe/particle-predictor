"""
Model building and evaluation functions.
"""

import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from config import MODEL_DIR, MODEL_PARAMS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

def build_model_pipeline():
    """
    Build a machine learning pipeline for PM10 prediction using Gradient Boosted Trees.
    
    Returns:
        tuple: (pipeline, feature_columns)
    """
    logger.info("Building Gradient Boosted Tree model pipeline...")
    
    # Create feature assembler
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="keep"
    )
    
    # Create gradient boosted tree regressor using parameters from MODEL_PARAMS
    gbt = GBTRegressor(
        labelCol="pm10",
        featuresCol="features",
        maxIter=MODEL_PARAMS["max_iter"],
        maxDepth=MODEL_PARAMS["max_depth"],
        stepSize=MODEL_PARAMS["step_size"],
        subsamplingRate=MODEL_PARAMS["subsampling_rate"],
        seed=MODEL_PARAMS["seed"]
    )
    
    # Create and return the pipeline
    pipeline = Pipeline(stages=[assembler, gbt])
    
    logger.info("GBT model pipeline created successfully")
    return pipeline, FEATURE_COLUMNS

def build_rf_model_pipeline():
    """
    Build a machine learning pipeline for PM10 prediction using Random Forest.
    
    Returns:
        tuple: (pipeline, feature_columns)
    """
    logger.info("Building Random Forest model pipeline for PM10 prediction...")
    
    # Create feature assembler
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="keep"
    )
    
    # Create random forest regressor using parameters from MODEL_PARAMS
    rf = RandomForestRegressor(
        labelCol="pm10",
        featuresCol="features",
        numTrees=MODEL_PARAMS.get("num_trees", 100),         # default to 100 trees if not provided
        maxDepth=MODEL_PARAMS.get("max_depth_rf", 5),          # you can set a separate max depth for RF
        seed=MODEL_PARAMS.get("seed")
    )
    
    # Create and return the pipeline
    pipeline = Pipeline(stages=[assembler, rf])
    
    logger.info("Random Forest model pipeline created successfully")
    return pipeline, FEATURE_COLUMNS

def train_model(pipeline, train_df):
    """
    Train the ML model on the training data.
    
    Parameters:
        pipeline: ML Pipeline object
        train_df: Training DataFrame
        
    Returns:
        PipelineModel: Trained model
    """
    logger.info("Training PM10 prediction model...")
    # Use datetime.now() for both timestamps to allow subtraction:
    start_time = datetime.now()
    
    # Train the model
    model = pipeline.fit(train_df)
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, test_df):
    """
    Evaluate the model performance on test data.
    
    Parameters:
        model: Trained PipelineModel
        test_df: Test DataFrame
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Set up evaluator for multiple metrics
    evaluator = RegressionEvaluator(labelCol="pm10")
    
    # Calculate metrics
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    
    # Log metrics
    logger.info("Model Evaluation Metrics:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    
    # Return metrics as a dictionary
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae
    }

def save_model(model, model_name="pm10_gbt_model"):
    """
    Save the trained model to disk.
    
    Parameters:
        model: Trained PipelineModel
        model_name (str): Base name for the model
        
    Returns:
        str: Path to saved model
    """
    # Add timestamp to model name for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}")
    
    # Save the model
    model.write().overwrite().save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    return model_path

def plot_predictions(predictions, sample_size=100, save_path=None):
    """
    Plot actual vs predicted PM10 values.
    
    Parameters:
        predictions: DataFrame with predictions
        sample_size (int): Number of points to plot
        save_path (str): Path to save the plot image
        
    Returns:
        str: Path to saved plot or None
    """
    try:
        # Convert to Pandas for plotting. The sample fraction is the minimum of 1.0 and sample_size/total_count.
        total_count = predictions.count()
        fraction = min(1.0, sample_size / total_count) if total_count > 0 else 1.0
        pdf = predictions.select("pm10", "prediction").sample(False, fraction, seed=42).toPandas()
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(pdf["pm10"], pdf["prediction"], alpha=0.5)
        plt.plot([0, pdf["pm10"].max()], [0, pdf["pm10"].max()], 'r--')
        plt.xlabel("Actual PM10")
        plt.ylabel("Predicted PM10")
        plt.title("Actual vs Predicted PM10 Values")
        
        # Save plot if path provided, otherwise save with default naming convention in MODEL_DIR
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Prediction plot saved to: {save_path}")
            return save_path
        else:
            plot_path = os.path.join(MODEL_DIR, f"prediction_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path)
            logger.info(f"Prediction plot saved to: {plot_path}")
            return plot_path
            
    except Exception as e:
        logger.error(f"Error creating prediction plot: {str(e)}")
        return None
