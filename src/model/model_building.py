"""
Model building and evaluation functions.
"""

import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from src.config import MODEL_DIR, MODEL_PARAMS, FEATURE_COLUMNS

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

def plot_predictions(predictions_df):
    """
    Plot both historical data and predictions.
    
    Parameters:
        predictions_df: DataFrame with both historical data and predictions
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    try:
        # Convert to pandas for easier plotting
        pdf = predictions_df.toPandas()
        
        # Convert datetime to proper format if needed
        if not isinstance(pdf['datetime'].iloc[0], datetime):
            pdf['datetime'] = pd.to_datetime(pdf['datetime'])
        
        # Sort by datetime
        pdf = pdf.sort_values('datetime')
        
        # Separate historical and future data
        historical = pdf[pdf['is_future'] == False]
        future = pdf[pdf['is_future'] == True]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical actual values
        ax.plot(historical['datetime'], historical['pm10'], 
                color='blue', label='Historical PM10', linewidth=2)
        
        # Plot historical predictions (for validation)
        if 'prediction' in historical.columns:
            ax.plot(historical['datetime'], historical['prediction'], 
                    color='green', label='Model Fit', linewidth=1, linestyle='--')
        
        # Plot future predictions
        if 'prediction' in future.columns:
            ax.plot(future['datetime'], future['prediction'], 
                    color='red', label='PM10 Forecast', linewidth=2)
        
        # Add PM2.5 if available
        if 'pm2_5' in historical.columns:
            ax2 = ax.twinx()
            ax2.plot(historical['datetime'], historical['pm2_5'], 
                    color='purple', label='Historical PM2.5', linewidth=1, alpha=0.7)
            
            if 'pm2_5_prediction' in future.columns:
                ax2.plot(future['datetime'], future['pm2_5_prediction'], 
                        color='magenta', label='PM2.5 Forecast', linewidth=1, alpha=0.7)
            
            ax2.set_ylabel('PM2.5 (μg/m³)', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
        
        # Add vertical line separating historical data from predictions
        if not future.empty and not historical.empty:
            separation_date = future['datetime'].min()
            ax.axvline(x=separation_date, color='black', linestyle='-', linewidth=1)
            ax.text(separation_date, ax.get_ylim()[1]*0.95, 'Forecast Start', 
                    ha='center', va='top', backgroundcolor='white')
        
        # Customize the plot
        ax.set_xlabel('Date')
        ax.set_ylabel('PM10 (μg/m³)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add grid, title and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.title('PM10 and PM2.5 Air Quality Forecast', fontsize=14)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        if 'pm2_5' in historical.columns:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_quality_forecast_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast plot saved as {filename}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting predictions: {str(e)}")

def load_model(model_path: str) -> PipelineModel:
    """
    Load a previously saved Spark ML PipelineModel from disk.
    If `model_path` itself doesn’t contain metadata/, look for a
    subdirectory that does.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"No model found at: {model_path}")

    # If this directory already has metadata/, load it directly
    if os.path.isdir(os.path.join(model_path, "metadata")):
        real_model_path = model_path
    else:
        # otherwise scan for a subdirectory that has metadata/
        candidates = [
            os.path.join(model_path, d)
            for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
        ]
        real_model_path = None
        for cand in candidates:
            if os.path.isdir(os.path.join(cand, "metadata")):
                real_model_path = cand
                break

        if real_model_path is None:
            logger.error(f"No Spark ML model found under: {model_path}")
            raise FileNotFoundError(f"No Spark ML model found in: {model_path}")

    logger.info(f"Loading Spark ML PipelineModel from: {real_model_path}")
    model = PipelineModel.load(real_model_path)
    logger.info("Model loaded successfully")
    return model