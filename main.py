"""
Main entry point for the PM10 prediction application.

Menu Options:
    1. Build model for 2024
         - Builds the model from scratch using historical Open-Meteo data for 2024.
         - The model files are stored locally.
    2. Fetch current sensor data
         - Starts continuous tracking of current sensor readings and uploads to database
    3. Compute hourly means
         - Process current sensor data and save hourly averages to air_quality_2025.
    4. Exit
         - Quit the application.

After completing a task, the menu is displayed again.
"""

import os
import sys
import logging
from datetime import datetime
from pyspark.sql import SparkSession
import time

# Set the Python executables for Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Add project root to path to enable imports from src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import project modules
from db_operations import check_db_ready
from utils import setup_logging, create_spark_session, cleanup_resources
from data_fetching import fetch_debrecen_data, fetch_current_data
from model_building import build_rf_model_pipeline, train_model, evaluate_model, save_model, plot_predictions
from data_processing import add_urban_features, validate_data, prepare_training_data

logger = setup_logging(logging.INFO)

def build_model_for_2024(spark):
    """
    Build the model for 2024 using historical Open-Meteo data.
    
    Steps:
      - Fetch data for Debrecen.
      - Process the data.
      - Train a Random Forest model.
      - Evaluate and save the model.
    """
    logger.info("Starting model training for 2024...")
    
    try:
        # Fetch historical data and add urban features
        df = fetch_debrecen_data(spark)
        df = add_urban_features(df)
        
        # Optionally validate and prepare the data
        if not validate_data(df):
            logger.error("Data validation failed.")
            return
        
        # For simplicity, we use the whole dataset as training data.
        train_df, _ = prepare_training_data(df, test_ratio=0.2)
        
        # Build the Random Forest pipeline and train the model
        pipeline, features = build_rf_model_pipeline()
        model = train_model(pipeline, train_df)
        
        # Evaluate the model (using training data as a placeholder)
        metrics = evaluate_model(model, train_df)
        logger.info(f"Evaluation Metrics: {metrics}")
        
        # Generate prediction plot and save model locally
        predictions = model.transform(train_df)
        plot_predictions(predictions)
        model_path = save_model(model, model_name="pm10_rf_model")
        logger.info(f"Model saved locally at: {model_path}")
    except Exception as e:
        logger.error(f"Error building model for 2024: {str(e)}")

def fetch_current_sensor_data(spark):
    """
    Starts continuous tracking of current sensor data and uploads to database
    """
    logger.info("Starting current data tracking...")
    try:
        while True:
            logger.info("Fetching latest sensor data...")
            df = fetch_current_data(spark)
            if df is not None:
                logger.info(f"Successfully saved {df.count()} current readings")
            else:
                logger.warning("No new data fetched in this cycle")
            
            # Wait before next fetch (15 seconds)
            logger.info("Next update in 15 seconds...")
            time.sleep(15)
            
    except KeyboardInterrupt:
        logger.info("Stopping data tracking...")
        return True
    except Exception as e:
        logger.error(f"Error in data tracking: {str(e)}")
        return False

def display_menu():
    print("\n==================== PM10 Prediction Pipeline Menu ====================")
    print("1. Build model for 2024")
    print("   - Build the model from scratch using historical Open-Meteo data for 2024.")
    print("2. Fetch current sensor data")
    print("   - Start continuous tracking and storage of current sensor readings")
    print("3. Exit")
    print("========================================================================")
    choice = input("Enter your choice number: ")
    return choice.strip()

def main():
    """
    Main function: creates Spark session, displays menu, and executes user's choices.
    """
    spark = None
    try:
        spark = create_spark_session()
        
        # Check database connectivity
        if not check_db_ready():
            logger.error("Database is not available. Exiting...")
            sys.exit(1)
            
        while True:
            choice = display_menu()
            if choice == "1":
                build_model_for_2024(spark)
                print("\nReturning to main menu...")
                time.sleep(2)
            elif choice == "2":
                print("\nStarting continuous data tracking (press Ctrl+C to stop)...")
                success = fetch_current_sensor_data(spark)
                message = "Data tracking stopped." if success else "Data tracking failed."
                print(f"\n{message}\nReturning to main menu...")
                time.sleep(2)
            elif choice == "3":
                print("Exiting application. Goodbye!")
                break
            else:
                print("Invalid selection. Please try again.")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
    finally:
        if spark:
            cleanup_resources(spark)
        sys.exit(0)

if __name__ == "__main__":
    main()