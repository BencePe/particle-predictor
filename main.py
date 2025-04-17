#!/usr/bin/env python3
"""
Main entry point for the PM10 prediction application.
"""

import os
import sys
import logging
import time
from datetime import datetime
import pandas as pd
import requests
import pytz
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Set the Python executables for Spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Add project root to path to enable imports from src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Import project modules
from db.db_operations import check_db_ready, execute_db_query
from utils import setup_logging, create_spark_session, cleanup_resources
from fetching.data_fetching import assemble_and_pass, fetch_current_data, get_prediction_input_data
from model.model_building import (
    build_rf_model_pipeline,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    plot_predictions,
)
from model.data_processing import add_urban_features, validate_data, prepare_training_data, add_unified_features
from config import MODEL_DIR, FEATURE_COLUMNS

logger = setup_logging(logging.INFO)

def build_model_for_2024(spark):
    logger.info("Starting model training for 2024...")
    try:
        save = input("Save raw data to db? (y/n): ")
        if save == 'n':
        
            df = assemble_and_pass(spark,'n','save','historical_2024')
            df = add_urban_features(df)
            if not validate_data(df):
                logger.error("Data validation failed.")
                return
            train_df, _ = prepare_training_data(df, test_ratio=0.2)
            pipeline, _ = build_rf_model_pipeline()
            model = train_model(pipeline, train_df)
            metrics = evaluate_model(model, train_df)
            logger.info(f"Evaluation Metrics: {metrics}")
            predictions = model.transform(train_df)
            plot_predictions(predictions)
            model_path = save_model(model, model_name="pm10_rf_model")
            logger.info(f"Model saved locally at: {model_path}")
       
        else:
            df = assemble_and_pass(spark,'y','save','historical_2024')
            df = add_urban_features(df)
            if not validate_data(df):
                logger.error("Data validation failed.")
                return
            train_df, _ = prepare_training_data(df, test_ratio=0.2)
            pipeline, _ = build_rf_model_pipeline()
            model = train_model(pipeline, train_df)
            metrics = evaluate_model(model, train_df)
            logger.info(f"Evaluation Metrics: {metrics}")
            predictions = model.transform(train_df)
            plot_predictions(predictions)
            model_path = save_model(model, model_name="pm10_rf_model")
            logger.info(f"Model saved locally at: {model_path}")
    
    except Exception as e:
        logger.error(f"Error building model for 2024: {str(e)}")


    try:
        input_df = get_prediction_input_data(spark)
        if input_df is None:
            logger.error("No input data available for prediction.")
            return None
        full_predictions = model.transform(input_df)
        full_predictions.show(10)
        future_predictions = full_predictions.filter(col("is_future") == True)
        logger.info("Future air quality predictions generated successfully.")
        future_predictions.show(10)
        return future_predictions
    except Exception as e:
        logger.error(f"Error predicting future air quality: {str(e)}")
        return None

def fetch_current_sensor_data(spark):
    logger.info("Starting current data tracking...")
    try:
        while True:
            logger.info("Fetching latest sensor data...")
            df = fetch_current_data(spark)
            if df is not None:
                logger.info(f"Successfully saved {df.count()} current readings")
            else:
                logger.warning("No new data fetched in this cycle")
            logger.info("Next update in 15 seconds...")
            time.sleep(15)
    except KeyboardInterrupt:
        logger.info("Stopping data tracking...")
        return True
    except Exception as e:
        logger.error(f"Error in data tracking: {str(e)}")
        return False

def interact_with_db():
    query = input("Enter your query: ")
    logger.info(f"Executing query: {query}")
    results = execute_db_query(query, fetch=True)
    if results is not None:
        for row in results:
            print(row)
        print("\nQuery finished.")
    else:
        print("Query failed.")

def load_latest_model_path(model_dir=MODEL_DIR):
    try:
        model_folders = [
            f for f in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, f)) and "pm10_rf_model" in f
        ]
        if not model_folders:
            logger.error("No model found in the model directory.")
            return None
        latest_model = sorted(model_folders)[-1]
        return os.path.join(model_dir, latest_model)
    except Exception as e:
        logger.error(f"Failed to find latest model: {str(e)}")
        return None

    logger.info("Loading latest model for prediction...")
    model_path = load_latest_model_path()
    if not model_path:
        logger.error("Prediction aborted. No model available.")
        return
    try:
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        predict_future_air_quality(spark, model)
    except Exception as e:
        logger.error(f"Failed to load and predict: {str(e)}")

def predict_future_pm10(spark: SparkSession):
    model_path = r"C:\Users\Bence\szakdoga\particle-predictor\models\pm10_rf_model_20250417_201040"
    
    raw_df = get_prediction_input_data(spark)
    if raw_df is None:
        raise RuntimeError("Failed to fetch prediction input data")

    feat_df = add_unified_features(raw_df)

    missing = [c for c in FEATURE_COLUMNS if c not in feat_df.columns]
    if missing:
        raise ValueError(f"After feature-engineering, missing columns: {missing}")

    pm10_model = load_model(model_path)

    predictions = pm10_model.transform(feat_df)

    predictions = predictions.select("datetime", "pm10", "pm2_5", "prediction", "is_future")

    # Plot the predictions
    from model.model_building import plot_predictions
    plot_predictions(predictions)

    return predictions

def display_menu():
    print("\n==================== PM10 Prediction Pipeline Menu ====================")
    print("1. Build model for 2024")
    print("   - Build the model from scratch using historical Open-Meteo data for 2024.")
    print("2. Fetch current sensor data")
    print("   - Start continuous tracking and storage of current sensor readings.")
    print("3. Create database query.")
    print("   - View the raw data in the database.")
    print("4. Load latest model and predict air quality")
    print("5. Exit")
    print("========================================================================")
    choice = input("Enter your choice number: ")
    return choice.strip()

def main():
    spark = None
    try:
        spark = create_spark_session()
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
                interact_with_db()
                print("\nReturning to main menu...")
                time.sleep(2)
            elif choice == "4":
                predict_future_pm10(spark)
                print("\nReturning to main menu...")
                time.sleep(2)
            elif choice == "5":
                print("\nExiting application. Goodbye!")
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
