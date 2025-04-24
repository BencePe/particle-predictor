#!/usr/bin/env python3
"""
Main entry point for the PM10 prediction application.
"""

import os
import sys
import logging
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor

from src.prediction.predict_future_air_quality import predict_future_air_quality
from src.db.db_operations import check_db_ready, execute_db_query
from src.utils import setup_logging, create_spark_session, cleanup_resources
from src.fetching.data_fetching import assemble_and_pass, fetch_current_data

# ──────────────────────────────────────────────────────────────────────────────
# bring in all the new functions
from src.model.model_building import (
    build_residual_pipeline,
    apply_residual_correction,
    build_improved_model_pipeline,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    analyze_feature_importance,
    hyperopt_gbt,
    hybrid_time_validation
)
from src.model.data_processing import add_unified_features, validate_data, prepare_training_data

from src.config import MODEL_DIR, FEATURE_COLUMNS, START_DATE, END_DATE, MODEL_PARAMS

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

logger = setup_logging(logging.INFO)


def build_model_for_2024(spark):
    logger.info("Starting model training (2020-2025)...")
    try:
        save = input("Save raw data to db? (y/n): ").strip().lower()

        # 1) Load & preprocess
        df = assemble_and_pass(spark, START_DATE, END_DATE, False, save, 'save', 'historical_2024')
        df = add_unified_features(df).withColumn("is_future", lit(False))

        if not validate_data(df):
            logger.error("Data validation failed.")
            return

        # 2) Time‑based split into train / test
        splits = hybrid_time_validation(df)
        train_df, test_df = splits[0]

        # 3) Build base pipeline (assembler + scaler only)
        base_pipe, feature_cols = build_improved_model_pipeline()
        assembler_stage, scaler_stage = base_pipe.getStages()

        # 4) Further split train_df → train_small / val_small for Hyperopt
        train_small, val_small = train_df.randomSplit([0.9, 0.1], seed=42)


        # 5) Hyperopt search for best GBT params
        best_params, trials = hyperopt_gbt(
            train_small, 
            val_small, 
            assembler_stage, 
            scaler_stage,
            max_evals=10
        )
        logger.info(f"Hyperopt returned: {best_params}")

        # 6) Build final GBT regressor with best params
        gbt = GBTRegressor(
            labelCol="pm10",
            featuresCol="features",
            maxDepth=int(best_params["maxDepth"]),
            maxIter=int(best_params["maxIter"]),
            featureSubsetStrategy=str(best_params["featureSubsetStrategy"]),
            stepSize=best_params["stepSize"],
            subsamplingRate=best_params["subsamplingRate"],
            seed=MODEL_PARAMS.get("seed", 42)
)
        full_pipeline = Pipeline(stages=[assembler_stage, scaler_stage, gbt])

        # 7) Train on full train_df
        model = train_model(full_pipeline, train_df)

        # 8) Feature importance on the tuned model
        analyze_feature_importance(model, feature_cols)

        # 9) Build & train residual‑stacking model
        res_model = build_residual_pipeline(model, train_df, feature_cols)

        # 10) Evaluate both base & residual forecasts on held‑out test_df
        # 10a) Base RMSE
        base_metrics = evaluate_model(model, test_df, prediction_col="prediction")
        logger.info(f"Base GBT metrics on test set: {base_metrics}")

        # 10b) Residual‑corrected RMSE
        corrected = apply_residual_correction(model, res_model, test_df)
        resid_metrics = evaluate_model(corrected, None, prediction_col="final_prediction", is_transformed=True)
        logger.info(f"Residual‑stacked metrics on test set: {resid_metrics}")

        # 11) Save both models
        gbt_path = save_model(model, model_name="pm10_gbt_model")
        resid_path = save_model(res_model, model_name="pm10_res_model")
        logger.info(f"Models saved:\n • GBT: {gbt_path}\n • Residual: {resid_path}")

        return {
            "model": model,
            "residual_model": res_model,
            "base_metrics": base_metrics,
            "resid_metrics": resid_metrics
        }

    except Exception as e:
        logger.error(f"Error building model for 2024: {str(e)}", exc_info=True)

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


def load_latest_and_predict(spark: SparkSession, model_dir=MODEL_DIR):
    logger.info("Loading latest GBT + Residual models for prediction...")
    try:
        model_folders = [
            f for f in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, f)) and "pm10_gbt_model" in f
        ]
        if not model_folders:
            logger.error("No GBT model found.")
            return

        latest_model = sorted(model_folders)[-1]
        model_path   = os.path.join(model_dir, latest_model)
        gbt_model    = load_model(model_path)

        # Try matching residual model folder
        residual_path = model_path.replace("pm10_gbt_model", "pm10_res_model")
        if not os.path.exists(residual_path):
            logger.warning("Residual model not found, using GBT only.")
            return predict_future_air_quality(spark, gbt_model)

        residual_model = load_model(residual_path)
        logger.info(f"GBT model: {model_path}, Residual model: {residual_path}")
        predict_future_air_quality(spark, gbt_model, residual_model=residual_model)

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")


def display_menu():
    print("\n==================== PM10 Prediction Pipeline Menu ====================")
    print("1. Build model")
    print("   - Build the model from scratch using historical Open-Meteo data.")
    print("2. Fetch current sensor data")
    print("   - Start continuous tracking and storage of current sensor readings.")
    print("3. Create database query.")
    print("   - View the raw data in the database.")
    print("4. Load latest model and predict air quality")
    print("5. Exit")
    print("========================================================================")
    return input("Enter your choice number: ").strip()


def main():
    spark = None
    try:
        spark = create_spark_session()
        # if not check_db_ready():
        #     logger.error("Database is not available. Exiting...")
        #     sys.exit(1)

        while True:
            choice = display_menu()
            if choice == "1":
                build_model_for_2024(spark)
                print("\nReturning to main menu...")
                time.sleep(2)
            elif choice == "2":
                print("\nStarting continuous data tracking (press Ctrl+C to stop)...")
                success = fetch_current_sensor_data(spark)
                msg = "Data tracking stopped." if success else "Data tracking failed."
                print(f"\n{msg}\nReturning to main menu...")
                time.sleep(2)
            elif choice == "3":
                interact_with_db()
                print("\nReturning to main menu...")
                time.sleep(2)
            elif choice == "4":
                load_latest_and_predict(spark)
                print("\nReturning to main menu...")
                time.sleep(2)
            elif choice == "5":
                print("\nExiting application. Goodbye!")
                break
            else:
                print("Invalid selection. Please try again.")

    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    finally:
        if spark:
            cleanup_resources(spark)
        sys.exit(0)


if __name__ == "__main__":
    main()
