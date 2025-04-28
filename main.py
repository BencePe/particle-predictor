#!/usr/bin/env python3
"""
Main entry point for the PM10 prediction application.
"""

from datetime import datetime
import os
import sys
import logging
import time
from xml.sax.handler import all_features

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql import functions as F


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
    hybrid_time_validation,
    plot_model_comparison,
    train_with_hybrid_cv
)
from src.model.data_processing import add_unified_features, validate_data

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

        # 2) Time‑based split into train / test
        splits = hybrid_time_validation(df)
        train_df, test_df = splits[0]  # Use first split for main training/testing

        # 3) Build base pipeline
        base_pipe, feature_cols  = build_improved_model_pipeline()
        
        # 4) Run Hybrid CV on untuned base pipeline
        # logger.info("Evaluating base pipeline with hybrid CV...")
        # base_cv_metrics = train_with_hybrid_cv(df, build_improved_model_pipeline)
        # logger.info(f"Baseline Hybrid CV metrics (untuned pipeline): {base_cv_metrics}")

        # 6) Extract assembler and scaler for hyperopt reuse
        assembler_stage, scaler_stage = base_pipe.getStages()

        # 7) Split into small train/val sets for Hyperopt
        train_small, val_small = train_df.randomSplit([0.9, 0.1], seed=42)

        # 8) Hyperopt search for best GBT params
        best_params, trials, best_model, best_metrics = hyperopt_gbt(
            train_small,
            val_small,
            assembler_stage,
            scaler_stage,
            max_evals=60,
        )
        logger.info(f"Hyperopt returned: {best_params}")
        logger.info(f"Hyperopt returned: {best_metrics}")

        # 9) Build final GBT regressor with best params
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

        # 10) Train tuned model
        model = train_model(full_pipeline, train_df)

        # 11) Feature importance
        analyze_feature_importance(model, feature_cols)

        # 13) Evaluate tuned model
        tuned_metrics = evaluate_model(model, test_df)
        logger.info(f"Base GBT metrics on test set: {tuned_metrics}")

        # 14) Generate residuals for residual model training/evaluation
        test_df_with_residuals = model.transform(test_df).withColumn(
            "residual", 
            F.col("pm10") - F.col("prediction")  # Calculate actual residuals
        )
        
        # 12) Build and train residual-stacking model
        res_model = build_residual_pipeline(model, train_df, feature_cols)

        # 16) Evaluate residual model predictions (now using residuals as labels)
        resid_metrics = evaluate_model(
            res_model, 
            test_df_with_residuals,  # Use the DF with residuals
            prediction_col="residual_prediction",
            label_col="residual"  # Critical: evaluate against residuals
        )
        logger.info(f"Residual model metrics: {resid_metrics}")

        # 17) Apply residual correction
        corrected_df = apply_residual_correction(model, res_model, test_df)

        # 18) Plot findings
        plot_model_comparison(
        model, 
        res_model, 
        test_df, 
        corrected_df, 
        tuned_metrics,  # Previously using 'base_metrics' which wasn't defined
        resid_metrics,  # This was correct
        feature_cols=FEATURE_COLUMNS  # Pass the feature columns for importance analysis
    )
        
        # 19) Save both models
        gbt_path = save_model(model, model_name="pm10_gbt_model")
        resid_path = save_model(res_model, model_name="pm10_res_model")
        logger.info(f"Models saved:\n • GBT: {gbt_path}\n • Residual: {resid_path}")

        return {
            "model": model,
            "residual_model": res_model,
            "tuned_metrics": tuned_metrics,
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

def extract_timestamp(name: str) -> datetime:
    try:
        return datetime.strptime(name.split("_")[-1], "%Y%m%d%H%M%S")
    except Exception:
        return datetime.min

def load_latest_and_predict(spark: SparkSession, model_dir=MODEL_DIR):
    logger.info("Loading best GBT + Residual models for prediction...")
    try:
        available_models = os.listdir(model_dir)
        logger.info(f"Available models in {model_dir}: {available_models}")

        # 1. Load latest GBT model
        gbt_models = [f for f in available_models if "pm10_gbt_best_hyperopt" in f]
        if not gbt_models:
            logger.error("No GBT model found.")
            return
        latest_gbt = sorted(gbt_models, key=extract_timestamp)[-1]
        gbt_path = os.path.join(model_dir, latest_gbt)
        gbt_model = load_model(gbt_path)

        # 2. Load latest available residual model
        residual_models = [f for f in available_models if "pm10_res_model" in f]
        residual_model = None
        if residual_models:
            latest_res = sorted(residual_models, key=extract_timestamp)[-1]
            residual_path = os.path.join(model_dir, latest_res)
            if os.path.exists(residual_path):
                try:
                    residual_model = load_model(residual_path)
                    logger.info(f"Residual model loaded from: {residual_path}")
                except Exception as e:
                    logger.warning(f"Failed to load residual model: {e}")
        else:
            logger.warning("No residual model found. Using GBT only.")

        logger.info(f"Using GBT model: {gbt_path}")
        predict_future_air_quality(spark, gbt_model, residual_model=residual_model)

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}", exc_info=True)



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
