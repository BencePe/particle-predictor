"""
Utility functions for the PM10 prediction project.
"""

import os
import logging
import sys
from datetime import datetime
from src.config import LOG_DIR, SPARK_CONFIG  # Fixed import

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"pm10_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("pm10_prediction")
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

def create_spark_session(app_name=None, config_dict=None):
    """
    Create and configure a Spark session.
    """
    from pyspark.sql import SparkSession
    
    # Set environment variables
    os.environ["JAVA_HOME"] = "C:\\Zulu\\zulu-21"
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    
    logger = logging.getLogger(__name__)
    logger.info("Initializing Spark session...")
    
    # Use provided config or default from config.py
    config_dict = config_dict or SPARK_CONFIG  # Use imported SPARK_CONFIG
    app_name = app_name or config_dict.get("app_name", "PM10_Prediction")
    
    builder = SparkSession.builder.appName(app_name)
    builder = builder.config("spark.jars", "C:\\Spark\\postgresql-42.7.5.jar")
    
    for key, value in config_dict.items():
        if key != "app_name":
            key_name = f"spark.{key}" if not key.startswith("spark.") else key
            builder = builder.config(key_name, value)
    


    spark = builder \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    logger.info(f"Spark session created: v{spark.version}")
    logger.debug(f"Spark configuration: {spark.sparkContext.getConf().getAll()}")
    return spark

def get_runtime_stats(spark):
    """
    Get runtime statistics of Spark application.
    """
    return {
        "applicationId": spark.sparkContext.applicationId,
        "num_executors": len(spark.sparkContext._jsc.sc().getExecutorMemoryStatus().keySet()),
        "executor_memory": spark.sparkContext.getConf().get("spark.executor.memory"),
        "driver_memory": spark.sparkContext.getConf().get("spark.driver.memory")
    }

def cleanup_resources(spark, temp_files=None):
    """
    Clean up resources when application is done.
    """
    logger = logging.getLogger(__name__)
    
    if temp_files:
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"Removed temporary file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file}: {str(e)}")
    
    if spark:
        logger.info("Stopping Spark session...")
        spark.stop()
        logger.info("Spark session stopped")