"""
Database operations for the PM10 prediction project.
"""

import psycopg2
from psycopg2 import sql
import logging
import time
from src.config import DB_CONFIG

logger = logging.getLogger(__name__)

def check_db_ready(max_attempts=5, delay=5):
    
    """
    Check if the database is ready and running.
    
    Parameters:
        max_attempts (int): Maximum number of connection attempts
        delay (int): Delay in seconds between attempts
        
    Returns:
        bool: True if database is available, False otherwise.
    """
    attempt = 1
    while attempt <= max_attempts:
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt}/{max_attempts})")
            conn = psycopg2.connect(
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"],
                database=DB_CONFIG["database"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"]
            )
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            conn.close()
            if result and result[0] == 1:
                logger.info("Database connection successful")
                return True
        except Exception as e:
            logger.warning(f"Database connection failed: {str(e)}")
            if attempt < max_attempts:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        attempt += 1
    logger.error("Failed to connect to the database after multiple attempts")
    return False

def db_data_transaction(spark, operation, table_name, data=None, query=None):
    """
    Perform a database transaction, either saving or loading data.
    
    Assumes that all tables share the same format (except for a future special case).
    
    Parameters:
        spark: SparkSession object.
        operation (str): "save" or "load".
        table_name (str): The target table in the database.
        data: (Spark DataFrame) Data to save when operation is "save".
        query: (str) SQL query to execute when operation is "load".
        
    Returns:
        For "save": True if successful, False otherwise.
        For "load": Spark DataFrame containing the query results.
    """
    from py4j.java_gateway import java_import
    java_import(spark._jvm, "org.postgresql.Driver")
    logger.info("PostgreSQL driver registered successfully")
    
    jdbc_url = f"jdbc:postgresql://{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    
    try:
        if operation.lower() == "save":
            if data is None:
                logger.error("No data provided for saving.")
                return False
                
            # Validate that all required columns are present
            if table_name in ["historical_2024"]:
                required_columns = ["datetime", "pm10", "temperature", "humidity", 
                                   "pressure", "wind_speed", "wind_dir", "elevation", "is_urban", "is_future"]
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns for {table_name}: {missing_columns}")
                    return False
            
            # Write to the database using Spark's JDBC writer in append mode.
            data.write \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("driver", "org.postgresql.Driver") \
                .option("dbtable", table_name) \
                .option("user", DB_CONFIG["user"]) \
                .option("password", DB_CONFIG["password"]) \
                .mode("append") \
                .save()
            logger.info(f"Successfully saved data to table '{table_name}'")
            return True

        elif operation.lower() == "load":
            if not query and table_name:
                query = f"SELECT * FROM {table_name}"
            if not query:
                logger.error("No query or table name provided for load operation.")
                return None

            df = spark.read \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("query", query) \
                .option("user", DB_CONFIG["user"]) \
                .option("password", DB_CONFIG["password"]) \
                .load()
            logger.info(f"Successfully loaded data from table '{table_name}'")
            return df

        else:
            logger.error("Invalid operation. Use 'save' or 'load'.")
            return None

    except Exception as e:
        logger.error(f"Error during {operation} operation on table '{table_name}': {str(e)}")
        if operation.lower() == "save":
            return False
        else:
            return None

def execute_db_query(query, params=None, fetch=False):
    """
    Execute a raw SQL query on the database.
    
    Parameters:
        query (str): SQL query to execute
        params (tuple): Parameters for the query
        fetch (bool): Whether to fetch and return results
        
    Returns:
        list or None: Query results if fetch=True, None otherwise
    """
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            
            if fetch:
                results = cursor.fetchall()
                conn.close()
                return results
            else:
                conn.commit()
                conn.close()
                return True
                
    except Exception as e:
        logger.error(f"Error executing database query: {str(e)}")
        return None