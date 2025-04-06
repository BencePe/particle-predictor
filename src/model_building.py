import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import StorageLevel
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Initialize Spark Session with optimizations
spark = SparkSession.builder \
    .appName("Debrecen_PM10_Prediction") \
    .config("spark.sql.shuffle.partitions", "48") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.nio=ALL-UNNAMED") \
    .getOrCreate()

# Configure Open-Meteo API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ===========================================
# 1. Data Fetching & Conversion
# ===========================================
def fetch_debrecen_data():
    debren_lat = 47.5317
    debren_lon = 21.6244
    
    # Get valid date range
    end_date = datetime.now().date()
    start_date = end_date.replace(year=end_date.year-1)
    
    # Fetch air quality data
    aq_response = openmeteo.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": debren_lat,
            "longitude": debren_lon,
            "hourly": ["pm10", "pm2_5"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    )[0]

    # Fetch weather data
    weather_response = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": debren_lat,
            "longitude": debren_lon,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "surface_pressure", "wind_speed_10m",
                "wind_direction_10m"
            ],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    )[0]

    # Process air quality data
    aq_hourly = aq_response.Hourly()
    aq_data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(aq_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=aq_hourly.Interval()),
            inclusive="left"
        ),
        "pm10": aq_hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": aq_hourly.Variables(1).ValuesAsNumpy()
    }

    # Process weather data
    weather_hourly = weather_response.Hourly()
    weather_data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(weather_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=weather_hourly.Interval()),
            inclusive="left"
        ),
        "temperature": weather_hourly.Variables(0).ValuesAsNumpy(),
        "humidity": weather_hourly.Variables(1).ValuesAsNumpy(),
        "pressure": weather_hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed": weather_hourly.Variables(3).ValuesAsNumpy(),
        "wind_dir": weather_hourly.Variables(4).ValuesAsNumpy()
    }

    # Merge as Pandas DataFrames
    aq_df = pd.DataFrame(aq_data)
    weather_df = pd.DataFrame(weather_data)
    merged_pd_df = pd.merge(aq_df, weather_df, on="datetime", how="inner")

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(merged_pd_df) \
                  .withColumn("elevation", lit(122)) \
                  .withColumn("is_urban", lit(1))

    # Optimize partitioning and persistence
    return spark_df.repartition(48, "datetime") \
                  .persist(StorageLevel.MEMORY_AND_DISK) \
                  .dropna()

# ===========================================
# 2. Optimized Feature Engineering
# ===========================================
def add_urban_features(df):
    # Temporal features
    df = df.withColumn("year", year("datetime")) \
           .withColumn("month", month("datetime")) \
           .withColumn("hour", hour("datetime")) \
           .withColumn("day_of_week", dayofweek("datetime")) \
           .withColumn("is_weekend", when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0))

    # Window specs
    window_lag_24h = Window.partitionBy("year", "month").orderBy("datetime")
    window_24h_avg = Window.partitionBy("year", "month").orderBy("datetime").rowsBetween(-24, 0)
    window_7d_avg = Window.partitionBy("year", "month").orderBy("datetime").rowsBetween(-168, 0)

    # Window-based features
    df = df.withColumn("pm10_lag24", lag("pm10", 24).over(window_lag_24h)) \
           .withColumn("weekly_pm10_avg", avg("pm10").over(window_7d_avg)) \
           .withColumn("pressure_trend", avg("pressure").over(window_24h_avg) - col("pressure"))

    # Wind features
    df = df.withColumn("wind_dir_8", ((col("wind_dir") + 22.5) % 360 / 45).cast("int")) \
           .withColumn("wind_speed_cat", 
                       when(col("wind_speed") < 2, 0)
                       .when(col("wind_speed") < 5, 1)
                       .when(col("wind_speed") < 10, 2)
                       .otherwise(3))

    # Pollution features
    df = df.withColumn("pm_ratio", col("pm2_5")/(col("pm10") + 1e-7)) \
           .withColumn("pollution_load", col("pm10") * col("wind_speed"))

    # Cleanup
    return df.withColumn("pm10_lag24", coalesce(col("pm10_lag24"), lit(0.0))) \
            .withColumn("weekly_pm10_avg", coalesce(col("weekly_pm10_avg"), col("pm10"))) \
            .drop("year", "day_of_week")  # Keep month/hour/is_weekend for modeling

# ===========================================
# 3. Model Pipeline
# ===========================================
def build_model():
    feature_cols = [
        'pm10_lag24', 'weekly_pm10_avg',
        'hour', 'month', 'is_weekend',
        'temperature', 'humidity', 'pressure',
        'wind_speed', 'wind_dir_8', 'wind_speed_cat',
        'pm_ratio', 'pollution_load', 'is_urban'
    ]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )
    
    return (
        Pipeline(stages=[
            assembler,
            GBTRegressor(
                labelCol="pm10",
                featuresCol="features",
                maxIter=150,
                maxDepth=5,
                stepSize=0.1,
                subsamplingRate=0.8,
                seed=42
            )
        ]),
        feature_cols
    )

# ===========================================
# 4. Main Workflow
# ===========================================
def main():
    try:
        # Data pipeline
        df = fetch_debrecen_data()
        df = add_urban_features(df)
        
        # Validate data
        print("Data Schema:")
        df.printSchema()
        print("\nNull Counts:")
        df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
        
        # Split data
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        
        # Build model
        pipeline, feature_cols = build_model()
        model = pipeline.fit(train)
        
        # Evaluate
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(labelCol="pm10")
        
        print(f"\nRMSE: {evaluator.evaluate(predictions):.2f}")
        print(f"R²: {evaluator.evaluate(predictions, {evaluator.metricName: 'r2'}):.2f}")
        
        PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # Adjust based on your structure
        MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
        MODEL_PATH = os.path.join(MODEL_DIR, "pm10_gbt_model")
        
        # Create models directory if not exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save model
        model.write().overwrite().save(MODEL_PATH)
        print(f"\nModel saved to: {MODEL_PATH}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()