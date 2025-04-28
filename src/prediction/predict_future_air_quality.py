import pandas as pd
import numpy as np

from src.fetching.data_fetching import get_prediction_input_data, logger
from src.model.model_building import plot_predictions


from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, min as spark_min, max as spark_max, sum as spark_sum
from pyspark.sql.types import DoubleType

from src.config import FEATURE_COLUMNS, DEBRECEN_ELEVATION

def predict_future_air_quality(spark: SparkSession, model: PipelineModel, residual_model: PipelineModel = None): 
    try:
        spark.conf.set("spark.sql.debug.maxToStringFields", 1000)
        hist_df, future_w = get_prediction_input_data(spark)

        if hist_df is None or hist_df.rdd.isEmpty():
            logger.error("No historical data available for prediction.")
            return None

        # 1) ensure we only forecast *after* our last observed time
        last_dt = hist_df.agg(spark_max("datetime")).first()[0]
        logger.info("Last historical datetime: %s", last_dt)
        
        # Log the original future weather data range
        future_min = future_w.agg(spark_min("datetime")).first()[0]
        future_max = future_w.agg(spark_max("datetime")).first()[0]
        logger.info("Original future weather range: %s to %s", 
                   future_min.strftime('%Y-%m-%d %H:%M'), 
                   future_max.strftime('%Y-%m-%d %H:%M'))
        
        # Filter but ensure we include at least 14 days worth of data
        future_w = future_w.filter(col("datetime") >= last_dt)
        
        # Log filtered range    
        if not future_w.rdd.isEmpty():
            filtered_min = future_w.agg(spark_min("datetime")).first()[0]
            filtered_max = future_w.agg(spark_max("datetime")).first()[0]
            logger.info("Filtered future weather range: %s to %s", 
                       filtered_min.strftime('%Y-%m-%d %H:%M'), 
                       filtered_max.strftime('%Y-%m-%d %H:%M'))
            
            # Check if we have enough data
            days_available = (filtered_max - filtered_min).days + 1
            logger.info("Days available for forecast: %d", days_available)
            
            if days_available < 14:
                logger.warning("Not enough future weather data for a 14-day forecast. Only %d days available.", days_available)
        else:
            logger.error("No future weather data available after filtering.")
            return None

        # 2) generate recursive forecasts
        preds_df = recursive_pm10_forecast(
    spark, model, hist_df, future_w, periods=7*24, residual_model=residual_model, base_time=last_dt)

        # 3) assemble history-only and forecast-only DataFrames
        # history-only: no prediction column
        hist_only = (
            hist_df
            .select(
                col("datetime"),
                col("pm10"),
                lit(None).cast(DoubleType()).alias("prediction"),
                lit(False).alias("is_future")
            )
        )

        # forecast-only: rename pm10 to prediction
        pred_only = (
            preds_df
            .select(
                col("datetime"),
                lit(None).cast(DoubleType()).alias("pm10"),
                col("pm10").alias("prediction"),
                col("is_future")
            )
        )

        # union and order
        result = hist_only.unionByName(pred_only).orderBy("datetime")

        # 4) logging ranges
        hist_range_min = hist_df.agg(spark_min("datetime")).first()[0]
        hist_range_max = hist_df.agg(spark_max("datetime")).first()[0]
        pred_range_min = preds_df.agg(spark_min("datetime")).first()[0]
        pred_range_max = preds_df.agg(spark_max("datetime")).first()[0]

        logger.info(
            "Historical Data Range: %s to %s",
            hist_range_min.strftime('%Y-%m-%d'),
            hist_range_max.strftime('%Y-%m-%d')
        )
        logger.info(
            "Forecast Data Range: %s to %s",
            pred_range_min.strftime('%Y-%m-%d'),
            pred_range_max.strftime('%Y-%m-%d')
        )

        # 5) plot and return
        plot_predictions(result)
        return result

    except Exception as e:
        logger.error(f"Error predicting future air quality: {str(e)}", exc_info=True)
        return None

def recursive_pm10_forecast(
    spark,
    model,
    hist_df,
    future_weather_df,
    periods=7 * 24,
    residual_model=None,
    base_time=None
):
    """
    Predict PM10 recursively using historical and future weather data,
    properly filling missing history before switching to true forecasting.
    """

    hist_pd = hist_df.toPandas().sort_values("datetime").reset_index(drop=True)
    future_pd = future_weather_df.toPandas().sort_values("datetime").reset_index(drop=True)

    if "precipitation" not in hist_pd.columns:
        hist_pd["precipitation"] = 0.0

    # Determine split point: last datetime in historical
    last_hist_time = hist_pd["datetime"].max()

    # Separate future overlap (patch) vs true future forecast
    future_patch = future_pd[future_pd["datetime"] <= last_hist_time]
    future_forecast = future_pd[future_pd["datetime"] >= last_hist_time]

    # Build buffers
    pm10_buffer = hist_pd["pm10"].iloc[-96:].tolist()
    pressure_buffer = hist_pd["pressure"].iloc[-96:].tolist()
    wind_buffer = hist_pd["wind_speed"].iloc[-96:].tolist()
    temp_buffer = hist_pd["temperature"].iloc[-96:].tolist()

    # Historical bounds
    hist_min = max(0, hist_pd["pm10"].min() * 0.5)
    hist_max = hist_pd["pm10"].max() * 1.5

    forecasts = []

    def buffer_lag(buffer, lag):
        return buffer[-lag] if len(buffer) >= lag else np.nan

    def buffer_roll(buffer, window, func):
        window_values = buffer[-window:]
        return func(window_values) if window_values else np.nan

    def predict_single_point(feat_dict, dt):
        """
        Predict a single timestep with optional residual correction
        """
        single_row_df = pd.DataFrame([feat_dict])
        feature_cols = [c for c in single_row_df.columns if c in FEATURE_COLUMNS]
        single_sdf = spark.createDataFrame(single_row_df)
        single_feats = single_sdf.select("datetime", *feature_cols)

        pred_row = model.transform(single_feats).first()
        pred = float(pred_row["prediction"])

        # if residual_model:
        #     enriched = single_feats.withColumn("base_prediction", lit(pred))
        #     if base_time:
        #         horizon = (dt - base_time).total_seconds() / 3600.0
        #         enriched = enriched.withColumn("forecast_horizon", lit(horizon))
        #     res_pred = residual_model.transform(enriched).first()["residual_prediction"]
        #     pred += float(res_pred)

        pred = max(hist_min, min(pred, hist_max))  # Stabilization
        return pred

    def build_features(dt, temp, hum, pres, wind, precip, pm10_target):
        dry_spell_hours = 0
        for hour_offset in range(1, min(73, len(hist_pd) + 1)):
            if hist_pd["precipitation"].iloc[-hour_offset] == 0:
                dry_spell_hours += 1
            else:
                break
        dry_spell_days = dry_spell_hours // 24

        prev_pressure = buffer_lag(pressure_buffer, 12)
        last_pm10 = pm10_buffer[-1] if len(pm10_buffer) > 0 else 0.0

        feat = {
            "datetime": dt,
            "pm10": pm10_target,
            "temperature": temp,
            "humidity": hum,
            "pressure": pres,
            "wind_speed": wind,
            "precipitation": precip,
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24.0),
            "month_sin": np.sin(2 * np.pi * dt.month / 12.0),
            "winter_indicator": int(dt.month == 12 or dt.month <= 2),
            "pm10_lag3": buffer_lag(pm10_buffer, 3),
            "pm10_lag24": buffer_lag(pm10_buffer, 24),
            "pm10_lag168": buffer_lag(pm10_buffer, 168),
            "3h_pm10_avg": buffer_roll(pm10_buffer, 3, np.mean),
            "3h_pm10_std": buffer_roll(pm10_buffer, 3, np.std),
            "24h_pm10_std": buffer_roll(pm10_buffer, 24, np.std),
            "rolling_max_pm10_24h": buffer_roll(pm10_buffer, 24, np.max),
            "rolling_min_pm10_24h": buffer_roll(pm10_buffer, 24, np.min),
            "3h_wind_speed_avg": buffer_roll(wind_buffer, 3, np.mean),
            "24h_wind_speed_avg": buffer_roll(wind_buffer, 24, np.mean),
            "pm10_diff_3h": last_pm10 - buffer_lag(pm10_buffer, 3) if not np.isnan(buffer_lag(pm10_buffer, 3)) else 0.0,
            "pm10_diff_24h": last_pm10 - buffer_lag(pm10_buffer, 24) if not np.isnan(buffer_lag(pm10_buffer, 24)) else 0.0,
            "pm10_volatility_3h": buffer_roll(pm10_buffer, 3, np.std) / (buffer_roll(pm10_buffer, 3, np.mean) + 1e-5),
            "is_precipitation": int(precip > 0.1),
            "precipitation_intensity": 0 if precip < 0.5 else (1 if precip < 5 else 2),
            "dew_point": temp - (100 - hum) / 5,
            "stagnation_index": int(wind < 2.5 and abs(pres - prev_pressure) < 1.0 and hum > 70),
            "heating_effect": int((dt.month == 12 or dt.month <= 2) and temp < 2 and hum > 80),
            "dry_spell_days": int(dry_spell_days),
        }
        return feat
    # --- Phase 1: Fill missing (patched) history ---
    for i in range(len(future_patch)):
        row = future_patch.iloc[i]
        dt = row["datetime"]
        feat = build_features(dt, row["temperature"], row["humidity"], row["pressure"], row["wind_speed"], row["precipitation"], np.nan)
        pred = predict_single_point(feat, dt)
        forecasts.append({"datetime": dt, "pm10": pred, "is_future": False})

        # Update buffers
        pm10_buffer.append(pred)
        pressure_buffer.append(row["pressure"])
        wind_buffer.append(row["wind_speed"])
        pm10_buffer = pm10_buffer[-96:]
        pressure_buffer = pressure_buffer[-96:]
        wind_buffer = wind_buffer[-96:]
        temp_buffer = temp_buffer[-96:]

        # Update hist_pd for dry_spell/rain checks
        hist_pd = pd.concat([hist_pd, pd.DataFrame({"datetime": [dt], "precipitation": [row["precipitation"]]})], ignore_index=True).iloc[-168:]

    # --- Phase 2: True recursive future forecast ---
    for i in range(min(periods, len(future_forecast))):
        row = future_forecast.iloc[i]
        dt = row["datetime"]
        feat = build_features(dt, row["temperature"], row["humidity"], row["pressure"], row["wind_speed"], row["precipitation"], np.nan)
        pred = predict_single_point(feat, dt)
        forecasts.append({"datetime": dt, "pm10": pred, "is_future": True})

        # Update buffers
        pm10_buffer.append(pred)
        pressure_buffer.append(row["pressure"])
        wind_buffer.append(row["wind_speed"])
        pm10_buffer = pm10_buffer[-96:]
        pressure_buffer = pressure_buffer[-96:]
        wind_buffer = wind_buffer[-96:]

        hist_pd = pd.concat([hist_pd, pd.DataFrame({"datetime": [dt], "precipitation": [row["precipitation"]]})], ignore_index=True).iloc[-168:]

    # --- Return full forecast DataFrame ---
    return spark.createDataFrame(pd.DataFrame(forecasts))
