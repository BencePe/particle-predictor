import pandas as pd
import numpy as np

from datetime import timedelta
from src.fetching.data_fetching import get_prediction_input_data, logger
from src.model.model_building import plot_predictions


from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, coalesce, min as spark_min, max as spark_max
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

from src.config import FEATURE_COLUMNS, DEBRECEN_ELEVATION

def predict_future_air_quality(spark: SparkSession, model: PipelineModel):
    try:
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
        preds_df = recursive_pm10_forecast(spark, model, hist_df, future_w, periods=5*24)

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
    spark: SparkSession,
    model: PipelineModel,
    hist_df,
    future_weather_df,
    periods: int = 5 * 24
):
    """
    Returns Spark DF [datetime, pm10(predicted), is_future] for next `periods` hours,
    building every feature in FEATURE_COLUMNS recursively.
    """
    # Pull history into pandas
    hist = hist_df.sort("datetime").toPandas().reset_index(drop=True)
    fw = future_weather_df.sort("datetime").toPandas().reset_index(drop=True)
    
    # Log key information
    logger.info(f"Historical data range: {hist['datetime'].min()} to {hist['datetime'].max()}")
    logger.info(f"Future weather data range: {fw['datetime'].min()} to {fw['datetime'].max()}")
    logger.info(f"Number of future weather rows available: {len(fw)}")
    
    # Ensure we have enough future weather data
    if len(fw) < periods:
        logger.warning(f"Not enough future weather data. Requested {periods} periods but only {len(fw)} available.")
        periods = min(periods, len(fw))
    
    # Get the last known PM10 value for diagnostics
    last_pm10 = hist["pm10"].iloc[-1]
    last_pm2_5 = hist["pm2_5"].iloc[-1]
    logger.info(f"Last known PM10 value: {last_pm10}")
    logger.info(f"Last known PM2_5 value: {last_pm2_5}")

    forecasts = [] 

    for i in range(periods):

        if i >= len(fw):
            logger.warning(f"Ran out of future weather data at iteration {i}")
            break

        dt = fw.loc[i, "datetime"]
        row = fw.loc[i]

        # Log each iteration
        logger.info(f"Forecasting for datetime: {dt}")


        # --- build core of feat dict ---
        feat = dict(
            datetime    = dt,
            pm10        = np.nan,
            pm2_5       = np.nan,
            temperature = row["temperature"],
            humidity    = row["humidity"],
            pressure    = row["pressure"],
            wind_speed  = row["wind_speed"],
            wind_dir    = row["wind_dir"],
            elevation   = DEBRECEN_ELEVATION,
            is_urban    = True,
            is_future   = True,
            month       = dt.month,
            hour        = dt.hour,
            is_weekend  = int(dt.weekday() >= 5),
            is_rush_hour= int((7 <= dt.hour <= 9) or (16 <= dt.hour <= 19)),
            hour_sin    = np.sin(2*np.pi * dt.hour / 24),
            hour_cos    = np.cos(2*np.pi * dt.hour / 24),
            month_sin   = np.sin(2*np.pi * dt.month/12),
            month_cos   = np.cos(2*np.pi * dt.month/12),
            day_of_year_sin = np.sin(2*np.pi * dt.timetuple().tm_yday / 365),
            day_of_year_cos = np.cos(2*np.pi * dt.timetuple().tm_yday / 365),
            hour_weekend   = dt.hour * int(dt.weekday() >= 5),
        )

        hist_pm = hist["pm10"]
        N = len(hist_pm)

        # --- safe lag features ---
        for lag in [12, 24, 48, 72, 168]:
            if N > lag:
                feat[f"pm10_lag{lag}"] = hist_pm.iloc[-lag]
            else:
                # fallback to earliest available
                feat[f"pm10_lag{lag}"] = hist_pm.iloc[0]

        # --- rolling stats (will automatically shrink if N < window) ---
        def safe_roll(series, window, func):
            """
            Compute rolling aggregation over the last `window` values of a pandas Series.
            Falls back to the entire series if its length is less than `window`.
            """
            # pick slice
            if N >= window:
                data = series.iloc[-window:]
            elif N > 0:
                data = series
            else:
                logger.warning("safe_roll: empty series for %s, returning NaN", func)
                return np.nan

            try:
                # e.g. data.mean() or data.std()
                result = getattr(data, func)()
                logger.debug("safe_roll: %s over %d rows = %s", func, len(data), result)
                return result
            except Exception as ex:
                logger.error(
                    "safe_roll ERROR for func=%s window=%d: %s",
                    func, window, str(ex),
                    exc_info=True
                )
                return np.nan


        feat.update({
            "12h_pm10_avg":    safe_roll(hist_pm, 12,  "mean"),
            "24h_pm10_avg":    safe_roll(hist_pm, 24,  "mean"),
            "48h_pm10_avg":    safe_roll(hist_pm, 48,  "mean"),
            "72h_pm10_avg":    safe_roll(hist_pm, 72,  "mean"),
            "weekly_pm10_avg": safe_roll(hist_pm,168,  "mean"),
            "12h_pm10_std":    safe_roll(hist_pm, 12,  "std"),
            "24h_pm10_std":    safe_roll(hist_pm, 24,  "std"),
            "weekly_pm10_std": safe_roll(hist_pm,168,  "std"),
        })

        feat["pm10_12h_avg_sq"] = feat["12h_pm10_avg"] * feat["12h_pm10_avg"]

        feat["pm10_volatility_24h"] = (
            feat["24h_pm10_std"] / (feat["24h_pm10_avg"] + 1e-5)
        )

        # Differences & acceleration (guard same way)
        if N > 49:
            feat["pm10_diff_12h"]   = hist_pm.iloc[-1] - hist_pm.iloc[-13]
            feat["pm10_diff_24h"]   = hist_pm.iloc[-1] - hist_pm.iloc[-25]
            feat["pm10_diff_48h"]   = hist_pm.iloc[-1] - hist_pm.iloc[-49]
            prev_diff_24            = hist_pm.iloc[-25] - hist_pm.iloc[-49]
            feat["pm10_acceleration"]= feat["pm10_diff_24h"] - prev_diff_24
        else:
            feat.update(dict.fromkeys([
                "pm10_diff_12h","pm10_diff_24h","pm10_diff_48h","pm10_acceleration"
            ], np.nan))

        feat["avg12h_times_diff12h"] = feat["12h_pm10_avg"] * feat.get("pm10_diff_12h", 0)

        # Weather & interaction trends
        pres = hist["pressure"]
        tmp  = hist["temperature"]
        feat.update({
            "pressure_12h_trend":  (pres.iloc[-12:].mean() - row["pressure"]) if N>=12 else np.nan,
            "pressure_24h_trend":  (pres.iloc[-24:].mean() - row["pressure"]) if N>=24 else np.nan,
            "temp_12h_trend":      (tmp.iloc[-12:].mean()  - row["temperature"]) if N>=12 else np.nan,
            "temp_24h_trend":      (tmp.iloc[-24:].mean()  - row["temperature"]) if N>=24 else np.nan,
            "humidity_temp_index": row["humidity"]*row["temperature"],
            "temp_wind_index":     row["temperature"]*row["wind_speed"],
            "pressure_change_velocity":(
                (row["pressure"] - pres.iloc[-3]) / 3.0
            ) if N>=3 else np.nan,
            "wind_dir_stability":  (hist["wind_dir"].iloc[-24:].std(ddof=0)/180.0) if N>=24 else np.nan,
            "pm_ratio":            feat["pm2_5"]/(hist_pm.iloc[-1]+1e-7),
            "pollution_load":      hist_pm.iloc[-1]*row["wind_speed"],
        })

        # predict
        single = spark.createDataFrame(pd.DataFrame([feat]))
        pred = float(model.transform(single).first()["prediction"])
        logger.info(f"Predicted PM10 for {dt}: {pred}")
        forecasts.append((dt, pred))

        # update hist for next iteration
        new_row = feat.copy()
        new_row["pm10"] = pred
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    # build and return Spark DF of forecasts
    logger.info(f"Total forecasts generated: {len(forecasts)}")
    if len(forecasts) == 0:
        logger.error("No forecasts were generated!")
        return None
        
    return (
        spark
        .createDataFrame(pd.DataFrame(forecasts, columns=["datetime","pm10"]))
        .withColumn("is_future", lit(True))
    )
