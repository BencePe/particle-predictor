import pandas as pd
import numpy as np

from datetime import timedelta
from src.fetching.data_fetching import get_prediction_input_data, logger
from src.model.model_building import plot_predictions, apply_residual_correction
from src.model.data_processing import add_unified_features


from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, coalesce, min as spark_min, max as spark_max, when
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

from src.config import FEATURE_COLUMNS, DEBRECEN_ELEVATION

def predict_future_air_quality(spark: SparkSession, model: PipelineModel, residual_model: PipelineModel = None):
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
        preds_df = recursive_pm10_forecast(
    spark, model, hist_df, future_w, periods=7*24, residual_model=residual_model
)

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
    periods: int = 7 * 24,
    residual_model: PipelineModel = None
):
    """
    Returns Spark DF [datetime, pm10, is_future=True] for the next `periods` hours,
    computing all FEATURE_COLUMNS in Pandas, then handing each row to Spark for prediction.
    """

    # 1) pull everything into pandas
    hist_pd   = hist_df.toPandas().sort_values("datetime").reset_index(drop=True)
    future_pd = future_weather_df.toPandas().sort_values("datetime").reset_index(drop=True)

    # we'll accumulate our forecasted rows here
    forecasts = []

    # how many steps we actually can do?
    n_steps = min(periods, len(future_pd))

    for i in range(n_steps):
        logger.debug("Length of history: ",len(hist_pd))
        # --- a) set up our new row's raw values ---
        dt    = future_pd.at[i, "datetime"]
        temp  = future_pd.at[i, "temperature"]
        hum   = future_pd.at[i, "humidity"]
        pres  = future_pd.at[i, "pressure"]
        wind  = future_pd.at[i, "wind_speed"]
        wdir  = future_pd.at[i, "wind_dir"]
        elev  = DEBRECEN_ELEVATION
   

        # --- b) build the feature dict using pandas history ---
        feat = {
            "datetime": dt,
            "pm10":    np.nan,             # placeholder
            "temperature": temp,
            "humidity":    hum,
            "pressure":    pres,
            "wind_speed":  wind,
            "wind_dir":    wdir,
            "elevation":   elev,
            "is_urban":    True,
            "is_future":   True,
            "month":       dt.month,
            "hour":        dt.hour,
            "is_weekend":  int(dt.weekday() >= 5),
            "is_rush_hour": int((6 <= dt.hour <= 10) or (16 <= dt.hour <= 22)),
            "hour_sin":    np.sin(2*np.pi * dt.hour    / 24.0),
            "hour_cos":    np.cos(2*np.pi * dt.hour    / 24.0),
            "month_sin":   np.sin(2*np.pi * dt.month   / 12.0),
            "month_cos":   np.cos(2*np.pi * dt.month   / 12.0),
            "day_of_year_sin": np.sin(2*np.pi * dt.timetuple().tm_yday / 365.0),
            "day_of_year_cos": np.cos(2*np.pi * dt.timetuple().tm_yday / 365.0),
            "hour_weekend": dt.hour * int(dt.weekday() >= 5),
            "spring_indicator": int(3 <= dt.month <= 5),
            "summer_indicator": int(6 <= dt.month <= 8),
            "fall_indicator":   int(9 <= dt.month <= 11),
            "winter_indicator": int(dt.month == 12 or dt.month <= 2),
        }

        # helper for safe lags
        def safe_lag(series, lag):
            try:
                return series.iloc[-lag]
            except IndexError:
                return series.iloc[0] if len(series) > 0 else np.nan


        # --- c) lag features ---
        for lag in (3,6,12,24,48,72,168):
            feat[f"pm10_lag{lag}"] = safe_lag(hist_pd["pm10"], lag)
        
        for lag in (3, 6, 12, 24):
            feat[f"temp_lag{lag}"] = safe_lag(hist_pd["temperature"], lag)
            feat[f"humidity_lag{lag}"] = safe_lag(hist_pd["humidity"], lag)
            feat[f"pressure_lag{lag}"] = safe_lag(hist_pd["pressure"], lag)
            feat[f"wind_speed_lag{lag}"] = safe_lag(hist_pd["wind_speed"], lag)
            
        # --- d) rolling averages & stddevs ---
        def safe_roll(series, window, func):
            data = series.iloc[-window:] if len(series) >= window else series
            return getattr(data, func)() if len(data)>0 else np.nan

        for w, name in ((3,"3h"),(6,"6h"),(12,"12h"),(24,"24h"),(48,"48h"),(72,"72h"),(168,"weekly")):
            feat[f"{name}_pm10_avg"] = safe_roll(hist_pd["pm10"], w, "mean")
            if name in ("3h","6h","12h","24h","weekly"):
                feat[f"{name}_pm10_std"] = safe_roll(hist_pd["pm10"], w, "std")
        feat["3h_temp_avg"]   = safe_roll(hist_pd["temperature"], 3, "mean")
        feat["12h_temp_avg"]  = safe_roll(hist_pd["temperature"],12,"mean")
        feat["3h_humidity_avg"]  = safe_roll(hist_pd["humidity"],3,"mean")
        feat["12h_humidity_avg"] = safe_roll(hist_pd["humidity"],12,"mean")
        feat["3h_pressure_avg"]  = safe_roll(hist_pd["pressure"],3,"mean")
        feat["12h_pressure_avg"] = safe_roll(hist_pd["pressure"],12,"mean")
        feat["3h_wind_speed_avg"] = safe_roll(hist_pd["wind_speed"],3,"mean")
        feat["12h_wind_speed_avg"]= safe_roll(hist_pd["wind_speed"],12,"mean")

        # --- e) diffs & volatility & acceleration & rates ---
        feat["pm10_diff_3h"]  = feat["pm10"] - feat["pm10_lag3"]   if not np.isnan(feat["pm10_lag3"]) else 0.0
        feat["pm10_diff_6h"]  = feat["pm10"] - feat["pm10_lag6"]   if not np.isnan(feat["pm10_lag6"]) else 0.0
        feat["pm10_diff_12h"] = feat["pm10"] - feat["pm10_lag12"]  if not np.isnan(feat["pm10_lag12"]) else 0.0
        feat["pm10_diff_24h"] = feat["pm10"] - feat["pm10_lag24"]  if not np.isnan(feat["pm10_lag24"]) else 0.0
        feat["pm10_diff_48h"] = feat["pm10"] - feat["pm10_lag48"]  if not np.isnan(feat["pm10_lag48"]) else 0.0

        # Volatility features 
        feat["pm10_volatility_3h"] = safe_roll(hist_pd["pm10"], 3, "std") / (feat["3h_pm10_avg"] + 1e-5)
        feat["pm10_volatility_6h"] = safe_roll(hist_pd["pm10"], 6, "std") / (feat["6h_pm10_avg"] + 1e-5)
        feat["pm10_volatility_12h"] = safe_roll(hist_pd["pm10"], 12, "std") / (feat["12h_pm10_avg"] + 1e-5)
        feat["pm10_volatility_24h"] = safe_roll(hist_pd["pm10"], 24, "std") / (feat["24h_pm10_avg"] + 1e-5)

        # Acceleration features
        feat["pm10_acceleration_3h"] = feat["pm10_diff_3h"] - (safe_lag(hist_pd["pm10"].diff(3).dropna(), 3) if len(hist_pd)>3 else 0.0)
        feat["pm10_acceleration_12h"] = feat["pm10_diff_12h"] - (safe_lag(hist_pd["pm10"].diff(12).dropna(), 12) if len(hist_pd)>12 else 0.0)
        feat["pm10_acceleration"] = feat["pm10_diff_24h"] - (safe_lag(hist_pd["pm10"].diff(24).dropna(), 24) if len(hist_pd)>24 else 0.0)

        # Rate of change features
        feat["pm10_rate_of_change_3h"] = feat["pm10_diff_3h"] / (feat["pm10_lag3"] + 1e-5)
        feat["pm10_rate_of_change_12h"] = feat["pm10_diff_12h"] / (feat["pm10_lag12"] + 1e-5)

        # --- f) precipitation features ---
        # Since the original function has no 'precipitation' column, we'll set these to 0
        feat["is_precipitation"] = 0
        feat["precipitation_intensity"] = 0
        feat["recent_rain"] = 0
        feat["rain_last_6h"] = 0
        feat["rain_last_12h"] = 0
        feat["rain_last_24h"] = 0
        feat["cumulative_24h_precip"] = 0

        # --- g) interaction and trend features ---
        feat["wind_speed_humidity"] = wind * hum
        feat["temp_pressure"] = temp * pres
        feat["dew_point"] = temp - (100 - hum)/5
        feat["pm10_dew_point"] = feat["pm10"] * feat["dew_point"] if not np.isnan(feat["pm10"]) else 0.0
        feat["pollution_drift_week"] = (feat["pm10"] - feat["weekly_pm10_avg"])/(feat["weekly_pm10_avg"] + 1e-5) if not np.isnan(feat["pm10"]) else 0.0

        # Pressure trend features
        feat["pressure_3h_trend"] = safe_roll(hist_pd["pressure"], 3, "mean") - pres if len(hist_pd) > 3 else 0.0
        feat["pressure_6h_trend"] = safe_roll(hist_pd["pressure"], 6, "mean") - pres if len(hist_pd) > 6 else 0.0
        feat["pressure_12h_trend"] = safe_roll(hist_pd["pressure"], 12, "mean") - pres if len(hist_pd) > 12 else 0.0
        feat["pressure_24h_trend"] = safe_roll(hist_pd["pressure"], 24, "mean") - pres if len(hist_pd) > 24 else 0.0

        # Temperature trend features
        feat["temp_3h_trend"] = safe_roll(hist_pd["temperature"], 3, "mean") - temp if len(hist_pd) > 3 else 0.0
        feat["temp_6h_trend"] = safe_roll(hist_pd["temperature"], 6, "mean") - temp if len(hist_pd) > 6 else 0.0
        feat["temp_12h_trend"] = safe_roll(hist_pd["temperature"], 12, "mean") - temp if len(hist_pd) > 12 else 0.0
        feat["temp_24h_trend"] = safe_roll(hist_pd["temperature"], 24, "mean") - temp if len(hist_pd) > 24 else 0.0

        # Humidity trend features
        feat["humidity_3h_trend"] = safe_roll(hist_pd["humidity"], 3, "mean") - hum if len(hist_pd) > 3 else 0.0
        feat["humidity_12h_trend"] = safe_roll(hist_pd["humidity"], 12, "mean") - hum if len(hist_pd) > 12 else 0.0

        # Additional interaction features
        feat["humidity_temp_index"] = hum * temp
        feat["humidity_pressure_index"] = hum * pres / 1000
        feat["temp_wind_index"] = temp * wind
        feat["pressure_change_velocity"] = (pres - safe_lag(hist_pd["pressure"], 3)) / 3.0 if len(hist_pd) > 3 else 0.0
        feat["rapid_pressure_change"] = 1 if abs(pres - safe_lag(hist_pd["pressure"], 6)) > 5 and len(hist_pd) > 6 else 0
        feat["rapid_humidity_increase"] = 1 if (hum - safe_lag(hist_pd["humidity"], 3)) > 15 and len(hist_pd) > 3 else 0
        feat["wind_dir_stability"] = np.std(hist_pd["wind_dir"].iloc[-24:]) / 180.0 if len(hist_pd) >= 24 else 0.0
        feat["wind_dir_8"] = int(((wdir + 22.5) % 360) / 45)
        feat["wind_speed_cat"] = 0 if wind < 2 else (1 if wind < 5 else (2 if wind < 10 else 3))
        feat["pollution_load"] = feat["pm10"] * wind if not np.isnan(feat["pm10"]) else 0.0
        feat["pm10_12h_avg_sq"] = feat["12h_pm10_avg"] ** 2
        feat["avg12h_times_diff12h"] = feat["12h_pm10_avg"] * feat["pm10_diff_12h"]
        feat["temp_humidity_interaction"] = temp * hum / 100
        feat["temp_pressure_interaction"] = temp * pres / 1000
        feat["wind_temp_cross"] = 1 if temp > 15 and wind > 5 else 0

        # --- h) weather change features ---
        feat["weather_change_index"] = abs(feat["pressure_6h_trend"])/5 + abs(feat["temp_6h_trend"])/3 + abs(feat["humidity_3h_trend"])/10
        feat["atmospheric_instability"] = abs(feat["pressure_change_velocity"]) * (feat["3h_pm10_std"] / (feat["3h_pm10_avg"] + 1))
        feat["weather_system_change"] = 1 if (abs(feat["pressure_24h_trend"]) > 10 or abs(feat["temp_24h_trend"]) > 5 or feat["rapid_pressure_change"] == 1) else 0

        # --- i) hand it to Spark to vectorize & scale & predict ---
        single_sdf = spark.createDataFrame(pd.DataFrame([feat]))
        single_feats = single_sdf.select("datetime", *FEATURE_COLUMNS)
        
        # Step 1: Predict using GBT model
        pred_row = model.transform(single_feats).first()
        base_pred = float(pred_row["prediction"])
        pred = base_pred

        # Step 2: If residual model exists, apply it using base_pred
        if residual_model:
            # Add base_prediction column manually
            enriched = single_feats.withColumn("base_prediction", lit(base_pred))
            res_row = residual_model.transform(enriched).first()
            residual_correction = float(res_row["residual_prediction"])
            pred += residual_correction



        # add this prediction to our forecasts
        forecasts.append({"datetime": dt, "pm10": pred})

        # write back into hist_pd for next iteration
        new_row = pd.DataFrame({
            "datetime":    [dt],
            "pm10":        [pred],
            "temperature": [temp],
            "humidity":    [hum],
            "pressure":    [pres],
            "wind_speed":  [wind],
            "wind_dir":    [wdir]
        })
        hist_pd = pd.concat([hist_pd, new_row], ignore_index=True)

    if i % 24 == 0:
       logger.debug(f"{dt} | GBT: {base_pred:.2f} | Residual: {residual_correction:.2f} | Final: {pred:.2f}")

    # build final Spark DataFrame
    out = pd.DataFrame(forecasts, columns=["datetime","pm10"])
    sdf = spark.createDataFrame(out) \
               .withColumn("is_future", lit(True))
    return sdf

