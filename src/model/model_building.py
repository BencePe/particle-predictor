"""
Model building and evaluation functions.
"""

import os
import logging
from datetime import datetime

from matplotlib import dates
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

from src.config import MODEL_DIR, MODEL_PARAMS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Utility: filter features
def filter_valid_features(df, feature_list):
    available_cols = set(df.columns)
    valid_features = [f for f in feature_list if f in available_cols]
    logger.info(f"Using {len(valid_features)} features for model training")
    return valid_features


# -------------------------------------------------------------------------------
# 1) Build improved base pipeline (assembler + scaler only)
def build_improved_model_pipeline():
    """
    Build an improved ML pipeline *without* regressor, using only top-ranked features.
    Returns (base_pipeline, feature_list).
    """
    logger.info("Building pruned pipeline for PM10 prediction (top 12 features)...")

    # Top 12 features from latest feature importance ranking
    top_features = [
        "12h_pm10_avg",
        "avg12h_times_diff12h",
        "pm10_lag12",
        "pm10_diff_12h",
        "hour",
        "12h_pm10_std",
        "wind_speed",
        "pm10_diff_48h",
        "pollution_load",
        "pm10_lag48",
        "pm10_lag24",
        "pm10_diff_24h"
    ]

    assembler = VectorAssembler(
        inputCols=top_features,
        outputCol="assembled_features",
        handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="assembled_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    pipeline = Pipeline(stages=[assembler, scaler])
    logger.info("Base pipeline (pruned) created successfully")
    return pipeline, top_features



# -------------------------------------------------------------------------------
# 2) Predefined RF pipeline
def build_rf_model_pipeline():
    logger.info("Building Random Forest pipeline for PM10 prediction...")
    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="keep"
    )
    rf = RandomForestRegressor(
        labelCol="pm10",
        featuresCol="features",
        numTrees=MODEL_PARAMS.get("num_trees", 250),
        maxDepth=MODEL_PARAMS.get("max_depth_rf", 3),
        seed=MODEL_PARAMS.get("seed"),
        featureSubsetStrategy=MODEL_PARAMS.get("featureSubsetStrategy", "sqrt"),
        minInstancesPerNode=MODEL_PARAMS.get("minInstancesPerNode", 5)
    )
    pipeline = Pipeline(stages=[assembler, rf])
    logger.info("RF pipeline created successfully")
    return pipeline, FEATURE_COLUMNS


# -------------------------------------------------------------------------------
# 3) Train & evaluate
def train_model(pipeline, train_df):
    logger.info("Training PM10 prediction model...")
    start_time = datetime.now()
    model = pipeline.fit(train_df)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Model training completed in {elapsed:.2f}s")
    return model


def evaluate_model(model_or_df, test_df, prediction_col="prediction", is_transformed=False):
    """
    Evaluate model performance.
    
    Args:
        model_or_df: Either a PipelineModel to transform the data or a DataFrame with predictions already
        test_df: Test data DataFrame
        prediction_col: Name of the prediction column
        is_transformed: If True, model_or_df is treated as already transformed data
    
    Returns:
        Dictionary of performance metrics
    """
    logger.info("Evaluating model performance...")
    
    try:
        if is_transformed:
            preds = model_or_df  # Already transformed
        else:
            preds = model_or_df.transform(test_df)
        
        evaluator_rmse = RegressionEvaluator(
            labelCol="pm10", predictionCol=prediction_col, metricName="rmse"
        )
        rmse = evaluator_rmse.evaluate(preds)
        
        evaluator_r2 = RegressionEvaluator(
            labelCol="pm10", predictionCol=prediction_col, metricName="r2"  
        )
        r2 = evaluator_r2.evaluate(preds)
        
        evaluator_mae = RegressionEvaluator(
            labelCol="pm10", predictionCol=prediction_col, metricName="mae"
        )
        mae = evaluator_mae.evaluate(preds)
        
        logger.info("Model Evaluation Metrics:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        
        return {"rmse": rmse, "r2": r2, "mae": mae}
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None

# -------------------------------------------------------------------------------
# 4) Save & load
def save_model(model, model_name="pm10_gbt_model"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}")
    model.write().overwrite().save(path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(model_path: str) -> PipelineModel:
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"No model at {model_path}")
    # find subfolder with metadata/
    if os.path.isdir(os.path.join(model_path, "metadata")):
        real_path = model_path
    else:
        subs = [os.path.join(model_path, d) for d in os.listdir(model_path)]
        real_path = next((d for d in subs if os.path.isdir(os.path.join(d, "metadata"))), None)
        if real_path is None:
            raise FileNotFoundError(f"No valid Spark model in {model_path}")

    logger.info(f"Loading model from {real_path}")
    return PipelineModel.load(real_path)


# -------------------------------------------------------------------------------
# 5) Plotting (unchanged)
def plot_predictions(predictions_df):
    """
    Enhanced version with date validation and better visual separation
    """
    try:
        pdf = predictions_df.toPandas()
        pdf['datetime'] = pd.to_datetime(pdf['datetime'])
        pdf = pdf.sort_values('datetime').reset_index(drop=True)

        historical = pdf[pdf['is_future'] == False]
        future     = pdf[pdf['is_future'] == True]

        if not historical.empty and not future.empty:
            gap = future['datetime'].min() - historical['datetime'].max()
            logger.info(f"Gap between history & forecast: {gap}")
            if gap.total_seconds() > 86400:
                logger.warning(f"Large gap: {gap}")

        fig, ax = plt.subplots(figsize=(18, 9))

        ax.plot(
            historical['datetime'], historical['pm10'],
            linewidth=2.5, label=f'Historical ({len(historical)/24} days)',
            marker='o', markersize=4, alpha=0.9
        )
        if not future.empty:
            ax.plot(
                future['datetime'], future['prediction'],
                linewidth=3.5, linestyle='--', marker='^', markersize=8,
                label='Forecast'
            )
            sep = future['datetime'].min()
            ax.axvline(sep, linestyle=':', linewidth=3)
            ax.annotate(
                f'Start: {sep.strftime("%b %d")}',
                xy=(sep, ax.get_ylim()[1]*0.95),
                xytext=(15, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->')
            )

        ax.xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlabel('Date')
        ax.set_ylabel('PM10 (μg/m³)')
        ax.set_title('PM10: History vs Forecast')
        ax.legend()
        plt.tight_layout()
        fname = f"pm10_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {fname}")
        plt.show()

    except Exception as e:
        logger.error(f"Plotting error: {e}")
        raise


# -------------------------------------------------------------------------------
# 6) CV & feature importance (unchanged)
def analyze_feature_importance(model, feature_list=None, top_n=20):
    logger.info("Analyzing feature importance...")
    if feature_list is None:
        feature_list = [c for c in FEATURE_COLUMNS if c not in ("pm10", "pm2_5")]

    tree = model.stages[-1] if isinstance(model, PipelineModel) else model
    try:
        importances = tree.featureImportances.toArray()
        feats = feature_list[: len(importances)]
        pairs = list(zip(feats, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        for i,(f,i_) in enumerate(pairs[:top_n], 1):
            logger.info(f"{i}. {f}: {i_:.4f}")

        # plot
        plt.figure(figsize=(12,8))
        top_feats = pairs[:top_n]
        names, vals = zip(*top_feats)
        plt.barh(range(len(names)), vals, align='center')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        logger.info("Feature importance plot saved as 'feature_importance.png'")
        return pairs
    except Exception as e:
        logger.error(f"Error extracting feature importances: {e}")
        return []


def select_top_features(feature_importances, top_n=15):
    top_feats = [f for f,_ in feature_importances[:top_n]]
    logger.info(f"Selected top {len(top_feats)} features: {top_feats}")
    return top_feats


def train_and_evaluate_cv(pipeline, splits):
    metrics_list = []
    for i,(train_df,test_df) in enumerate(splits,1):
        logger.info(f"CV split {i}")
        model = train_model(pipeline, train_df)
        metrics = evaluate_model(model, test_df)
        metrics_list.append(metrics)
        logger.info(f" Split {i} → RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    avg = {
        k: sum(m[k] for m in metrics_list)/len(metrics_list)
        for k in metrics_list[0]
    }
    logger.info(f"Avg CV → RMSE={avg['rmse']:.4f}, R²={avg['r2']:.4f}")
    return avg, metrics_list


# -------------------------------------------------------------------------------
# 7) Residual stacking
def build_residual_pipeline(base_model, train_df, feature_cols):
    # Add base model predictions and calculate residuals
    preds = base_model.transform(train_df).withColumnRenamed("prediction", "base_prediction")
    
    # Calculate residuals
    preds = preds.withColumn("residual", col("pm10") - col("base_prediction"))
    
    # Build the pipeline as before
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="res_assembled",
        handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="res_assembled",
        outputCol="res_features",
        withStd=True,
        withMean=True
    )
    
    # Set a custom output column name for LinearRegression to avoid conflicts
    lr = LinearRegression(
        labelCol="residual",
        featuresCol="res_features",
        regParam=0.1,
        elasticNetParam=0.5,
        predictionCol="residual_prediction"  # Custom name instead of default "prediction"
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    res_model = pipeline.fit(preds)
    logger.info("Residual model trained")
    return res_model


def apply_residual_correction(base_model, residual_model, df):
    # Apply base model
    base_preds = base_model.transform(df).withColumnRenamed("prediction", "base_prediction")
    
    # Apply residual model (will create "residual_prediction" column)
    corrected = residual_model.transform(base_preds)
    
    # Calculate final prediction by adding base prediction + residual correction
    corrected = corrected.withColumn(
        "final_prediction", 
        col("base_prediction") + col("residual_prediction")
    )
    
    return corrected

# -------------------------------------------------------------------------------
# 8) Hyperopt GBT search
def hyperopt_gbt(train_df, val_df, assembler_stage, scaler_stage, max_evals=30):
    def objective(params):
        gbt = GBTRegressor(
            labelCol="pm10",
            featuresCol="features",
            maxDepth=int(params["maxDepth"]),
            maxIter=int(params["maxIter"]),
            stepSize=params["stepSize"],
            subsamplingRate=params["subsamplingRate"],
            seed=MODEL_PARAMS.get("seed", 42)
        )
        pipe = Pipeline(stages=[assembler_stage, scaler_stage, gbt])
        mdl = pipe.fit(train_df)
        rmse = RegressionEvaluator(labelCol="pm10", metricName="rmse") \
                 .evaluate(mdl.transform(val_df))
        return {"loss": rmse, "status": STATUS_OK}

    space = {
        "maxDepth": scope.int(hp.quniform("maxDepth",  2,   5,   1)),
        "maxIter":  scope.int(hp.quniform("maxIter",  30, 70, 10)),
        "stepSize":        hp.loguniform("stepSize", -4.5,  -2),
        "subsamplingRate": hp.uniform("subsamplingRate", 0.6, 0.9)
    }
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    logger.info(f"Hyperopt best params: {best}")
    return best, trials

#maxDepth': np.float64(5.0), 'maxIter': np.float64(70.0), 'stepSize': np.float64(0.1530311792275945), 'subsamplingRate': np.float64(0.9024031209262541)