"""
Model building and evaluation functions.
"""

import os
import logging
from datetime import datetime

from matplotlib import dates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

from src.config import MODEL_DIR, MODEL_PARAMS, FEATURE_COLUMNS
from src.model.time_splits import hybrid_time_validation

logger = logging.getLogger(__name__)

# def filter_valid_features(df, feature_list):
#     available_cols = set(df.columns)
#     valid_features = [f for f in feature_list if f in available_cols]
#     logger.info(f"Using {len(valid_features)} features for model training")
#     return valid_features

def build_improved_model_pipeline():
    logger.info("Building pruned pipeline for PM10 prediction...")
    
    top_features = [c for c in FEATURE_COLUMNS if c != "pm10"]
    assembler = VectorAssembler(
        inputCols=top_features,
        outputCol="temp_assembled_features",
        handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="temp_assembled_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    pipeline = Pipeline(stages=[assembler, scaler])
    logger.info("Base Vector Assembler, Standard Scaler and GBT added.")
    return pipeline, top_features

def train_model(pipeline, train_df):
    logger.info("Training PM10 prediction model...")
    start_time = datetime.now()
    model = pipeline.fit(train_df)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Model training completed in {elapsed:.2f}s")
    return model

def evaluate_model(model, df, prediction_col="prediction", label_col="pm10", is_transformed=False):
    if not is_transformed and model is not None:  # Added model existence check
        df = model.transform(df)
    
    # Verify prediction column exists
    if prediction_col not in df.columns:
        available = "\n".join(df.columns)
        raise ValueError(f"Missing {prediction_col} column. Available columns:\n{available}")

    # Verify label column exists
    if label_col not in df.columns:
        available = "\n".join(df.columns)
        raise ValueError(f"Missing {label_col} label column. Available columns:\n{available}")

    # Create evaluators with dynamic label column
    evaluator_rmse = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col, 
        metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="r2"
    )

    rmse = evaluator_rmse.evaluate(df)
    mae = evaluator_mae.evaluate(df)
    r2 = evaluator_r2.evaluate(df)
    
    return {"rmse": rmse, "mae": mae, "r2": r2}

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

def analyze_feature_importance(model, feature_list=None, top_n=30):
    logger.info("Analyzing feature importance...")
    if feature_list is None:
        feature_list = [c for c in FEATURE_COLUMNS if c not in ("pm10")]

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

def select_top_features(feature_importances, top_n=20):
    top_feats = [f for f,_ in feature_importances[:top_n]]
    logger.info(f"Selected top {len(top_feats)} features: {top_feats}")
    return top_feats

def train_and_evaluate_cv(pipeline, splits):
    """
    Train model and evaluate on multiple train/test splits.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Spark ML pipeline to train
    splits : list of tuples
        List of (train_df, test_df) splits
    """
    metrics_list = []
    for i, (train_df, test_df) in enumerate(splits, 1):
        logger.info(f"CV split {i}")
        model = train_model(pipeline, train_df)

        # Transform test_df to get predictions
        pred_df = model.transform(test_df)
        if pred_df is None:
            logger.error(f"Model.transform returned None on CV split {i}")
            continue
        
        # Evaluate on transformed data
        metrics = evaluate_model(model, pred_df, prediction_col="prediction")
        metrics_list.append(metrics)

        logger.info(f" Split {i} → RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    avg = {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in metrics_list[0]
    }
    logger.info(f"Avg CV → RMSE={avg['rmse']:.4f}, R²={avg['r2']:.4f}")
    return avg, metrics_list

def train_with_hybrid_cv(df, pipeline_or_builder, datetime_col="datetime"):
    logger.info("Generating hybrid time-based validation splits...")

    # Pre-partition for performance
    df = df.withColumn("year", F.year(F.col(datetime_col))) \
           .withColumn("month", F.month(F.col(datetime_col))) \
           .repartition(60, "year", "month") \
           .cache()

    # Create splits
    splits = hybrid_time_validation(df, datetime_col=datetime_col)

    # 🛠 Check if we received a callable (builder function) or an already-built pipeline
    if callable(pipeline_or_builder):
        pipeline, feats = pipeline_or_builder()
    else:
        pipeline = pipeline_or_builder  # Already a Pipeline object

    # Run evaluation across splits
    avg_metrics, all_metrics = train_and_evaluate_cv(pipeline, splits)
    
    logger.info(f"Avg CV metrics: {avg_metrics}")
    
    return avg_metrics

def build_residual_pipeline(base_model, train_df, feature_cols):
    # Get predictions from complete base model
    preds = base_model.transform(train_df)
    
    # Calculate residuals using proper prediction column
    preds = preds.withColumn("residual", col("pm10") - col("prediction"))
    
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
        predictionCol="residual_prediction"
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    res_model = pipeline.fit(preds)
    logger.info("Residual model trained")
    return res_model

def apply_residual_correction(model, residual_model, df):
    # Apply base model
    base_preds = model.transform(df).withColumnRenamed("prediction", "tuned_prediction")
    
    # Apply residual model (will create "residual_prediction" column)
    corrected = residual_model.transform(base_preds)
    
    # Calculate final prediction by adding base prediction + residual correction
    corrected = corrected.withColumn(
        "final_prediction", 
        col("tuned_prediction") + col("residual_prediction")
    )
    
    return corrected

def hyperopt_gbt(train_df, val_df, assembler_stage, scaler_stage, max_evals=30):
    # Get feature names for targeted penalties
    feature_cols = assembler_stage.getInputCols()
    
    # Add a variable to track the best model
    best_model = None
    best_metrics = None
    best_loss = float('inf')
    
    def objective(params):
        nonlocal best_model, best_metrics, best_loss
        
        if params["stepSize"] > 0.15 and params["maxDepth"] > 3:
            return {"loss": float("inf"), "status": STATUS_OK, "msg": "Too deep + too aggressive"}

        # 1. Create GBT with modified colSampleRate to reduce feature dominance
        gbt = GBTRegressor(
            labelCol="pm10",
            featuresCol="features",
            maxDepth=int(params["maxDepth"]),
            maxIter=int(params["maxIter"]),
            stepSize=float(params["stepSize"]),
            maxBins=int(params["maxBins"]),
            minInstancesPerNode=int(params["minInstancesPerNode"]),
            subsamplingRate=float(params["subsamplingRate"]),
            featureSubsetStrategy=params["featureSubsetStrategy"],
            seed=MODEL_PARAMS.get("seed", 42)
        )

        pipeline = Pipeline(stages=[assembler_stage, scaler_stage, gbt])
        model = pipeline.fit(train_df)
        metrics = evaluate_model(model, val_df, prediction_col="prediction")
        val_rmse = metrics["rmse"]

        # Get feature importances
        gbt_model_stage = model.stages[-1]
        importances = gbt_model_stage.featureImportances.toArray()
        
        # Find 3h_pm10_avg importance if it exists
        pm10_avg_importance = 0
        try:
            # Find index of 3h_pm10_avg in feature columns
            if "3h_pm10_avg" and "rolling_max_pm10_24h" in feature_cols:
                idx = feature_cols.index("3h_pm10_avg")
                idy = feature_cols.index("rolling_max_pm10_24h")
                if idx < len(importances):
                    pm10_avg_importance = importances[idx]
                    rolling_max_pm10_24h_importance = importances[idy]
        except:
            pass  # If there's an error, we'll use max importance
            
        max_importance = importances.max()
        
        # Calculate feature distribution metrics
        std_importance = importances.std()
        top_3_sum = np.sort(importances)[-3:].sum()  # Sum of top 3 feature importances
        
        # 1. Imbalance penalty with progressive scaling - reduced weight
        imbalance_penalty = 0.0
        if max_importance > 0.13:  # Adjusted threshold (was 0.10)
            # Linear penalty with gentler scaling
            imbalance_penalty = (max_importance - 0.15) * 20.0  # Reduced from 33.0 * (1.0 + max_importance)
            
        # 2.1 Progressive penalty for 3h_pm10_avg
        pm10_avg_penalty = 0.0
        if pm10_avg_importance > 0:
            # Base penalty for any usage
            pm10_avg_penalty = 0.0
            
            # Progressive penalty based on importance level
            if pm10_avg_importance > 0.15:
                pm10_avg_penalty += (pm10_avg_importance - 0.05) * 10.0
        
        # 2.2 Progressive penalty for 3h_pm10_avg
        rolling_max_pm10_24h_penalty = 0.0
        if rolling_max_pm10_24h_importance > 0:
            # Base penalty for any usage
            rolling_max_pm10_24h_penalty = 0.0
            
            # Progressive penalty based on importance level
            if rolling_max_pm10_24h_importance > 0.15:
                rolling_max_pm10_24h_penalty += (rolling_max_pm10_24h_importance - 0.05) * 10.0
        
        # 3. Penalty for having a few dominant features - reduced weight
        concentration_penalty = 0.0
        if top_3_sum > 0.6:
            concentration_penalty = (top_3_sum - 0.6) * 7.0  # Reduced from 10.0
            
        # 4. Standard deviation penalty - reward more balanced distributions
        distribution_penalty = std_importance * 14.0  # Reduced from 25.0

        # 5. Complexity penalty - adjusted weights
        complexity_penalty = (
            0.3 * int(params["maxDepth"]) +  # Reduced from 0.5
            4.0 * (float(params["stepSize"]) ** 2) +  # Reduced from 5.5
            1.5 * max(0, float(params["subsamplingRate"]) - 0.65)  # Relaxed threshold (was 0.7) and reduced weight
        )

        # Total penalty
        total_penalty = imbalance_penalty + complexity_penalty + rolling_max_pm10_24h_penalty +\
                        pm10_avg_penalty + concentration_penalty + distribution_penalty
                        
        # Base loss
        base_loss = val_rmse + total_penalty
        
        # Add scaled penalty for 3h_pm10_avg instead of hard rejection
        penalized_loss = base_loss
        if pm10_avg_importance > 0:
            # Use a scaled penalty instead of flat 100
            # Start with moderate penalty that increases with importance
            scaled_penalty = 3.0 + pm10_avg_importance * 50.0
            penalized_loss += scaled_penalty
            logger.info(f"  - Applied 3h_pm10_avg penalty: {scaled_penalty:.2f}")
        
        logger.info(f"RMSE: {val_rmse:.4f}, Total penalty: {total_penalty:.4f}, Final loss: {penalized_loss:.4f}")
        logger.info(f"  - Max importance: {max_importance:.4f}, 3h_pm10_avg importance: {pm10_avg_importance:.4f}")
        logger.info(f"  - Penalties: imbalance={imbalance_penalty:.2f}, pm10_avg={pm10_avg_penalty:.2f}, " +
                    f"concentration={concentration_penalty:.2f}, distribution={distribution_penalty:.2f}")
        
        # Modified best model selection logic
        # Allow models with some 3h_pm10_avg usage if they're significantly better
        adjusted_loss = base_loss
        if pm10_avg_importance > 0:
            # Add a modest fixed penalty to prefer non-3h_pm10_avg models
            adjusted_loss += 1 * pm10_avg_importance
        
        # Update best model if better than current best
        if adjusted_loss < best_loss:
            best_loss = adjusted_loss
            best_model = model
            best_metrics = {
                "rmse": val_rmse,
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "max_importance": max_importance,
                "pm10_avg_importance": pm10_avg_importance,
                "penalty": total_penalty,
                "adjusted_loss": adjusted_loss,
                "penalties": {
                    "imbalance": imbalance_penalty,
                    "pm10_avg": pm10_avg_penalty,
                    "concentration": concentration_penalty,
                    "distribution": distribution_penalty,
                    "complexity": complexity_penalty
                }
            }
            logger.info(f"New best model found! Adjusted Loss: {adjusted_loss:.4f}, RMSE: {val_rmse:.4f}")
            
            # Special logging for models with exceptional RMSE
            if val_rmse < 2.3:
                logger.info(f"Exceptional RMSE model! Params: {params}")
        
        return {
            "loss": penalized_loss,
            "status": STATUS_OK,
            "eval_rmse": val_rmse,
            "base_loss": base_loss,
            "penalty": total_penalty,
            "imbalance_penalty": imbalance_penalty,
            "pm10_avg_penalty": pm10_avg_penalty,
            "concentration_penalty": concentration_penalty, 
            "distribution_penalty": distribution_penalty,
            "complexity_penalty": complexity_penalty,
            "max_feature_importance": float(max_importance),
            "pm10_avg_importance": float(pm10_avg_importance),
            "params": params
        }
    
    # Modified search space to focus on promising areas
    space = {
        "maxDepth": hp.choice("maxDepth", [3,4,5]),
        "maxIter": hp.choice("maxIter", [150, 170, 190]),
        "stepSize": hp.uniform("stepSize", 0.1, 0.3),  # Narrower range
        "maxBins": hp.choice("maxBins", [64, 100, 128]),
        "minInstancesPerNode": hp.choice("minInstancesPerNode", [5,8,10]),
        "subsamplingRate": hp.uniform("subsamplingRate", 0.5, 0.75),  # Narrower range
        "featureSubsetStrategy": hp.choice("featureSubsetStrategy", ["auto","sqrt", "onethird", "log2", "0.5"])
    }
    # Run optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials 
    )
    
    # Map best params back to their original types/values
    featureSubsetStrategy_options = ["sqrt", "onethird", "log2", "0.5"]
    maxDepth_options = [3,4,5]
    maxIter_options = [150, 170, 190]
    maxBins_options = [100, 120, 140]
    minInstancesPerNode_options = [5,8,10]
    
    best_params = {
        "maxDepth": maxDepth_options[best["maxDepth"]],
        "maxIter": maxIter_options[best["maxIter"]],
        "stepSize": float(best["stepSize"]),
        "maxBins": maxBins_options[best["maxBins"]],
        "minInstancesPerNode": minInstancesPerNode_options[best["minInstancesPerNode"]],
        "subsamplingRate": float(best["subsamplingRate"]),
        "featureSubsetStrategy": featureSubsetStrategy_options[best["featureSubsetStrategy"]]
    }
    
    logger.info(f"Best hyperparameters: {best_params}")
    
    # Analyze trials to understand feature importance trends
    logger.info("Analyzing feature importance trends across trials:")
    if trials.trials:
        # Extract metrics from valid trials
        valid_trials = [t for t in trials.trials if 'result' in t and isinstance(t['result'], dict)]
        
        if valid_trials:
            # Extract various metrics
            rmses = [t['result'].get('eval_rmse', float('inf')) for t in valid_trials]
            max_imps = [t['result'].get('max_feature_importance', 0) for t in valid_trials]
            pm10_imps = [t['result'].get('pm10_avg_importance', 0) for t in valid_trials]
            
            # Count models without 3h_pm10_avg
            models_without_pm10_avg = sum(1 for imp in pm10_imps if imp == 0)
            
            # Log statistics
            logger.info(f"  - RMSE range: {min(rmses):.4f} - {max(rmses):.4f}")
            logger.info(f"  - Max feature importance range: {min(max_imps):.4f} - {max(max_imps):.4f}")
            logger.info(f"  - 3h_pm10_avg importance range: {min(pm10_imps):.4f} - {max(pm10_imps):.4f}")
            logger.info(f"  - Models without 3h_pm10_avg: {models_without_pm10_avg}/{len(valid_trials)}")
            
            # Find best RMSE model and report if it uses 3h_pm10_avg
            best_rmse_idx = np.argmin(rmses)
            best_rmse = rmses[best_rmse_idx]
            best_rmse_uses_pm10 = pm10_imps[best_rmse_idx] > 0
            logger.info(f"  - Best RMSE model: {best_rmse:.4f} (uses 3h_pm10_avg: {best_rmse_uses_pm10})")
    
    # Save the best model if one was found
    if best_model is not None:
        logger.info(f"Retraining best model on full train set with best hyperparameters...")

        # ⚡ Rebuild best GBT model for full data training
        best_gbt = GBTRegressor(
            labelCol="pm10",
            featuresCol="features",
            maxDepth=int(best_params["maxDepth"]),
            maxIter=int(best_params["maxIter"]),
            stepSize=float(best_params["stepSize"]),
            maxBins=int(best_params["maxBins"]),
            minInstancesPerNode=int(best_params["minInstancesPerNode"]),
            subsamplingRate=float(best_params["subsamplingRate"]),
            featureSubsetStrategy=best_params["featureSubsetStrategy"],
            seed=MODEL_PARAMS.get("seed", 42)
        )

        full_pipeline = Pipeline(stages=[assembler_stage, scaler_stage, best_gbt])

        # Retrain fully
        full_model = full_pipeline.fit(train_df)

        # Save it
        model_path = save_model(full_model, model_name="pm10_gbt_best_hyperopt")
        logger.info(f"Full retrained best model saved to: {model_path}")
        
        # Return this full model now
        return best_params, trials, full_model, best_metrics

    else:
        logger.warning("No suitable model found")
        return best_params, trials, None, None


def plot_model_comparison(model, res_model, test_df, corrected_df, base_metrics, resid_metrics, feature_cols=None):
    """
    Create and save visualization comparing base and residual-stacked models
    using real model predictions on test_df.
    """

    logger.info("Generating prediction plots...")

    # Step 1: Prepare data
    # Transform test data with base model - prediction column will be named "prediction"
    base_preds = model.transform(test_df).select("datetime", "pm10", "prediction").toPandas()
    
    # Use the final_prediction column from corrected dataframe
    corrected_preds = corrected_df.select("datetime", "pm10", "final_prediction").toPandas()

    base_preds['datetime'] = pd.to_datetime(base_preds['datetime'])
    corrected_preds['datetime'] = pd.to_datetime(corrected_preds['datetime'])

    # Merge datasets on datetime
    merged = base_preds.merge(corrected_preds[['datetime', 'final_prediction']], on='datetime', how='inner')
    merged = merged.sort_values('datetime')

    # Step 2: Prepare residual predictions separately if needed
    try:
        residual_preds = res_model.transform(test_df).select("datetime", "residual_prediction").toPandas()
        residual_preds['datetime'] = pd.to_datetime(residual_preds['datetime'])
        # Could merge this with merged dataframe if needed for additional plots
    except Exception as e:
        logger.warning(f"Could not extract residual predictions: {e}")
        residual_preds = None

    # Step 3: Extract feature importance if possible
    feature_importance = {}
    try:
        # Check if the model is a PipelineModel and get the GBT stage
        if isinstance(model, PipelineModel):
            # The GBT is likely the last stage
            gbt = model.stages[-1]
        else:
            gbt = model  # Use the model directly if it's not a pipeline
            
        importances = gbt.featureImportances.toArray()
        
        if feature_cols and len(feature_cols) == len(importances):
            feature_importance = {feature_cols[i]: float(importances[i]) for i in range(len(importances))}
        else:
            feature_importance = {f"feature_{i}": float(importances[i]) for i in range(len(importances))}
    except Exception as e:
        logger.warning(f"Feature importance extraction error: {e}")

    # Step 4: Create the figure with GridSpec
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)

    # --- 1. Performance Metrics ---
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['RMSE', 'MAE', 'R²']
    base_values = [base_metrics['rmse'], base_metrics['mae'], base_metrics['r2']]
    resid_values = [resid_metrics['rmse'], resid_metrics['mae'], resid_metrics['r2']]
    x = range(len(metrics))

    ax1.bar([i - 0.2 for i in x], base_values, width=0.4, label="Base Model", color="#6A5ACD")
    ax1.bar([i + 0.2 for i in x], resid_values, width=0.4, label="Residual-Corrected", color="#20B2AA")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_title("Model Metrics Comparison")
    ax1.legend()
    ax1.grid(True)

    # --- 2. Time Series ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(merged['datetime'], merged['pm10'], label="Actual PM10", color="black", linewidth=2)
    ax2.plot(merged['datetime'], merged['prediction'], label="Base GBT", linestyle="--", color="blue")
    ax2.plot(merged['datetime'], merged['final_prediction'], label="Residual-Corrected", linestyle="-.", color="green")
    ax2.set_title("Predictions Over Time")
    ax2.legend()
    ax2.grid(True)

    # --- 3. Feature Importance ---
    feature_cols = [c for c in FEATURE_COLUMNS if c != "pm10"]
    ax3 = fig.add_subplot(gs[1, 0])
    if feature_importance:
        # Sort features by importance and get top 20 (or fewer if less available)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_n = min(20, len(sorted_features))
        names, scores = zip(*sorted_features[:top_n])
        
        # Plot horizontal bars (reversed to show most important at top)
        y_pos = range(len(names))
        ax3.barh(y_pos, scores, color="#6A5ACD")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names)
        ax3.invert_yaxis()  # Most important at the top
        ax3.set_title(f"Feature Importance (Top {top_n})")
    else:
        ax3.text(0.5, 0.5, "Feature importance not available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)
    ax3.grid(True)

    # --- 4. Residual Distribution ---
    ax4 = fig.add_subplot(gs[1, 1])
    base_errors = merged['pm10'] - merged['prediction']
    corrected_errors = merged['pm10'] - merged['final_prediction']

    bins = np.linspace(min(base_errors.min(), corrected_errors.min()), 
                       max(base_errors.max(), corrected_errors.max()), 
                       50)
    ax4.hist(base_errors, bins=bins, alpha=0.5, label="Base GBT Errors", color="#6A5ACD")
    ax4.hist(corrected_errors, bins=bins, alpha=0.5, label="Residual Corrected Errors", color="#20B2AA")
    ax4.axvline(0, linestyle="--", color="black")
    ax4.set_title("Error Distribution")
    ax4.legend()
    ax4.grid(True)

    plt.suptitle("PM10 Model Comparison", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = f"pm10_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Model comparison plot saved to {output_path}")
    
    # Show the plot if in an interactive environment
    plt.show()