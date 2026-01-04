"""
Training script for movie rating prediction model.
Integrates with MLflow for experiment tracking and model registry.

Usage:
    python src/train.py --pipeline-version v1 --register-model
"""

import argparse
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DIR,
    MODELS_DIR,
    BEST_MODEL_CONFIG,
    RANDOM_STATE
)


def load_data(pipeline_version="v1"):
    """Load transformed training and test data."""
    print(f"\n{'='*70}")
    print(f"Loading {pipeline_version} transformed data...")
    print(f"{'='*70}\n")
    
    X_train = pd.read_csv(PROCESSED_DIR / f"X_train_transformed_{pipeline_version}.csv")
    X_test = pd.read_csv(PROCESSED_DIR / f"X_test_transformed_{pipeline_version}.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    
    print(f"Data loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test, hyperparameters=None):
    """Train a LightGBM model with given hyperparameters."""
    
    # Use best config if no hyperparameters provided
    if hyperparameters is None:
        hyperparameters = BEST_MODEL_CONFIG["hyperparameters"]
    
    print(f"\n{'='*70}")
    print("Training LightGBM Model")
    print(f"{'='*70}\n")
    
    # Create model
    model = lgb.LGBMRegressor(**hyperparameters)
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    metrics = {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "cv_rmse": cv_rmse,
        "cv_std": cv_std,
        "training_time": training_time
    }
    
    print(f"Training complete!")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  CV RMSE: {cv_rmse:.4f} ± {cv_std:.4f}")
    
    return model, metrics


def log_to_mlflow(model, metrics, hyperparameters, pipeline_version, register_model=False):
    """Log model and metrics to MLflow."""
    
    print(f"\n{'='*70}")
    print("Logging to MLflow")
    print(f"{'='*70}\n")
    
    # Set tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"production_{pipeline_version}_{int(time.time())}"):
        # Log parameters
        mlflow.log_param("pipeline_version", pipeline_version)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("environment", "production")
        
        for param, value in hyperparameters.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"movie_rating_model_{pipeline_version}" if register_model else None
        )
        
        run_id = mlflow.active_run().info.run_id
        
        print(f"Logged to MLflow")
        print(f"  Run ID: {run_id}")
        print(f"  Experiment: {MLFLOW_EXPERIMENT_NAME}")
        if register_model:
            print(f"  Registered as: movie_rating_model_{pipeline_version}")
        
        return run_id


def save_model_locally(model, pipeline_version):
    """Save model to local models directory as backup."""
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_path = MODELS_DIR / f"production_model_{pipeline_version}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved locally: {model_path}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train movie rating prediction model")
    parser.add_argument(
        "--pipeline-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Feature pipeline version (v1=TF-IDF, v2=SentenceTransformer)"
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in MLflow Model Registry"
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default=None,
        help="Path to JSON file with custom hyperparameters (optional)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("MOVIE RATING PREDICTION - TRAINING")
    print(f"{'='*70}")
    print(f"Pipeline Version: {args.pipeline_version}")
    print(f"Register Model: {args.register_model}")
    print(f"{'='*70}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(args.pipeline_version)
    
    # Get hyperparameters
    hyperparameters = BEST_MODEL_CONFIG["hyperparameters"]
    if args.hyperparameters:
        import json
        with open(args.hyperparameters, 'r') as f:
            hyperparameters = json.load(f)
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_test, y_test, hyperparameters)
    
    # Log to MLflow
    run_id = log_to_mlflow(
        model, metrics, hyperparameters, 
        args.pipeline_version, args.register_model
    )
    
    # Save locally as backup
    model_path = save_model_locally(model, args.pipeline_version)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model ready for production")
    print(f"   Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"   Test MAE: {metrics['test_mae']:.4f}")
    print(f"   MLflow Run ID: {run_id}")
    print(f"   Local backup: {model_path}")
    print(f"\nTo use this model:")
    print(f"   python src/predict.py --run-id {run_id}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

