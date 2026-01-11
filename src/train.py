"""
Training script for movie rating prediction model.
Trains the best model configuration and registers it as a new version in MLflow Model Registry.

Creates a single pipeline that includes preprocessing and model training.

Usage:
    python src/train.py
    python src/train.py --model-name movie_rating_model
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import from src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import joblib
import time
import importlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# Force reload the transformers module to pick up any changes (removes cached version)
if 'src.transformers' in sys.modules:
    importlib.reload(sys.modules['src.transformers'])
if 'src.config' in sys.modules:
    importlib.reload(sys.modules['src.config'])
if 'src' in sys.modules:
    importlib.reload(sys.modules['src'])

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    RAW_DATA_PATH,
    BEST_MODEL_CONFIG,
    PRODUCTION_PIPELINE_NAME,
    RANDOM_STATE,
    TEST_SIZE,
    COLUMNS_TO_KEEP,
    TARGET_COLUMN,
)

# Import custom transformers
from src.transformers import (
    ColumnSelector,
    DataTypeFixer,
    YearBinning,
    GenreMultiLabelEncoder,
    LanguageGrouper,
    LightweightTextEmbedder,
    SelectiveStandardScaler,
    CategoricalOneHotEncoder,
)


def build_full_pipeline():
    """
    Build the complete pipeline (preprocessing + model) in one go.
    
    Returns:
        sklearn Pipeline with preprocessing and model
    """
    config = BEST_MODEL_CONFIG
    pipeline_name = config.get("pipeline_name", "Unknown")
    
    # Get preprocessing parameters from config
    preprocessing_params = config.get("preprocessing", {})
    overview_max_features = preprocessing_params.get("overview_max_features", 50)
    title_max_features = preprocessing_params.get("title_max_features", 30)
    overview_ngram_range = preprocessing_params.get("overview_ngram_range", (1, 3))
    title_ngram_range = preprocessing_params.get("title_ngram_range", (1, 3))
    language_threshold = preprocessing_params.get("language_threshold", 0.01)
    
    # Get model hyperparameters
    hyperparameters = BEST_MODEL_CONFIG["hyperparameters"].copy()
    
    # Build complete pipeline (preprocessing + model)
    full_pipeline = SklearnPipeline([
        # Preprocessing steps
        ('select_columns', ColumnSelector(columns=COLUMNS_TO_KEEP)),
        ('fix_dtypes', DataTypeFixer()),
        ('bin_years', YearBinning()),
        ('encode_genres', GenreMultiLabelEncoder()),
        ('group_languages', LanguageGrouper(threshold=language_threshold)),
        ('embed_overview', LightweightTextEmbedder(
            column='overview',
            max_features=overview_max_features,
            ngram_range=overview_ngram_range,
            prefix='overview'
        )),
        ('embed_title', LightweightTextEmbedder(
            column='title',
            max_features=title_max_features,
            ngram_range=title_ngram_range,
            prefix='title'
        )),
        ('onehot_encode', CategoricalOneHotEncoder(columns=['year_bin', 'original_language'])),
        ('scale_features', SelectiveStandardScaler()),
        # Model
        ('model', lgb.LGBMRegressor(**hyperparameters))
    ])
    
    return full_pipeline, pipeline_name


def load_raw_data():
    """Load and prepare raw data."""
    print(f"\n{'='*70}")
    print("Loading Raw Data")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = [col.lower() for col in df.columns]
    
    # Clean data
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df = df[~df[TARGET_COLUMN].isna()]
    df = df[df[TARGET_COLUMN] != 0]
    
    # Split features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Data loaded:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train: {len(X_train)}")
    print(f"  Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate_pipeline(full_pipeline, X_train, y_train, X_test, y_test):
    """
    Train the full pipeline and evaluate it.
    
    Args:
        full_pipeline: Complete pipeline (preprocessing + model)
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        Tuple of (metrics dictionary, n_features, pipeline_fit_time, training_time)
    """
    print(f"\n{'='*70}")
    print("Training Full Pipeline (Preprocessing + Model)")
    print(f"{'='*70}\n")
    
    # Get feature count after preprocessing (before training)
    preprocessing_steps = full_pipeline.steps[:-1]  # All steps except the model
    temp_pipeline = SklearnPipeline(preprocessing_steps)
    X_train_transformed = temp_pipeline.fit_transform(X_train)
    n_features = X_train_transformed.shape[1]
    
    # Measure pipeline fit time (preprocessing fitting)
    pipeline_fit_start = time.time()
    temp_pipeline.fit(X_train)
    pipeline_fit_time = time.time() - pipeline_fit_start
    
    # Train the entire pipeline at once
    train_start = time.time()
    full_pipeline.fit(X_train, y_train)
    training_time = time.time() - train_start
    
    print(f"Pipeline training complete in {training_time:.2f}s")
    print(f"  Features: {X_train.shape[1]} → {n_features}")
    
    # Evaluate on training set
    y_train_pred = full_pipeline.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = full_pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation on the full pipeline
    print("Performing 5-fold cross-validation on full pipeline...")
    cv_scores = cross_val_score(
        full_pipeline, X_train, y_train,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    cv_rmse = -cv_scores
    cv_rmse_mean = cv_rmse.mean()
    cv_rmse_std = cv_rmse.std()
    
    metrics = {
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "training_time": training_time,
        "pipeline_fit_time": pipeline_fit_time
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
    
    return metrics, n_features, pipeline_fit_time, training_time


def save_pipeline_locally(full_pipeline, pipeline_name):
    """Save the full pipeline to local models directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    pipeline_path = MODELS_DIR / f"{pipeline_name}.joblib"
    joblib.dump(full_pipeline, pipeline_path)
    
    print(f"\nv Full pipeline saved locally: {pipeline_path}")
    return pipeline_path


def register_model_version(full_pipeline, metrics, hyperparameters, pipeline_info, model_name, n_features):
    """
    Register the model as a new version in MLflow Model Registry.
    Logs the same parameters as the notebook for consistency.
    
    Args:
        full_pipeline: Complete pipeline (preprocessing + model)
        metrics: Dictionary of evaluation metrics
        hyperparameters: Model hyperparameters
        pipeline_info: Dictionary with pipeline version and name
        model_name: Name for the registered model in MLflow
        n_features: Number of features after preprocessing
    
    Returns:
        Model version info
    """
    print(f"\n{'='*70}")
    print("Registering Model in MLflow Model Registry")
    print(f"{'='*70}\n")
    
    # Set tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"production_{pipeline_info['pipeline_version']}_{int(time.time())}"):
        # Log parameters
        mlflow.log_param("pipeline_version", pipeline_info['pipeline_version'])
        mlflow.log_param("pipeline_name", pipeline_info['pipeline_name'])
        mlflow.log_param("model_name", BEST_MODEL_CONFIG['model_name'])  # "LightGBM"
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_features", n_features)
        
        # Log TF-IDF configuration
        preprocessing_params = BEST_MODEL_CONFIG.get("preprocessing", {})
        mlflow.log_param("overview_max_features", preprocessing_params.get("overview_max_features", 50))
        mlflow.log_param("title_max_features", preprocessing_params.get("title_max_features", 30))
        mlflow.log_param("overview_ngram_range", str(preprocessing_params.get("overview_ngram_range", (1, 3))))
        mlflow.log_param("title_ngram_range", str(preprocessing_params.get("title_ngram_range", (1, 3))))
        
        # Log model hyperparameters
        for param, value in hyperparameters.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log the full pipeline
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model"
        )
        
        run_id = mlflow.active_run().info.run_id
        
        # Register model as a new version
        model_version = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name
        )
        
        print(f"v Model registered successfully!")
        print(f"  Run ID: {run_id}")
        print(f"  Model Name: {model_name}")
        print(f"  Model Version: {model_version.version}")
        print(f"  Stage: {model_version.current_stage}")
        print(f"  Experiment: {MLFLOW_EXPERIMENT_NAME}")
        
        return model_version


def main():
    parser = argparse.ArgumentParser(
        description="Train the best model and register it as a new version in MLflow"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="movie_review_rating_trigrams_lightgbm",
        help="Name for the registered model in MLflow Model Registry"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("MOVIE RATING PREDICTION - TRAINING BEST MODEL")
    print(f"{'='*70}")
    print(f"Model Name: {args.model_name}")
    print(f"Best Config: {BEST_MODEL_CONFIG['pipeline_version']} ({BEST_MODEL_CONFIG['pipeline_name']})")
    print(f"{'='*70}")
    
    # Load raw data
    X_train, X_test, y_train, y_test = load_raw_data()
    
    # Build complete pipeline (preprocessing + model)
    print(f"\n{'='*70}")
    print("Building Complete Pipeline (Preprocessing + Model)")
    print(f"{'='*70}\n")
    
    pipeline_version = BEST_MODEL_CONFIG['pipeline_version']
    full_pipeline, pipeline_name = build_full_pipeline()
    
    print(f"Pipeline: {pipeline_name} ({pipeline_version})")
    print(f"Pipeline steps: {len(full_pipeline.steps)}")
    print("  - Preprocessing steps: 9")
    print("  - Model: LightGBM")
    
    # Update hyperparameters if custom file provided
    hyperparameters = BEST_MODEL_CONFIG["hyperparameters"].copy()
    
    # Train and evaluate the full pipeline
    metrics, n_features, pipeline_fit_time, training_time = train_and_evaluate_pipeline(
        full_pipeline, X_train, y_train, X_test, y_test
    )
    
    # Save locally
    pipeline_info = {
        'pipeline_version': pipeline_version,
        'pipeline_name': pipeline_name
    }
    local_path = save_pipeline_locally(full_pipeline, PRODUCTION_PIPELINE_NAME)
    
    # Register in MLflow Model Registry
    model_version = register_model_version(
        full_pipeline,
        metrics,
        hyperparameters,
        pipeline_info,
        args.model_name,
        n_features
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"v Model trained and registered successfully")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Model Version: {model_version.version}")
    print(f"  Local Path: {local_path}")
    print(f"\nThe model is registered as '{args.model_name}' version {model_version.version}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
