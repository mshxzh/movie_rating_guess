"""
Configuration file for movie rating prediction project.
Centralized configuration for training, prediction, and MLflow.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "movies.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# MLflow configuration
MLFLOW_TRACKING_URI = f"sqlite:///{NOTEBOOKS_DIR}/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "movie-rating-production"
MLFLOW_REGISTRY_URI = MLFLOW_TRACKING_URI  # For model registry

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering
PIPELINE_VERSION = "v1"  # Options: "v1" (TF-IDF), "v2" (Sentence Transformers)
LANGUAGE_THRESHOLD = 0.01
OVERVIEW_MAX_FEATURES = 50
TITLE_MAX_FEATURES = 30

# Best model configuration (from experiments)
BEST_MODEL_CONFIG = {
    "model_name": "LightGBM",
    "dataset_version": "v1",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }
}

# Columns to keep for feature engineering
COLUMNS_TO_KEEP = ['release_date', 'title', 'overview', 'original_language', 'genre']
TARGET_COLUMN = 'vote_average'

