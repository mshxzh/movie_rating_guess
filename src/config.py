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
MLFLOW_TRACKING_URI = "http://ec2-34-253-187-162.eu-west-1.compute.amazonaws.com:5000" 
MLFLOW_EXPERIMENT_NAME = "movie-rating-experiments"
MLFLOW_REGISTRY_URI = MLFLOW_TRACKING_URI  # For model registry

# Model Registry configuration
MLFLOW_MODEL_NAME = "movie_review_rating_trigrams_lightgbm"  # Registered model name
MLFLOW_MODEL_STAGE = "None"  # Stage: None, "Staging", "Production", or "Archived"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering constants
COLUMNS_TO_KEEP = ['release_date', 'title', 'overview', 'original_language', 'genre']
TARGET_COLUMN = 'vote_average'

# Production pipeline configuration
# The best model is saved as a combined pipeline (preprocessing + model)
PRODUCTION_PIPELINE_NAME = "production_pipeline_v1"
PRODUCTION_PIPELINE_PATH = MODELS_DIR / f"{PRODUCTION_PIPELINE_NAME}.joblib"

# Best model configuration (from experiments)
# Updated based on actual best performing model from experiments
BEST_MODEL_CONFIG = {
    "pipeline_version": "v4_trigrams",  # Best pipeline from experiments
    "pipeline_name": "Trigrams + LightGBM",
    "model_name": "LightGBM",
    # Preprocessing parameters (best performing from experiments)
    "preprocessing": {
        "overview_max_features": 50,
        "title_max_features": 30,
        "overview_ngram_range": (1, 3),  # Unigrams + Bigrams + Trigrams
        "title_ngram_range": (1, 3),  # Unigrams + Bigrams + Trigrams
        "language_threshold": 0.01
    },
    # Model hyperparameters
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

