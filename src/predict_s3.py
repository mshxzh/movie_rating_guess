"""
Production predictor for movie rating model.

- Loads model from MLflow Model Registry (S3) or local fallback
- Predicts for **one** movie input at a time (no batch).

IMPORTANT: All custom transformer classes must be imported before loading the pipeline.
This ensures joblib can find the class definitions when unpickling.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path so we can import from src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import mlflow
import mlflow.sklearn
from src.schemas import MovieInput
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_STAGE,
    PRODUCTION_PIPELINE_NAME,
    MODELS_DIR,
)

# CRITICAL: Import all custom transformer classes before loading the pipeline
# This ensures joblib can find them when unpickling the saved pipeline
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


@dataclass(frozen=True)
class ProdArtifacts:
    stage: str = "prod"
    pipeline_name: str = "production_pipeline_v1"
    use_mlflow_registry: bool = True  # Load from MLflow registry by default
    model_name: Optional[str] = None  # MLflow model name (from config if None)
    model_stage: Optional[str] = None  # MLflow model stage (from config if None)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def pipeline_path(self) -> Path:
        return self.project_root / "models" / f"{self.pipeline_name}.joblib"


class ProductionPredictor:
    """
    Loads production pipeline from MLflow Model Registry (S3) or local fallback.
    Predicts for a single movie input.
    """

    def __init__(self, artifacts: ProdArtifacts | None = None):
        self.artifacts = artifacts or ProdArtifacts()
        self.pipeline = None
        self.loaded_from: Optional[str] = None  # Track where model was loaded from

    def load_from_mlflow_registry(self) -> None:
        """Load the model from MLflow Model Registry (S3)."""
        print("Loading model from MLflow Model Registry...")
        print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"  Model Name: {self.artifacts.model_name or MLFLOW_MODEL_NAME}")
        print(f"  Stage: {self.artifacts.model_stage or MLFLOW_MODEL_STAGE}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Build model URI
        model_name = self.artifacts.model_name or MLFLOW_MODEL_NAME
        model_stage = self.artifacts.model_stage or MLFLOW_MODEL_STAGE
        
        if model_stage:
            model_uri = f"models:/{model_name}/{model_stage}"
        else:
            # Load latest version if no stage specified
            model_uri = f"models:/{model_name}/latest"
        
        print(f"  Model URI: {model_uri}")
        
        try:
            # First, try to verify the model exists in the registry
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            print(f"  Checking if model '{model_name}' exists in registry...")
            try:
                if model_stage:
                    # Get model version by stage
                    latest_versions = client.get_latest_versions(model_name, stages=[model_stage])
                    if not latest_versions:
                        raise ValueError(
                            f"Model '{model_name}' not found in stage '{model_stage}'. "
                            f"Available stages may be different. Try checking the MLflow UI."
                        )
                    version_info = latest_versions[0]
                    print(f"Found model version {version_info.version} in stage '{model_stage}'")
                else:
                    # Get latest version
                    latest_versions = client.get_latest_versions(model_name)
                    if not latest_versions:
                        raise ValueError(f"Model '{model_name}' not found in registry.")
                    version_info = latest_versions[0]
                    print(f"Found latest model version {version_info.version}")
            except Exception as registry_error:
                print(f"Registry check failed: {registry_error}")
                raise
            
            # Load model from registry
            # This downloads from S3 if artifacts are stored there
            print(f"  Downloading model artifacts from S3...")
            self.pipeline = mlflow.sklearn.load_model(model_uri)
            self.loaded_from = f"MLflow Registry ({model_uri})"
            print(f"Model loaded from MLflow Registry")
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            raise

    def load(self) -> None:
        """
        Load the combined production pipeline.
        Tries MLflow Registry first, falls back to local file if registry fails.
        """
        if self.artifacts.use_mlflow_registry:
            try:
                self.load_from_mlflow_registry()
                return
            except Exception as e:
                print(f"\nMLflow Registry load failed: {e}")

    def is_loaded(self) -> bool:
        return self.pipeline is not None
    
    @staticmethod
    def list_available_models() -> dict:
        """
        List all available models in the MLflow Model Registry.
        Useful for debugging to see what models/stages are available.
        
        Returns:
            dict: Dictionary with model names and their available stages/versions
        """
        try:
            from mlflow.tracking import MlflowClient
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            # Get all registered models
            registered_models = client.search_registered_models()
            
            result = {}
            for model in registered_models:
                model_name = model.name
                versions = []
                for version in model.latest_versions:
                    versions.append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status
                    })
                result[model_name] = versions
            
            return result
        except Exception as e:
            return {"error": str(e)}

    def predict_one(self, movie: MovieInput) -> float:
        """Predict a rating for a single movie."""
        if not self.is_loaded():
            raise RuntimeError("ProductionPredictor not loaded. Call `.load()` first.")

        raw_df = pd.DataFrame([movie.model_dump()])
        pred = self.pipeline.predict(raw_df)
        return float(pred[0])
