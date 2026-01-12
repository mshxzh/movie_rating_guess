"""
Production predictor for movie rating model.

- Loads **local** production artifact:
  - `models/production_pipeline_v1.joblib` (combined preprocessing + model)
- Predicts for **one** movie input at a time (no batch).

IMPORTANT: All custom transformer classes must be imported before loading the pipeline.
This ensures joblib can find the class definitions when unpickling.
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import from src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
from dataclasses import dataclass

import pandas as pd
from src.schemas import MovieInput

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

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def pipeline_path(self) -> Path:
        return self.project_root / "models" / f"{self.pipeline_name}.joblib"


class ProductionPredictor:
    """Loads local production pipeline and predicts for a single movie input."""

    def __init__(self, artifacts: ProdArtifacts | None = None):
        self.artifacts = artifacts or ProdArtifacts()
        self.pipeline = None
        self.loaded_from: str = "Local file"  # Track where model was loaded from

    def load(self) -> None:
        """Load the combined production pipeline (preprocessing + model) from local."""
        if not self.artifacts.pipeline_path.exists():
            raise FileNotFoundError(f"Missing production pipeline: {self.artifacts.pipeline_path}")

        self.pipeline = joblib.load(self.artifacts.pipeline_path)
        self.loaded_from = f"Local file ({self.artifacts.pipeline_path})"

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict_one(self, movie: MovieInput) -> float:
        """Predict a rating for a single movie."""
        if not self.is_loaded():
            raise RuntimeError("ProductionPredictor not loaded. Call `.load()` first.")

        raw_df = pd.DataFrame([movie.model_dump()])
        pred = self.pipeline.predict(raw_df)
        return float(pred[0])
