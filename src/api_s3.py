"""
FastAPI application for movie rating prediction with S3/MLflow Model Registry support.

This API loads models from MLflow Model Registry (S3) with automatic fallback to local files.

Usage:
    # Start the API server
    uvicorn src.api_s3:app --reload
    
    # Or with specific host/port
    uvicorn src.api_s3:app --host 0.0.0.0 --port 8000
    
    # Access docs at: http://localhost:8000/docs
    
Environment Variables (for S3 access):
    - AWS_ACCESS_KEY_ID: AWS access key (if not using IAM role)
    - AWS_SECRET_ACCESS_KEY: AWS secret key (if not using IAM role)
    - AWS_DEFAULT_REGION: AWS region (defaults to us-east-1)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# CRITICAL: Import custom transformers BEFORE importing predictor
# This ensures they're available when the pipeline is loaded
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

from src.predict_s3 import ProductionPredictor, ProdArtifacts
from src.schemas import (
    MovieInput,
    PredictionOutput,
)
from src.schemas import HealthResponse


# Initialize FastAPI app
app = FastAPI(
    title="Movie Rating Prediction API (S3/MLflow)",
    description="Predict movie ratings using models loaded from MLflow Model Registry (S3) with local fallback.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance + load state
predictor: Optional[ProductionPredictor] = None
_load_task: Optional[asyncio.Task] = None
_load_error: Optional[str] = None
_load_complete: bool = False


@app.on_event("startup")
async def startup_event():
    """
    Initialize the predictor on startup.
    Loads model from MLflow Model Registry (S3) or local fallback in the background.
    
    The predictor will:
    1. First attempt to load from MLflow Model Registry (S3)
    2. If that fails, fall back to local file
    """
    global predictor, _load_task, _load_error, _load_complete
    
    print("\n" + "="*70)
    print("Movie Rating Prediction API (S3/MLflow) Starting...")
    print("="*70)
    
    # Import config to show what we're using
    from src.config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE
    
    print(f"MLflow Configuration:")
    print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  Model Name: {MLFLOW_MODEL_NAME}")
    print(f"  Model Stage: {MLFLOW_MODEL_STAGE}")
    
    # Load from MLflow Registry by default, with local fallback
    predictor = ProductionPredictor(
        artifacts=ProdArtifacts(
            stage="prod",
            use_mlflow_registry=True,  # Try MLflow Registry first
            model_name=None,  # Use from config
            model_stage=None  # Use from config
        )
    )
    
    print(f"\nLoading strategy: MLflow Registry (S3) → Local fallback")

    async def _load():
        global _load_error, _load_complete
        try:
            print("\nStarting model loading...")
            await asyncio.to_thread(predictor.load)
            _load_complete = True
            print("\nModel loaded successfully!")
            print(f"  Loaded from: {predictor.loaded_from}")
            print(f"  Pipeline type: {type(predictor.pipeline)}")
        except Exception as e:
            _load_error = str(e)
            _load_complete = True  # Mark as complete even on error
            print(f"\nModel loading failed: {e}")
            import traceback
            traceback.print_exc()

    _load_task = asyncio.create_task(_load())
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\nShutting down Movie Rating Prediction API...")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if predictor is None:
        return HealthResponse(status="unhealthy", model_loaded=False, pipeline_version=None, model_info=None)
    
    if _load_error:
        return HealthResponse(
            status="error",
            model_loaded=False,
            pipeline_version="v1_original",
            model_info={"stage": "prod", "error": _load_error},
        )
    
    # Check if loading is complete and model is actually loaded
    loaded = _load_complete and predictor.is_loaded()
    
    if not _load_complete:
        status = "loading"
    elif not loaded:
        status = "error"
    else:
        status = "healthy"
    
    model_info = {"stage": "prod"}
    if predictor and predictor.loaded_from:
        model_info["loaded_from"] = predictor.loaded_from
    
    return HealthResponse(
        status=status,
        model_loaded=loaded,
        pipeline_version="v1_original",
        model_info=model_info,
    )


@app.get("/models", tags=["Debug"])
async def list_models():
    """
    List all available models in MLflow Model Registry.
    Useful for debugging to see what models/stages are available.
    """
    try:
        models = ProductionPredictor.list_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_single(movie: MovieInput):
    """
    Predict rating for a single movie.
    
    Example request body:
    ```json
    {
        "release_date": "2023-07-21",
        "title": "Oppenheimer",
        "overview": "The story of American scientist J. Robert Oppenheimer...",
        "genre": "Drama, History, Thriller",
        "original_language": "en"
    }
    ```
    """
    # Check if predictor exists and is fully loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized. Please check API logs."
        )
    
    if not _load_complete:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Please wait and check /health endpoint."
        )
    
    if _load_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model loading failed: {_load_error}"
        )
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health endpoint for details."
        )
    
    try:
        prediction = predictor.predict_one(movie)
        
        return PredictionOutput(
            title=movie.title,
            predicted_rating=round(prediction, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


def main():
    """Run the API server with S3/MLflow model loading."""
    uvicorn.run(
        "src.api_s3:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()

