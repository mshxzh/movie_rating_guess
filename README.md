# Movie Rating Prediction API

A machine learning system for predicting movie ratings based on metadata such as title, overview, genre, release date, and language. The project includes exploratory data analysis, feature engineering experiments, model training, and a production-ready FastAPI service.

## 📋 Table of Contents

1. [EDA and Data Description](#1-eda-and-data-description)
2. [ML Experiments and Feature Pipeline](#2-ml-experiments-and-feature-pipeline)
3. [Train / Predict Files](#3-train--predict-files)
4. [API](#4-api)
   - [Local Model API](#local-model-api)
   - [S3/MLflow Model Registry API](#s3mlflow-model-registry-api)
5. [Docker](#5-docker)

---

## 1. EDA and Data Description

### Data Source

The dataset is located in `data/raw/movies.csv` and contains movie metadata from TMDB (The Movie Database). Available at: https://huggingface.co/datasets/Pablinho/movies-dataset

### Data Structure

The dataset includes the following features:
- **release_date**: Date when the movie was released
- **title**: Movie title
- **overview**: Text description/synopsis of the movie
- **genre**: Comma-separated list of genres (e.g., "Drama, Action, Thriller")
- **original_language**: ISO 639-1 language code (e.g., "en", "fr", "es")
- **vote_average**: Target variable - average rating (0-10 scale)

### Exploratory Data Analysis

The EDA is performed in `notebooks/01_eda.ipynb` and includes:

- **Data Quality Checks**: Missing values, data types, duplicates
- **Target Variable Analysis**: Distribution of ratings, outliers
- **Feature Analysis**:
  - Release date trends over time
  - Genre distribution and popularity
  - Language distribution
  - Text feature statistics (title/overview length)
- **Correlation Analysis**: Relationships between features and target
- **Data Cleaning**: Handling missing values, outliers, invalid dates

### Data Preprocessing

- Removed movies with missing or zero ratings
- Standardized date formats
- Handled missing text fields (filled with empty strings)
- Language standardization (grouped rare languages)

---

## 2. ML Experiments and Feature Pipeline

### Experiment Notebook

All machine learning experiments are documented in `notebooks/02_experiments.ipynb`.

### Feature Engineering Pipelines

The project implements multiple feature engineering pipelines using custom sklearn transformers:

#### Pipeline Versions

1. **v1_original**: Baseline pipeline
   - TF-IDF on overview (50 features, unigrams)
   - TF-IDF on title (30 features, unigrams)
   - One-hot encoding for categorical features

2. **v2_bigrams**: Extended n-grams
   - TF-IDF on overview (50 features, unigrams + bigrams)
   - TF-IDF on title (30 features, unigrams + bigrams)

3. **v3_bigrams_more_features**: More features
   - TF-IDF on overview (100 features, unigrams + bigrams)
   - TF-IDF on title (50 features, unigrams + bigrams)

4. **v4_trigrams**: Best performing pipeline
   - TF-IDF on overview (50 features, unigrams + bigrams + trigrams)
   - TF-IDF on title (30 features, unigrams + bigrams + trigrams)

#### Custom Transformers

Located in `src/transformers.py`:

- **ColumnSelector**: Selects specific columns from DataFrame
- **DataTypeFixer**: Handles missing values and data type conversion
- **YearBinning**: Bins release years into meaningful categories
- **GenreMultiLabelEncoder**: Encodes multi-label genre strings
- **LanguageGrouper**: Groups rare languages (below threshold) into "other"
- **LightweightTextEmbedder**: TF-IDF vectorization for text features
- **SelectiveStandardScaler**: Standardizes only numeric features
- **CategoricalOneHotEncoder**: One-hot encodes categorical features

### Model Experiments

The notebook trains and evaluates multiple models:

- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-based**: Decision Tree, Random Forest, Gradient Boosting
- **Boosting**: XGBoost, LightGBM

### Best Model

**LightGBM with v4_trigrams pipeline** achieved the best performance:
- **Test RMSE**: ~0.85
- **Test MAE**: ~0.65
- **Test R²**: ~0.45

### Hyperparameter Tuning

The notebook includes hyperparameter optimization using:
- **RandomizedSearchCV**: Broad parameter search
- **GridSearchCV**: Conservative grid search
- **Optuna**: Bayesian optimization (most efficient)

---

## 3. Train / Predict Files

### Training Script: `src/train.py`

Trains the best model configuration and saves it locally and to MLflow Model Registry.

#### Usage

```bash
# Train with default model name
python src/train.py

# Train with custom model name
python src/train.py --model-name movie_rating_model_v2
```

#### What it does:

1. **Loads raw data** from `data/raw/movies.csv`
2. **Builds the complete pipeline** (preprocessing + LightGBM model)
3. **Trains the model** on the full training set
4. **Evaluates performance** (train/test metrics, cross-validation)
5. **Saves locally** as `models/production_pipeline_v1.joblib`
6. **Registers in MLflow** Model Registry (if MLflow server is accessible)

#### Output

- **Local model**: `models/production_pipeline_v1.joblib` (combined preprocessing + model)
- **MLflow registration**: New version in Model Registry with all metrics and parameters

### Prediction Script: `src/predict.py`

Loads the local production model and makes predictions.

#### Usage

```python
from src.predict import ProductionPredictor, ProdArtifacts

# Initialize predictor
predictor = ProductionPredictor(
    artifacts=ProdArtifacts(
        stage="prod",
        pipeline_name="production_pipeline_v1"
    )
)

# Load model
predictor.load()

# Make prediction
from src.schemas import MovieInput

movie = MovieInput(
    release_date="2023-07-21",
    title="Oppenheimer",
    overview="The story of American scientist J. Robert Oppenheimer...",
    genre="Drama, History, Thriller",
    original_language="en"
)

rating = predictor.predict_one(movie)
print(f"Predicted rating: {rating}")
```

#### Features

- Loads model from local file (`models/production_pipeline_v1.joblib`)
- Handles all preprocessing automatically
- Returns float prediction (0-10 scale)

---

## 4. API

The project provides two API implementations:

### Local Model API

**File**: `src/api.py`

**Purpose**: Production API that loads the model from a local file included in the Docker image.

#### Why Local Model Loading?

To avoid dependencies on external infrastructure (EC2 with MLflow), the production API loads the model directly from a local file baked into the Docker image. This provides:

- ✅ **No external dependencies**: No need for MLflow server or S3 access
- ✅ **Faster startup**: Model loads instantly from local file
- ✅ **Simpler deployment**: No AWS credentials or network configuration needed
- ✅ **More reliable**: No risk of network failures or service unavailability
- ✅ **Smaller image**: No MLflow/Boto3 dependencies (~300-500MB saved)

#### Usage

```bash
# Start the API server
uvicorn src.api:app --reload

# Or with specific host/port
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### Endpoints

- **GET `/health`**: Health check and model status
- **POST `/predict`**: Predict rating for a single movie
- **GET `/docs`**: Interactive API documentation (Swagger UI)
- **GET `/redoc`**: Alternative API documentation

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "release_date": "2023-07-21",
    "title": "Oppenheimer",
    "overview": "The story of American scientist J. Robert Oppenheimer...",
    "genre": "Drama, History, Thriller",
    "original_language": "en"
  }'
```

#### Response

```json
{
  "title": "Oppenheimer",
  "predicted_rating": 8.2
}
```

### S3/MLflow Model Registry API

**File**: `src/api_s3.py`

**Purpose**: Alternative API implementation that loads models from MLflow Model Registry (S3).

#### When to Use

- When you want to dynamically load different model versions
- When models are stored in S3 and managed via MLflow
- When you need model versioning and A/B testing capabilities

#### Requirements

- MLflow server running and accessible
- AWS credentials (IAM role or access keys)
- Model registered in MLflow Model Registry

#### Usage

```bash
# Set MLflow configuration
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
export MLFLOW_USERNAME="admin"  # If authentication enabled
export MLFLOW_PASSWORD="password"  # If authentication enabled

# Set AWS credentials (if not using IAM role)
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="eu-west-1"

# Start the API server
uvicorn src.api_s3:app --reload
```

#### Configuration

Update `src/config.py`:

```python
MLFLOW_TRACKING_URI = "http://your-mlflow-server:5000"
MLFLOW_MODEL_NAME = "movie_review_rating_trigrams_lightgbm"
MLFLOW_MODEL_STAGE = "Production"  # or "Staging", "None"
```

#### Features

- Loads model from MLflow Model Registry
- Automatic fallback to local file if registry unavailable
- Model versioning support
- Debug endpoint: `GET /models` - List available models in registry

---

## 5. Docker

### Overview

The Dockerfile creates an optimized production image with:
- Multi-stage build for minimal size
- Non-root user for security
- Model baked into the image (no external dependencies)
- Health checks for monitoring

### Build the Image

```bash
# Build the image
docker build -t movie-rating-guessr:latest .

# Build with specific tag
docker build -t movie-rating-guessr:v1.0.0 .
```

### Image Size

The optimized image is approximately **400-600MB** (down from 2.26GB) thanks to:
- Multi-stage build (removes build tools)
- No MLflow/Boto3 dependencies (local model loading)
- Python cache cleanup
- Minimal base image (`python:3.11.9-slim`)

### Run the Container

```bash
# Basic run
docker run -d \
  --name movie-rating-api \
  -p 8000:8000 \
  movie-rating-api:latest

# With environment variables
docker run -d \
  --name movie-rating-api \
  -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  movie-rating-api:latest
```

### Test the Container

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "release_date": "2023-07-21",
    "title": "Oppenheimer",
    "overview": "The story of American scientist J. Robert Oppenheimer...",
    "genre": "Drama, History, Thriller",
    "original_language": "en"
  }'

# View logs
docker logs -f movie-rating-api
```

### Dockerfile Structure

```dockerfile
# Stage 1: Builder
- Installs dependencies with build tools
- Creates virtual environment
- Cleans up Python cache

# Stage 2: Runtime
- Minimal base image
- Only runtime dependencies
- Non-root user (appuser)
- Model and code copied
- Health check configured
```

### Security Features

- ✅ **Non-root user**: Runs as `appuser` (UID 1000)
- ✅ **Minimal attack surface**: Only necessary dependencies
- ✅ **No secrets in image**: All credentials via environment variables
- ✅ **Health checks**: Automatic container health monitoring

### Production Deployment

#### AWS ECS / Fargate

```bash
# Tag for ECR
docker tag movie-rating-api:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/movie-rating-api:latest

# Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/movie-rating-api:latest
```

#### Google Cloud Run

```bash
# Tag for GCR
docker tag movie-rating-api:latest \
  gcr.io/PROJECT_ID/movie-rating-api:latest

# Push to GCR
docker push gcr.io/PROJECT_ID/movie-rating-api:latest
```

#### Azure Container Instances

```bash
# Tag for ACR
docker tag movie-rating-api:latest \
  myregistry.azurecr.io/movie-rating-api:latest

# Push to ACR
docker push myregistry.azurecr.io/movie-rating-api:latest
```

---

## 📦 Project Structure

```
movie_rating_guess/
├── data/
│   ├── raw/
│   │   └── movies.csv              # Raw dataset
│   └── processed/                  # Transformed datasets
├── models/
│   └── production_pipeline_v1.joblib  # Production model
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_experiments.ipynb        # ML experiments
├── src/
│   ├── api.py                      # Local model API (production)
│   ├── api_s3.py                   # S3/MLflow API (alternative)
│   ├── predict.py                  # Local model predictor
│   ├── predict_s3.py               # S3/MLflow predictor
│   ├── train.py                    # Training script
│   ├── config.py                   # Configuration
│   ├── schemas.py                  # Pydantic models
│   └── transformers.py             # Custom sklearn transformers
├── Dockerfile                       # Production Docker image
├── pyproject.toml                   # Dependencies
├── uv.lock                         # Locked dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Run EDA

```bash
# Open Jupyter notebook
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Run Experiments

```bash
# Open experiments notebook
jupyter notebook notebooks/02_experiments.ipynb
```

### 4. Train Model

```bash
# Train the best model
python src/train.py
```

### 5. Run API Locally

```bash
# Start the API
uvicorn src.api:app --reload

# Test it
curl http://localhost:8000/health
```

### 6. Build and Run Docker

```bash
# Build image
docker build -t movie-rating-api .

# Run container
docker run -d -p 8000:8000 movie-rating-api

# Test
curl http://localhost:8000/health
```

## 📊 Model Performance

**Best Model**: LightGBM with v4_trigrams pipeline

- **Test RMSE**: ~0.85
- **Test MAE**: ~0.65
- **Test R²**: ~0.45
- **Cross-Validation RMSE**: ~0.87 ± 0.05

## 🔧 Configuration

All configuration is centralized in `src/config.py`:

- **Data paths**: Raw and processed data locations
- **Model paths**: Production model location
- **MLflow settings**: Tracking URI, experiment name (for S3 API)
- **Feature engineering**: TF-IDF parameters, language threshold
- **Model hyperparameters**: Best performing configuration

## 📝 Dependencies

Core dependencies (for API):
- `pandas`, `numpy`, `scikit-learn`
- `lightgbm`
- `fastapi`, `uvicorn`, `pydantic`

Optional dependencies:
- `mlflow` (for training and S3 API)
- `jupyter`, `matplotlib`, `seaborn` (for notebooks)
- `pytest` (for testing)

See `pyproject.toml` for complete list.

## 📄 License

See `LICENSE` file for details.

## 🙏 Acknowledgments

- TMDB for the movie dataset
- MLflow for experiment tracking
- FastAPI for the API framework
