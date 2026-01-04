"""
Prediction script for movie rating model.
Can load models from MLflow or local files.

Usage:
    # Using MLflow run ID
    python src/predict.py --run-id abc123 --input data.csv
    
    # Using registered model
    python src/predict.py --model-name movie_rating_model_v1 --input data.csv
    
    # Using local model file
    python src/predict.py --model-path models/production_model_v1.pkl --input data.csv
    
    # Interactive prediction
    python src/predict.py --run-id abc123 --interactive
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Union

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MLFLOW_TRACKING_URI, MODELS_DIR, PROCESSED_DIR
from src.schemas import MovieInput, MovieBatchInput


class MovieRatingPredictor:
    """
    Movie rating predictor that loads models from MLflow or local files.
    """
    
    def __init__(self):
        self.model = None
        self.feature_pipeline = None
        self.pipeline_version = None
        
    def load_from_mlflow_run(self, run_id):
        """Load model from MLflow run ID."""
        print(f"\n{'='*70}")
        print(f"Loading model from MLflow run: {run_id}")
        print(f"{'='*70}\n")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        try:
            # Load model
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Get run info to determine pipeline version
            run = mlflow.get_run(run_id)
            self.pipeline_version = run.data.params.get("pipeline_version", "v1")
            
            print(f"Model loaded successfully")
            print(f"  Pipeline version: {self.pipeline_version}")
            print(f"  Test RMSE: {run.data.metrics.get('test_rmse', 'N/A')}")
            
            # Load feature pipeline
            self._load_feature_pipeline()
            
            return True
            
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return False
    
    def load_from_registry(self, model_name, version="latest"):
        """Load model from MLflow Model Registry."""
        print(f"\n{'='*70}")
        print(f"Loading model from registry: {model_name} (version: {version})")
        print(f"{'='*70}\n")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        try:
            if version == "latest":
                model_uri = f"models:/{model_name}/latest"
            else:
                model_uri = f"models:/{model_name}/{version}"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Extract pipeline version from model name
            if "v1" in model_name:
                self.pipeline_version = "v1"
            elif "v2" in model_name:
                self.pipeline_version = "v2"
            else:
                self.pipeline_version = "v1"  # default
            
            print(f"Model loaded from registry")
            print(f"  Pipeline version: {self.pipeline_version}")
            
            # Load feature pipeline
            self._load_feature_pipeline()
            
            return True
            
        except Exception as e:
            print(f"Error loading model from registry: {e}")
            return False
    
    def load_from_file(self, model_path, pipeline_version="v1"):
        """Load model from local file."""
        print(f"\n{'='*70}")
        print(f"Loading model from file: {model_path}")
        print(f"{'='*70}\n")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.pipeline_version = pipeline_version
            
            print(f"Model loaded from file")
            print(f"  Pipeline version: {self.pipeline_version}")
            
            # Load feature pipeline
            self._load_feature_pipeline()
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model from file: {e}")
            return False
    
    def _load_feature_pipeline(self):
        """Load feature engineering pipeline."""
        pipeline_path = MODELS_DIR.parent / "models" / f"feature_pipeline_{self.pipeline_version}_original.pkl"
        
        if self.pipeline_version == "v2":
            pipeline_path = MODELS_DIR.parent / "models" / f"feature_pipeline_{self.pipeline_version}_sentence_transformer.pkl"
        
        try:
            with open(pipeline_path, 'rb') as f:
                self.feature_pipeline = pickle.load(f)
            print(f"  Feature pipeline: {pipeline_path.name}")
        except Exception as e:
            print(f"  Warning: Could not load feature pipeline: {e}")
            print(f"  Make sure to provide pre-transformed data")
    
    def predict_from_raw(self, raw_data: Union[pd.DataFrame, MovieInput, List[MovieInput]]):
        """
        Make predictions from raw movie data (requires feature pipeline).
        
        Args:
            raw_data: Can be:
                - DataFrame with columns: release_date, title, overview, original_language, genre
                - Single MovieInput Pydantic model
                - List of MovieInput Pydantic models
        
        Returns:
            numpy array of predictions
        """
        if self.feature_pipeline is None:
            raise ValueError("Feature pipeline not loaded. Cannot transform raw data.")
        
        # Convert Pydantic models to DataFrame if necessary
        if isinstance(raw_data, MovieInput):
            raw_data = pd.DataFrame([raw_data.model_dump()])
        elif isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], MovieInput):
            raw_data = pd.DataFrame([movie.model_dump() for movie in raw_data])
        elif not isinstance(raw_data, pd.DataFrame):
            raise TypeError(
                "raw_data must be a DataFrame, MovieInput, or List[MovieInput]"
            )
        
        # Transform features
        X_transformed = self.feature_pipeline.transform(raw_data)
        
        # Predict
        predictions = self.model.predict(X_transformed)
        
        return predictions
    
    def predict_single(self, movie: MovieInput) -> float:
        """
        Make prediction for a single movie using Pydantic model.
        
        Args:
            movie: MovieInput Pydantic model
        
        Returns:
            Predicted rating as a float
        """
        predictions = self.predict_from_raw(movie)
        return float(predictions[0])
    
    def predict_batch_pydantic(self, movies: List[MovieInput]) -> List[float]:
        """
        Make predictions for multiple movies using Pydantic models.
        
        Args:
            movies: List of MovieInput Pydantic models
        
        Returns:
            List of predicted ratings
        """
        predictions = self.predict_from_raw(movies)
        return [float(pred) for pred in predictions]
    
    def predict_from_transformed(self, X_transformed):
        """
        Make predictions from already transformed data.
        
        Args:
            X_transformed: DataFrame or array with transformed features
        """
        predictions = self.model.predict(X_transformed)
        return predictions
    
    def predict_batch(self, input_file, output_file=None, is_raw=True):
        """
        Make predictions on a batch of data from CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (optional)
            is_raw: Whether input is raw data or transformed features
        """
        print(f"\n{'='*70}")
        print(f"Batch Prediction")
        print(f"{'='*70}\n")
        
        # Load data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples from {input_file}")
        
        # Make predictions
        if is_raw:
            predictions = self.predict_from_raw(df)
        else:
            predictions = self.predict_from_transformed(df)
        
        # Add predictions to dataframe
        df['predicted_rating'] = predictions
        
        # Save if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        
        # Show sample
        print(f"\nSample predictions:")
        if 'vote_average' in df.columns:
            print(df[['title', 'vote_average', 'predicted_rating']].head(10).to_string(index=False))
        else:
            print(df[['title', 'predicted_rating']].head(10).to_string(index=False))
        
        return df
    
    def predict_interactive(self):
        """Interactive prediction mode."""
        print(f"\n{'='*70}")
        print("INTERACTIVE PREDICTION MODE")
        print(f"{'='*70}\n")
        print("Enter movie details (or 'quit' to exit):\n")
        
        while True:
            try:
                print("\n" + "-"*70)
                title = input("Title: ").strip()
                if title.lower() == 'quit':
                    break
                
                release_date = input("Release Date (YYYY-MM-DD): ").strip()
                overview = input("Overview: ").strip()
                genre = input("Genre(s) - comma separated: ").strip()
                language = input("Original Language (e.g., en, fr): ").strip()
                
                # Create dataframe
                movie_data = pd.DataFrame([{
                    'release_date': release_date,
                    'title': title,
                    'overview': overview,
                    'genre': genre,
                    'original_language': language
                }])
                
                # Predict
                prediction = self.predict_from_raw(movie_data)[0]
                
                print(f"Predicted Rating: {prediction:.2f}/10")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Predict movie ratings")
    
    # Model loading options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--run-id", type=str, help="MLflow run ID")
    model_group.add_argument("--model-name", type=str, help="MLflow registered model name")
    model_group.add_argument("--model-path", type=str, help="Path to local model file")
    
    # Prediction options
    parser.add_argument("--input", type=str, help="Input CSV file for batch prediction")
    parser.add_argument("--output", type=str, help="Output CSV file for predictions")
    parser.add_argument("--interactive", action="store_true", help="Interactive prediction mode")
    parser.add_argument("--pipeline-version", type=str, default="v1", choices=["v1", "v2"],
                       help="Pipeline version (only needed with --model-path)")
    parser.add_argument("--is-transformed", action="store_true", 
                       help="Input data is already transformed")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MovieRatingPredictor()
    
    # Load model
    if args.run_id:
        success = predictor.load_from_mlflow_run(args.run_id)
    elif args.model_name:
        success = predictor.load_from_registry(args.model_name)
    elif args.model_path:
        success = predictor.load_from_file(args.model_path, args.pipeline_version)
    
    if not success:
        print("Failed to load model")
        return
    
    # Make predictions
    if args.interactive:
        predictor.predict_interactive()
    elif args.input:
        predictor.predict_batch(args.input, args.output, is_raw=not args.is_transformed)
    else:
        print("Please specify --input or --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()

