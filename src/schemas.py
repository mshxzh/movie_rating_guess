"""
Pydantic schemas for API request/response validation.
Defines the structure of movie data for predictions.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import date


class MovieInput(BaseModel):
    """
    Schema for a single movie input for prediction.
    
    All fields match the expected input for the feature pipeline.
    """
    release_date: str = Field(
        ...,
        description="Release date in YYYY-MM-DD format",
        examples=["2023-01-15", "2020-12-25"]
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Movie title",
        examples=["The Matrix", "Inception"]
    )
    overview: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Movie plot overview/description",
        examples=["A computer hacker learns about the true nature of reality..."]
    )
    genre: str = Field(
        ...,
        description="Movie genres, comma-separated",
        examples=["Action, Sci-Fi", "Drama, Thriller", "Comedy"]
    )
    original_language: str = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Original language code (ISO 639-1)",
        examples=["en", "fr", "es", "ja"]
    )
    
    @field_validator('release_date')
    @classmethod
    def validate_release_date(cls, v: str) -> str:
        """Validate that release_date is in correct format."""
        try:
            # Try to parse as date to validate format
            parts = v.split('-')
            if len(parts) != 3:
                raise ValueError("Date must be in YYYY-MM-DD format")
            
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Basic validation
            if year < 1800 or year > 2100:
                raise ValueError("Year must be between 1800 and 2100")
            if month < 1 or month > 12:
                raise ValueError("Month must be between 1 and 12")
            if day < 1 or day > 31:
                raise ValueError("Day must be between 1 and 31")
                
            return v
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {v}. Error: {e}")
    
    @field_validator('original_language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure language code is lowercase."""
        return v.lower().strip()
    
    @field_validator('genre')
    @classmethod
    def validate_genre(cls, v: str) -> str:
        """Ensure genre is properly formatted."""
        # Remove extra spaces around commas
        genres = [g.strip() for g in v.split(',')]
        return ', '.join(genres)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "release_date": "2023-07-21",
                    "title": "Oppenheimer",
                    "overview": "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb.",
                    "genre": "Drama, History, Thriller",
                    "original_language": "en"
                },
                {
                    "release_date": "2023-07-19",
                    "title": "Barbie",
                    "overview": "Barbie and Ken are having the time of their lives in the colorful and seemingly perfect world of Barbie Land.",
                    "genre": "Comedy, Adventure, Fantasy",
                    "original_language": "en"
                }
            ]
        }
    }


class MovieBatchInput(BaseModel):
    """Schema for batch prediction requests."""
    movies: List[MovieInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of movies to predict ratings for"
    )


class PredictionOutput(BaseModel):
    """Schema for a single prediction result."""
    title: str
    predicted_rating: float = Field(..., ge=0.0, le=10.0)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Oppenheimer",
                    "predicted_rating": 8.2
                }
            ]
        }
    }


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction results."""
    predictions: List[PredictionOutput]
    count: int = Field(..., description="Number of predictions")
    model_info: dict = Field(..., description="Information about the model used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    model_loaded: bool
    pipeline_version: Optional[str] = None
    model_info: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    error_type: str

