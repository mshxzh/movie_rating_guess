"""
Movie Rating Prediction - Custom Transformers Package

This package contains custom sklearn transformers for feature engineering
in the movie rating prediction pipeline.
"""

from .transformers import (
    ColumnSelector,
    DataTypeFixer,
    YearBinning,
    GenreMultiLabelEncoder,
    LanguageGrouper,
    LightweightTextEmbedder,
    SelectiveStandardScaler,
    CategoricalOneHotEncoder,
    SentenceTransformerEmbedder,
)

__all__ = [
    'ColumnSelector',
    'DataTypeFixer',
    'YearBinning',
    'GenreMultiLabelEncoder',
    'LanguageGrouper',
    'LightweightTextEmbedder',
    'SelectiveStandardScaler',
    'CategoricalOneHotEncoder',
    'SentenceTransformerEmbedder',
]

__version__ = '0.1.0'

