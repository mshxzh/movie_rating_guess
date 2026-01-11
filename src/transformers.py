"""
Custom transformers for movie rating prediction feature engineering.

These classes are used in the feature engineering pipeline and must be
available when unpickling saved pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from DataFrame"""
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not hasattr(X, "iloc"):
            raise ValueError("ColumnSelector requires pandas DataFrame")
        return X[self.columns].copy()


class DataTypeFixer(BaseEstimator, TransformerMixin):
    """Handle missing values and clean data"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['release_date'] = pd.to_datetime(X['release_date'], errors='coerce')
        for col in ['title', 'overview']:
            X[col] = X[col].fillna('')
        X['genre'] = X['genre'].fillna('')
        X['original_language'] = X['original_language'].fillna('unknown')
        return X


class YearBinning(BaseEstimator, TransformerMixin):
    """Bin years: Before 1960, 1960-1979, then 5-year bins from 1980"""
    
    def __init__(self):
        self.known_bins = set()
        
    @staticmethod
    def assign_bin(year):
        """Assign year to appropriate bin"""
        if pd.isna(year):
            return 'Unknown'
        
        year = int(year)
        
        if year < 1960:
            return 'Before 1960'
        elif year < 1980:
            return '1960-1979'
        else:
            bin_start = ((year - 1980) // 5) * 5 + 1980
            return f'{bin_start}-{bin_start + 4}'
    
    def fit(self, X, y=None):
        X = X.copy()
        X['year'] = X['release_date'].dt.year
        self.known_bins = set(X['year'].apply(self.assign_bin).unique())
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("YearBinning requires pandas DataFrame")
            
        X = X.copy()
        X['year'] = X['release_date'].dt.year
        X['year_bin'] = X['year'].apply(self.assign_bin)
        
        unknown_mask = ~X['year_bin'].isin(self.known_bins)
        if unknown_mask.any():
            X.loc[unknown_mask, 'year_bin'] = 'Future'
        
        X = X.drop(['year', 'release_date'], axis=1)
        return X


class GenreMultiLabelEncoder(BaseEstimator, TransformerMixin):
    """Multi-label one-hot encoding for genres"""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.known_genres = set()
        
    @staticmethod
    def parse_genres(genre_str):
        """Parse comma-separated genre string into list"""
        if pd.isna(genre_str) or genre_str == '':
            return []
        return [g.strip() for g in str(genre_str).split(',')]
    
    def fit(self, X, y=None):
        genre_lists = X['genre'].apply(self.parse_genres)
        self.mlb.fit(genre_lists)
        self.known_genres = set(self.mlb.classes_)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GenreMultiLabelEncoder requires pandas DataFrame")
            
        X = X.copy()
        genre_lists = X['genre'].apply(self.parse_genres)
        
        # Filter out unseen genres
        genre_lists = genre_lists.apply(
            lambda genres: [g for g in genres if g in self.known_genres]
        )
        
        # Transform to multi-label binary matrix
        genre_encoded = self.mlb.transform(genre_lists)
        genre_df = pd.DataFrame(
            genre_encoded,
            columns=[f'genre_{c}' for c in self.mlb.classes_],
            index=X.index
        )
        
        X = X.drop('genre', axis=1)
        return pd.concat([X, genre_df], axis=1)


class LanguageGrouper(BaseEstimator, TransformerMixin):
    """Group rare languages (<threshold of movies) into 'other'"""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.common_languages = set()
        
    def fit(self, X, y=None):
        lang_counts = X['original_language'].value_counts()
        lang_pct = lang_counts / len(X)
        self.common_languages = set(lang_pct[lang_pct >= self.threshold].index)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("LanguageGrouper requires pandas DataFrame")
            
        X = X.copy()
        X['original_language'] = X['original_language'].apply(
            lambda x: x if x in self.common_languages else 'other'
        )
        
        # Handle completely unseen languages
        all_categories = self.common_languages.union({'other'})
        X['original_language'] = X['original_language'].apply(
            lambda x: x if x in all_categories else 'other'
        )
        
        return X


class LightweightTextEmbedder(BaseEstimator, TransformerMixin):
    """Lightweight text embeddings using TF-IDF with configurable ngrams"""
    
    def __init__(self, column, max_features=50, ngram_range=(1, 1), prefix=''):
        """
        Args:
            column: Column name to embed
            max_features: Maximum number of TF-IDF features to keep
            ngram_range: Tuple (min_n, max_n) for n-gram range, e.g., (1, 1) for unigrams, (1, 2) for bigrams
            prefix: Prefix for output feature names
        """
        self.column = column
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.prefix = prefix or column
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        
    def fit(self, X, y=None):
        texts = X[self.column].fillna('').astype(str)
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("LightweightTextEmbedder requires pandas DataFrame")
            
        X = X.copy()
        texts = X[self.column].fillna('').astype(str)
        
        tfidf_matrix = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'{self.prefix}_tfidf_{f}' for f in feature_names],
            index=X.index
        )
        
        X = X.drop(self.column, axis=1)
        return pd.concat([X, tfidf_df], axis=1)


class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Standard scaling for non-embedding features only"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.scale_cols = []
        self.no_scale_cols = []
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SelectiveStandardScaler requires pandas DataFrame")
            
        self.feature_names_in_ = X.columns.tolist()
        
        # Don't scale TF-IDF or embedding features
        self.no_scale_cols = [col for col in X.columns 
                              if 'tfidf' in col.lower() or 'emb' in col.lower()]
        
        # Select only numeric columns from remaining columns
        remaining_cols = [col for col in X.columns if col not in self.no_scale_cols]
        self.scale_cols = X[remaining_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if self.scale_cols:
            self.scaler.fit(X[self.scale_cols])
        
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SelectiveStandardScaler requires pandas DataFrame")
            
        X = X.copy()
        
        if self.scale_cols:
            X[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        
        return X


class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoding for categorical variables"""
    
    def __init__(self, columns):
        self.columns = columns if isinstance(columns, list) else [columns]
        self.encoders = {}
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CategoricalOneHotEncoder requires pandas DataFrame")
            
        for col in self.columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CategoricalOneHotEncoder requires pandas DataFrame")
            
        X = X.copy()
        
        for col in self.columns:
            encoder = self.encoders[col]
            encoded_array = encoder.transform(X[[col]])
            feature_names = encoder.get_feature_names_out([col])
            
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=feature_names,
                index=X.index
            )
            
            X = X.drop(col, axis=1)
            X = pd.concat([X, encoded_df], axis=1)
        
        return X


class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    """Lightweight sentence embeddings using sentence-transformers"""
    
    def __init__(self, column, model_name='all-MiniLM-L6-v2', prefix=''):
        """
        Args:
            column: Column to embed
            model_name: Sentence transformer model
                - 'all-MiniLM-L6-v2': 22M params, 384 dim (FAST, good for Lambda)
                - 'all-MiniLM-L12-v2': 33M params, 384 dim (slower, better quality)
            prefix: Prefix for output columns
        """
        self.column = column
        self.model_name = model_name
        self.prefix = prefix or column
        self.model = None
        self.embedding_dim = None
        
    def fit(self, X, y=None):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            test_emb = self.model.encode(['test'], show_progress_bar=False)
            self.embedding_dim = test_emb.shape[1]
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv pip install sentence-transformers"
            )
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SentenceTransformerEmbedder requires pandas DataFrame")
        
        X = X.copy()
        texts = X[self.column].fillna('').astype(str).tolist()
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Create DataFrame with embeddings
        emb_cols = [f'{self.prefix}_emb_{i}' for i in range(self.embedding_dim)]
        emb_df = pd.DataFrame(
            embeddings,
            columns=emb_cols,
            index=X.index
        )
        
        X = X.drop(self.column, axis=1)
        return pd.concat([X, emb_df], axis=1)

