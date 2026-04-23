"""
Advanced feature engineering pipeline for ML-based retrospective analysis.

This module provides feature extraction, transformation, and fusion for
government procurement tender data, combining text embeddings, categorical
encodings, numerical features, and temporal patterns into a unified
representation suitable for unsupervised clustering and similarity learning.

REFACTORED VERSION: Uses configurable field names from config.settings
and ml_config helper functions instead of hardcoded values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from core.preprocessing import (
    parse_russian_number,
    parse_russian_date,
    clean_russian_text,
    extract_date_features,
    compute_field_statistics,
    preprocess_tender
)
from core.embeddings import EmbeddingGenerator, get_default_generator

logger = logging.getLogger(__name__)

# Import configuration helpers
try:
    from ml_config import (
        get_default_text_fields,
        get_default_categorical_fields,
        get_default_numerical_fields,
        get_default_date_fields
    )
    _has_ml_config = True
except ImportError:
    _has_ml_config = False


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    
    # Text embedding configuration
    text_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    text_embedding_device: str = "auto"
    text_fields_to_embed: List[str] = None
    
    # Categorical encoding configuration
    categorical_encoding_method: str = "onehot"  # "onehot", "target", "embedding"
    max_categories_per_feature: int = 50
    categorical_fields: List[str] = None
    
    # Numerical feature configuration
    numerical_normalization_method: str = "standard"  # "standard", "minmax", "log", "robust"
    numerical_fields: List[str] = None
    
    # Temporal feature configuration
    temporal_encoding_method: str = "cyclical"  # "cyclical", "unix", "components"
    reference_date: Optional[str] = None
    date_fields: List[str] = None
    
    # Feature fusion configuration
    fusion_method: str = "concatenate"  # "concatenate", "weighted", "pca"
    output_dimension: Optional[int] = None  # If using dimensionality reduction
    
    # Performance configuration
    batch_size: int = 32
    cache_embeddings: bool = True
    
    def __post_init__(self):
        """Set default values if None, using configuration system."""
        # Text fields to embed
        if self.text_fields_to_embed is None:
            if _has_ml_config:
                self.text_fields_to_embed = get_default_text_fields()
            else:
                # Fallback to hardcoded defaults
                self.text_fields_to_embed = [
                    'Наименование объекта закупки_cleaned',
                    'Наименование закупки_cleaned',
                    'Заказчик_cleaned',
                    'Требования к участникам_cleaned',
                    'Критерии оценки заявок_cleaned'
                ]
        
        # Categorical fields
        if self.categorical_fields is None:
            if _has_ml_config:
                self.categorical_fields = get_default_categorical_fields()
            else:
                # Fallback to hardcoded defaults
                self.categorical_fields = [
                    'Способ определения поставщика',
                    'Регион',
                    'Валюта',
                    'Источник финансирования',
                    'Статус закупки',
                    'Электронная площадка'
                ]
        
        # Numerical fields
        if self.numerical_fields is None:
            if _has_ml_config:
                self.numerical_fields = get_default_numerical_fields()
            else:
                # Fallback to hardcoded defaults
                self.numerical_fields = [
                    'Начальная (максимальная) цена контракта_parsed',
                    'Цена контракта_parsed',
                    'Размер обеспечения заявки_parsed',
                    'Размер обеспечения исполнения контракта_parsed'
                ]
        
        # Date fields
        if self.date_fields is None:
            if _has_ml_config:
                self.date_fields = get_default_date_fields()
            else:
                # Fallback to hardcoded defaults
                self.date_fields = [
                    'Дата публикации',
                    'Дата окончания подачи заявок',
                    'Дата подведения итогов'
                ]


class FeatureEngineeringPipeline:
    """
    Advanced feature engineering pipeline for tender data.
    
    This pipeline transforms raw tender data into feature vectors suitable
    for ML algorithms by:
    1. Generating text embeddings for key text fields
    2. Encoding categorical variables
    3. Normalizing numerical features
    4. Extracting temporal patterns
    5. Fusing all features into a unified representation
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.embedding_generator = None
        self.categorical_encoder = None
        self.numerical_scaler = None
        self.temporal_reference_date = None
        self.feature_names = []
        self.is_fitted = False
        
        # Initialize embedding generator
        self._init_embedding_generator()
        
        # Initialize other components (will be fitted during training)
        self._init_encoders()
    
    def _init_embedding_generator(self):
        """Initialize the text embedding generator."""
        try:
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.config.text_embedding_model,
                device=self.config.text_embedding_device
            )
            logger.info(f"Initialized embedding generator with model: {self.config.text_embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding generator: {e}. Using fallback.")
            self.embedding_generator = None
    
    def _init_encoders(self):
        """Initialize encoders and scalers."""
        # Categorical encoder
        if self.config.categorical_encoding_method == "onehot":
            self.categorical_encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                max_categories=self.config.max_categories_per_feature
            )
        elif self.config.categorical_encoding_method == "target":
            # Target encoding will be implemented separately
            self.categorical_encoder = None
        else:
            self.categorical_encoder = None
        
        # Numerical scaler
        if self.config.numerical_normalization_method == "standard":
            self.numerical_scaler = StandardScaler()
        elif self.config.numerical_normalization_method == "minmax":
            self.numerical_scaler = MinMaxScaler()
        elif self.config.numerical_normalization_method == "robust":
            from sklearn.preprocessing import RobustScaler
            self.numerical_scaler = RobustScaler()
        else:
            self.numerical_scaler = None
    
    def fit(self, tenders: List[Dict[str, Any]]) -> 'FeatureEngineeringPipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            tenders: List of preprocessed tender dictionaries
            
        Returns:
            self: Fitted pipeline
        """
        logger.info(f"Fitting feature engineering pipeline on {len(tenders)} tenders")
        
        # Extract features for fitting
        text_features_list = []
        categorical_features_list = []
        numerical_features_list = []
        temporal_features_list = []
        
        for tender in tenders:
            # Extract text embeddings
            text_features = self._extract_text_features(tender)
            text_features_list.append(text_features)
            
            # Extract categorical features
            categorical_features = self._extract_categorical_features(tender)
            categorical_features_list.append(categorical_features)
            
            # Extract numerical features
            numerical_features = self._extract_numerical_features(tender)
            numerical_features_list.append(numerical_features)
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(tender)
            temporal_features_list.append(temporal_features)
        
        # Fit categorical encoder
        if self.categorical_encoder is not None and categorical_features_list:
            cat_array = np.array(categorical_features_list)
            self.categorical_encoder.fit(cat_array)
        
        # Fit numerical scaler
        if self.numerical_scaler is not None and numerical_features_list:
            num_array = np.array(numerical_features_list)
            self.numerical_scaler.fit(num_array)
        
        # Set temporal reference date if not specified
        if self.config.reference_date is None and temporal_features_list:
            # Use the most recent date as reference
            timestamps = [feat.get('timestamp', 0) for feat in temporal_features_list]
            if any(ts > 0 for ts in timestamps):
                max_timestamp = max(ts for ts in timestamps if ts > 0)
                from datetime import datetime
                self.temporal_reference_date = datetime.fromtimestamp(max_timestamp)
                logger.info(f"Set temporal reference date to: {self.temporal_reference_date}")
        
        self.is_fitted = True
        logger.info("Feature engineering pipeline fitted successfully")
        
        return self
    
    def transform(self, tenders: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform tenders into feature vectors.
        
        Args:
            tenders: List of preprocessed tender dictionaries
            
        Returns:
            feature_matrix: numpy array of shape (n_tenders, n_features)
        """
        if not self.is_fitted:
            logger.warning("Pipeline not fitted. Calling fit_transform instead.")
            return self.fit_transform(tenders)
        
        logger.info(f"Transforming {len(tenders)} tenders into feature vectors")
        
        features_list = []
        
        for tender in tenders:
            # Extract all feature types
            text_features = self._extract_text_features(tender)
            categorical_features = self._extract_categorical_features(tender)
            numerical_features = self._extract_numerical_features(tender)
            temporal_features = self._extract_temporal_features(tender)
            
            # Transform categorical features
            if self.categorical_encoder is not None:
                cat_array = np.array(categorical_features).reshape(1, -1)
                try:
                    categorical_encoded = self.categorical_encoder.transform(cat_array)[0]
                except Exception:
                    # Handle unknown categories
                    categorical_encoded = np.zeros(self.categorical_encoder.n_features_out_)
            else:
                categorical_encoded = np.array(categorical_features)
            
            # Transform numerical features
            if self.numerical_scaler is not None:
                num_array = np.array(numerical_features).reshape(1, -1)
                numerical_normalized = self.numerical_scaler.transform(num_array)[0]
            else:
                numerical_normalized = np.array(numerical_features)
            
            # Transform temporal features
            temporal_vector = self._temporal_features_to_vector(temporal_features)
            
            # Fuse features
            fused_features = self._fuse_features(
                text_features, categorical_encoded, 
                numerical_normalized, temporal_vector
            )
            
            features_list.append(fused_features)
        
        feature_matrix = np.array(features_list)
        logger.info(f"Generated feature matrix with shape {feature_matrix.shape}")
        
        return feature_matrix
    
    def fit_transform(self, tenders: List[Dict[str, Any]]) -> np.ndarray:
        """
        Fit the pipeline and transform the data.
        
        Args:
            tenders: List of preprocessed tender dictionaries
            
        Returns:
            feature_matrix: numpy array of shape (n_tenders, n_features)
        """
        self.fit(tenders)
        return self.transform(tenders)
    
    def _extract_text_features(self, tender: Dict[str, Any]) -> np.ndarray:
        """
        Extract text embeddings for a tender.
        
        Args:
            tender: Preprocessed tender dictionary
            
        Returns:
            text_embeddings: Concatenated text embeddings
        """
        if self.embedding_generator is None:
            # Fallback: return zeros
            embedding_dim = 384  # Default for multilingual MiniLM
            return np.zeros(len(self.config.text_fields_to_embed) * embedding_dim)
        
        embeddings = []
        
        for field in self.config.text_fields_to_embed:
            text = tender.get(field, "")
            if not text:
                # Try to get cleaned version
                orig_field = field.replace('_cleaned', '')
                text = tender.get(orig_field, "")
            
            # Generate embedding
            if self.embedding_generator:
                field_embedding = self.embedding_generator.encode(text)
                if len(field_embedding) > 0:
                    embeddings.append(field_embedding[0])
                else:
                    embeddings.append(np.zeros(self.embedding_generator.dimension))
            else:
                embeddings.append(np.zeros(384))
        
        # Concatenate all text embeddings
        if embeddings:
            return np.concatenate(embeddings)
        else:
            return np.array([])
    
    def _extract_categorical_features(self, tender: Dict[str, Any]) -> List[str]:
        """
        Extract categorical features as strings.
        
        Args:
            tender: Preprocessed tender dictionary
            
        Returns:
            categorical_values: List of categorical values
        """
        values = []
        
        for field in self.config.categorical_fields:
            value = tender.get(field, "")
            if value is None:
                value = ""
            values.append(str(value))
        
        return values
    
    def _extract_numerical_features(self, tender: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features.
        
        Args:
            tender: Preprocessed tender dictionary
            
        Returns:
            numerical_values: List of numerical values
        """
        values = []
        
        for field in self.config.numerical_fields:
            value = tender.get(field, 0.0)
            if isinstance(value, str):
                # Parse if string
                value = parse_russian_number(value)
            elif value is None:
                value = 0.0
            
            # Apply log transformation if configured
            if self.config.numerical_normalization_method == "log" and value > 0:
                value = np.log1p(value)
            
            values.append(float(value))
        
        return values
    
    def _extract_temporal_features(self, tender: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract temporal features from date fields.
        
        Args:
            tender: Preprocessed tender dictionary
            
        Returns:
            temporal_features: Dictionary of temporal features
        """
        # Use the first date field as primary temporal feature
        primary_date_field = self.config.date_fields[0] if self.config.date_fields else None
        
        if primary_date_field and primary_date_field in tender:
            date_str = tender[primary_date_field]
            if isinstance(date_str, str):
                # Parse date string
                dt = parse_russian_date(date_str)
                if dt:
                    # Extract features
                    features = extract_date_features(date_str, self.temporal_reference_date)
                    return features
        
        # Return default features if no date found
        return {
            'timestamp': 0.0,
            'day_of_week': 0.0,
            'month': 0.0,
            'quarter': 0.0,
            'year': 0.0,
            'days_from_reference': 0.0
        }
    
    def _temporal_features_to_vector(self, temporal_features: Dict[str, float]) -> np.ndarray:
        """
        Convert temporal features dictionary to vector.
        
        Args:
            temporal_features: Dictionary of temporal features
            
        Returns:
            temporal_vector: numpy array
        """
        if self.config.temporal_encoding_method == "cyclical":
            # Cyclical encoding for periodic features
            day_of_week = temporal_features.get('day_of_week', 0)
            month = temporal_features.get('month', 1)
            
            # Encode day of week as sin/cos
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Encode month as sin/cos
            month_sin = np.sin(2 * np.pi * (month - 1) / 12)
            month_cos = np.cos(2 * np.pi * (month - 1) / 12)
            
            # Include other features
            timestamp = temporal_features.get('timestamp', 0)
            days_from_reference = temporal_features.get('days_from_reference', 0)
            
            return np.array([
                timestamp,
                days_from_reference,
                day_sin, day_cos,
                month_sin, month_cos,
                temporal_features.get('quarter', 1),
                temporal_features.get('year', 2024)
            ])
        
        elif self.config.temporal_encoding_method == "unix":
            # Simple Unix timestamp encoding
            return np.array([temporal_features.get('timestamp', 0)])
        
        else:  # "components"
            # All components as separate features
            return np.array([
                temporal_features.get('timestamp', 0),
                temporal_features.get('day_of_week', 0),
                temporal_features.get('month', 1),
                temporal_features.get('quarter', 1),
                temporal_features.get('year', 2024),
                temporal_features.get('days_from_reference', 0)
            ])
    
    def _fuse_features(self, 
                      text_features: np.ndarray,
                      categorical_features: np.ndarray,
                      numerical_features: np.ndarray,
                      temporal_features: np.ndarray) -> np.ndarray:
        """
        Fuse different feature types into a single vector.
        
        Args:
            text_features: Text embeddings
            categorical_features: Encoded categorical features
            numerical_features: Normalized numerical features
            temporal_features: Temporal feature vector
            
        Returns:
            fused_features: Combined feature vector
        """
        if self.config.fusion_method == "concatenate":
            # Simple concatenation
            return np.concatenate([
                text_features,
                categorical_features,
                numerical_features,
                temporal_features
            ])
        
        elif self.config.fusion_method == "weighted":
            # Weighted combination (requires dimension matching)
            # For simplicity, we'll use concatenation with weights
            # This would need more sophisticated implementation
            weights = {
                'text': 0.4,
                'categorical': 0.2,
                'numerical': 0.2,
                'temporal': 0.2
            }
            
            # Scale each feature type by weight
            text_weighted = text_features * weights['text']
            cat_weighted = categorical_features * weights['categorical']
            num_weighted = numerical_features * weights['numerical']
            temp_weighted = temporal_features * weights['temporal']
            
            return np.concatenate([
                text_weighted,
                cat_weighted,
                num_weighted,
                temp_weighted
            ])
        
        else:  # "pca" or other methods would require additional processing
            # Default to concatenation
            return np.concatenate([
                text_features,
                categorical_features,
                numerical_features,
                temporal_features
            ])
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of each feature type.
        
        Returns:
            Dictionary with feature type dimensions
        """
        # Estimate dimensions based on configuration
        text_dim = 0
        if self.embedding_generator:
            text_dim = self.embedding_generator.dimension * len(self.config.text_fields_to_embed)
        else:
            text_dim = 384 * len(self.config.text_fields_to_embed)
        
        cat_dim = 0
        if self.categorical_encoder and hasattr(self.categorical_encoder, 'n_features_out_'):
            cat_dim = self.categorical_encoder.n_features_out_
        else:
            # Estimate based on number of categories
            cat_dim = len(self.config.categorical_fields) * 10  # Rough estimate
        
        num_dim = len(self.config.numerical_fields)
        
        temp_dim = 0
        if self.config.temporal_encoding_method == "cyclical":
            temp_dim = 8
        elif self.config.temporal_encoding_method == "unix":
            temp_dim = 1
        else:  # "components"
            temp_dim = 6
        
        total_dim = text_dim + cat_dim + num_dim + temp_dim
        
        return {
            'text': text_dim,
            'categorical': cat_dim,
            'numerical': num_dim,
            'temporal': temp_dim,
            'total': total_dim
        }
    
    def save(self, filepath: str):
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath: Path to save the pipeline
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved feature engineering pipeline to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureEngineeringPipeline':
        """
        Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to load the pipeline from
            
        Returns:
            Loaded FeatureEngineeringPipeline instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        logger.info(f"Loaded feature engineering pipeline from {filepath}")
        return pipeline


# ============================================================================
# Configuration-aware helper functions
# ============================================================================

def get_feature_config_from_settings() -> FeatureConfig:
    """
    Create FeatureConfig from configuration settings.
    
    Returns:
        FeatureConfig instance with fields from configuration system
    """
    config = FeatureConfig()
    
    # Update fields from configuration if available
    if _has_ml_config:
        config.text_fields_to_embed = get_default_text_fields()
        config.categorical_fields = get_default_categorical_fields()
        config.numerical_fields = get_default_numerical_fields()
        config.date_fields = get_default_date_fields()
    
    return config


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Pipeline Example")
    print("=" * 50)
    
    # Create configuration
    config = FeatureConfig()
    print(f"Text fields to embed: {config.text_fields_to_embed[:2]}...")
    print(f"Categorical fields: {config.categorical_fields[:2]}...")
    print(f"Numerical fields: {config.numerical_fields[:2]}...")
    print(f"Date fields: {config.date_fields[:2]}...")
    
    # Create pipeline
    pipeline = FeatureEngineeringPipeline(config)
    print(f"\nPipeline initialized with {len(config.text_fields_to_embed)} text fields")
    
    # Example tender
    example_tender = {
        "Наименование объекта закупки_cleaned": "поставка компьютерной техники",
        "Наименование закупки_cleaned": "закупка оборудования",
        "Заказчик_cleaned": "министерство образования",
        "Способ определения поставщика": "Электронный аукцион",
        "Регион": "Москва",
        "Начальная (максимальная) цена контракта_parsed": 1250000.0,
        "Дата публикации": "15.03.2024 10:30"
    }
    
    # Extract features (pipeline needs to be fitted first)
    print("\nNote: Pipeline needs to be fitted with multiple tenders before transformation")
    print("Example configuration check complete.")