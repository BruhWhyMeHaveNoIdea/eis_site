"""
ML-specific modules for retrospective analysis.

This package contains feature engineering, clustering, similarity learning, and FAISS search.
"""

# Import key classes for easier access
from .feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from .clustering import TenderClustering, ClusteringConfig, ClusterAnalyzer
from .similarity_learning import ContrastiveLearner, ContrastiveLearningConfig, SimilarityMetricLearner
from .faiss_index import SimilaritySearchEngine, FaissConfig

__all__ = [
    "FeatureEngineeringPipeline",
    "FeatureConfig",
    "TenderClustering",
    "ClusteringConfig",
    "ClusterAnalyzer",
    "ContrastiveLearner",
    "ContrastiveLearningConfig",
    "SimilarityMetricLearner",
    "SimilaritySearchEngine",
    "FaissConfig",
]