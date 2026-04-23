"""
Main ML-based retrospective analysis algorithm for government procurement tenders.

This module implements Algorithm 2 from the design documentation:
ML-based approach using unsupervised clustering and similarity learning.
It integrates feature engineering, UMAP+HDBSCAN clustering, contrastive
learning, and FAISS-based similarity search.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

from ml.feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from ml.clustering import TenderClustering, ClusteringConfig, ClusterAnalyzer
from ml.similarity_learning import ContrastiveLearner, ContrastiveLearningConfig, SimilarityMetricLearner
from ml.faiss_index import SimilaritySearchEngine, FaissConfig
from core.preprocessing import preprocess_tender, compute_field_statistics, load_tenders_from_json
from core.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class MLRetrospectiveConfig:
    """Configuration for ML-based retrospective analysis."""
    
    # Feature engineering configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Clustering configuration
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)
    
    # Similarity learning configuration
    similarity_config: ContrastiveLearningConfig = field(default_factory=ContrastiveLearningConfig)
    
    # FAISS index configuration
    faiss_config: FaissConfig = field(default_factory=FaissConfig)
    
    # Pipeline configuration
    enable_clustering: bool = True           # Enable clustering for pattern discovery
    enable_similarity_learning: bool = True  # Enable contrastive learning
    enable_faiss_index: bool = True          # Enable FAISS for efficient search
    
    # Training configuration
    train_on_all_data: bool = True           # Train on all available data
    validation_split: float = 0.2            # Validation split for similarity learning
    
    # Inference configuration
    search_k: int = 10                       # Default number of similar tenders to return
    min_similarity_threshold: float = 0.0    # Minimum similarity threshold
    include_cluster_info: bool = True        # Include cluster information in results
    include_similarity_breakdown: bool = True # Include similarity breakdown
    
    # Performance configuration
    batch_size: int = 32                     # Batch size for processing
    n_jobs: int = -1                         # Number of parallel jobs (-1 for all)
    
    # Output configuration
    output_format: Dict[str, Any] = field(default_factory=lambda: {
        'include_full_tenders': True,
        'include_breakdown': True,
        'include_explanation': True,
        'include_metadata': True,
        'pretty_print': True,
        'indent': 2
    })


class MLRetrospectiveAnalyzer:
    """
    Main ML-based retrospective analysis algorithm.
    
    This class implements the complete ML pipeline for finding similar
    historical tenders using unsupervised clustering and similarity learning.
    """
    
    def __init__(self, config: Optional[MLRetrospectiveConfig] = None):
        """
        Initialize the ML retrospective analyzer.
        
        Args:
            config: Configuration for the analyzer
        """
        self.config = config or MLRetrospectiveConfig()
        self.feature_pipeline = None
        self.clustering_model = None
        self.similarity_learner = None
        self.search_engine = None
        self.cluster_analyzer = None
        self.embedding_generator = None
        
        # Data storage
        self.tenders = []                    # Original tender data
        self.processed_tenders = []          # Preprocessed tender data
        self.features = None                 # Feature matrix
        self.embeddings = None               # Learned embeddings
        self.cluster_labels = None           # Cluster assignments
        self.cluster_stats = None            # Cluster statistics
        
        # State flags
        self.is_trained = False
        self.is_index_built = False
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize ML components."""
        # Feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(self.config.feature_config)
        
        # Clustering model
        if self.config.enable_clustering:
            self.clustering_model = TenderClustering(self.config.clustering_config)
        
        # Similarity learner
        if self.config.enable_similarity_learning:
            self.similarity_learner = ContrastiveLearner(self.config.similarity_config)
        
        # FAISS search engine
        if self.config.enable_faiss_index:
            self.search_engine = SimilaritySearchEngine(
                embedding_dim=self.config.faiss_config.dimension,
                index_type=self.config.faiss_config.index_type,
                metric=self.config.faiss_config.metric_type
            )
        
        # Embedding generator for text
        try:
            self.embedding_generator = EmbeddingGenerator()
        except ImportError:
            logger.warning("Embedding generator not available")
            self.embedding_generator = None
        
        logger.info("ML retrospective analyzer initialized")
    
    def load_and_preprocess(self, tenders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Load and preprocess tender data.
        
        Args:
            tenders: List of raw tender dictionaries
            
        Returns:
            List of preprocessed tender dictionaries
        """
        logger.info(f"Preprocessing {len(tenders)} tenders")
        
        # Store original tenders
        self.tenders = tenders
        
        # Preprocess each tender
        self.processed_tenders = []
        for tender in tenders:
            processed = preprocess_tender(tender)
            self.processed_tenders.append(processed)
        
        logger.info(f"Preprocessing completed")
        return self.processed_tenders
    
    def extract_features(self) -> np.ndarray:
        """
        Extract features from preprocessed tenders.
        
        Returns:
            Feature matrix
        """
        if not self.processed_tenders:
            raise ValueError("No processed tenders available. Call load_and_preprocess first.")
        
        logger.info("Extracting features from tenders")
        
        # Extract features using feature engineering pipeline
        self.features = self.feature_pipeline.fit_transform(self.processed_tenders)
        
        logger.info(f"Extracted features with shape: {self.features.shape}")
        return self.features
    
    def train_clustering(self) -> Optional[np.ndarray]:
        """
        Train clustering model on extracted features.
        
        Returns:
            Cluster labels if clustering is enabled
        """
        if not self.config.enable_clustering or self.clustering_model is None:
            logger.info("Clustering disabled, skipping")
            return None
        
        if self.features is None:
            self.extract_features()
        
        logger.info("Training clustering model")
        
        # Train clustering model
        self.cluster_labels = self.clustering_model.fit_predict(self.features)
        
        # Get cluster statistics
        self.cluster_stats = self.clustering_model.get_cluster_statistics(self.features)
        
        # Initialize cluster analyzer
        self.cluster_analyzer = ClusterAnalyzer(self.clustering_model)
        
        logger.info(f"Clustering completed: found {self.clustering_model.n_clusters_} clusters")
        logger.info(f"Cluster distribution: {self._get_cluster_distribution()}")
        
        return self.cluster_labels
    
    def train_similarity_model(self) -> Optional[np.ndarray]:
        """
        Train similarity learning model.
        
        Returns:
            Learned embeddings if similarity learning is enabled
        """
        if not self.config.enable_similarity_learning or self.similarity_learner is None:
            logger.info("Similarity learning disabled, skipping")
            return None
        
        if self.features is None:
            self.extract_features()
        
        logger.info("Training similarity learning model")
        
        # Use cluster labels for supervised pair generation if available
        labels = self.cluster_labels if self.cluster_labels is not None else None
        
        # Train similarity model
        self.similarity_learner.train(self.features, labels)
        
        # Generate embeddings
        self.embeddings = self.similarity_learner.encode(self.features)
        
        logger.info(f"Similarity learning completed, embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def build_search_index(self):
        """
        Build FAISS search index for efficient similarity search.
        """
        if not self.config.enable_faiss_index or self.search_engine is None:
            logger.info("FAISS index disabled, skipping")
            return
        
        # Determine what to index
        if self.embeddings is not None and self.config.enable_similarity_learning:
            # Use learned embeddings
            index_vectors = self.embeddings
            logger.info("Using learned embeddings for index")
        elif self.features is not None:
            # Use raw features
            index_vectors = self.features
            logger.info("Using raw features for index")
        else:
            raise ValueError("No features or embeddings available for indexing")
        
        # Create metadata for each tender
        metadata = []
        for i, tender in enumerate(self.processed_tenders):
            meta = {
                'index': i,
                'tender_id': tender.get('Идентификационный код закупки (ИКЗ)', f'tender_{i}'),
                'cluster': int(self.cluster_labels[i]) if self.cluster_labels is not None else -1
            }
            metadata.append(meta)
        
        # Build index
        logger.info(f"Building FAISS index with {len(index_vectors)} vectors")
        self.search_engine.build_index(index_vectors, metadata)
        self.is_index_built = True
        
        logger.info("Search index built successfully")
    
    def train(self, tenders: List[Dict[str, Any]]) -> 'MLRetrospectiveAnalyzer':
        """
        Complete training pipeline: preprocessing, feature extraction,
        clustering, similarity learning, and index building.
        
        Args:
            tenders: List of raw tender dictionaries
            
        Returns:
            self: Trained analyzer
        """
        logger.info("Starting complete training pipeline")
        
        # Step 1: Preprocess data
        self.load_and_preprocess(tenders)
        
        # Step 2: Extract features
        self.extract_features()
        
        # Step 3: Train clustering
        self.train_clustering()
        
        # Step 4: Train similarity model
        self.train_similarity_model()
        
        # Step 5: Build search index
        self.build_search_index()
        
        self.is_trained = True
        logger.info("Training pipeline completed successfully")
        
        return self
    
    def _validate_embedding_dimension(self, query_embedding: np.ndarray) -> bool:
        """
        Validate that query embedding dimension matches FAISS index dimension.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            True if dimension matches or no FAISS index, False otherwise
        """
        if not self.is_index_built or self.search_engine is None:
            return True
            
        # Get FAISS index dimension
        if self.search_engine.index is None:
            return True
            
        # Check if we can access the FAISS index dimension
        # search_engine.index is a FaissIndex object, which has an .index attribute
        if hasattr(self.search_engine.index, 'index') and self.search_engine.index.index is not None:
            # Try to get dimension from the actual FAISS index
            if hasattr(self.search_engine.index.index, 'd'):
                index_dim = self.search_engine.index.index.d
            elif hasattr(self.search_engine.index, 'config'):
                # Fall back to config dimension
                index_dim = self.search_engine.index.config.dimension
            else:
                # Can't determine dimension, assume it's OK
                return True
        elif hasattr(self.search_engine, 'embedding_dim'):
            # Use embedding_dim from search engine
            index_dim = self.search_engine.embedding_dim
        else:
            # Can't determine dimension, assume it's OK
            return True
            
        query_dim = query_embedding.shape[0]
        
        if query_dim != index_dim:
            logger.warning(f"Dimension mismatch: query embedding dimension={query_dim}, "
                          f"FAISS index dimension={index_dim}")
            return False
            
        return True
    
    def _fix_embedding_dimension(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Attempt to fix embedding dimension mismatch.
        
        Args:
            query_embedding: Query embedding vector with wrong dimension
            
        Returns:
            Fixed embedding vector if possible, otherwise raises ValueError
        """
        if not self.is_index_built or self.search_engine is None:
            return query_embedding
            
        # Get FAISS index dimension using same logic as _validate_embedding_dimension
        index_dim = None
        if self.search_engine.index is not None:
            # Check if we can access the FAISS index dimension
            # search_engine.index is a FaissIndex object, which has an .index attribute
            if hasattr(self.search_engine.index, 'index') and self.search_engine.index.index is not None:
                # Try to get dimension from the actual FAISS index
                if hasattr(self.search_engine.index.index, 'd'):
                    index_dim = self.search_engine.index.index.d
                elif hasattr(self.search_engine.index, 'config'):
                    # Fall back to config dimension
                    index_dim = self.search_engine.index.config.dimension
            elif hasattr(self.search_engine, 'embedding_dim'):
                # Use embedding_dim from search engine
                index_dim = self.search_engine.embedding_dim
        
        if index_dim is None:
            # Can't determine dimension, return original
            logger.warning("Cannot determine FAISS index dimension, returning original embedding")
            return query_embedding
            
        query_dim = query_embedding.shape[0]
        
        if query_dim == index_dim:
            return query_embedding
            
        logger.warning(f"Attempting to fix dimension mismatch: {query_dim} -> {index_dim}")
        
        # Case 1: Query dimension is raw features, need to use similarity_learner
        if (self.similarity_learner is not None and
            self.config.enable_similarity_learning and
            query_dim == self.features.shape[1] if self.features is not None else False):
            try:
                # Re-encode using similarity learner
                query_embedding = self.similarity_learner.encode(query_embedding.reshape(1, -1))[0]
                logger.info(f"Re-encoded query using similarity_learner, new dimension: {query_embedding.shape[0]}")
                return query_embedding
            except Exception as e:
                logger.error(f"Failed to re-encode query: {e}")
                
        # Case 2: Pad or truncate (last resort)
        if query_dim > index_dim:
            # Truncate
            fixed_embedding = query_embedding[:index_dim]
            logger.warning(f"Truncated embedding from {query_dim} to {index_dim} dimensions")
        else:
            # Pad with zeros
            fixed_embedding = np.zeros(index_dim)
            fixed_embedding[:query_dim] = query_embedding
            logger.warning(f"Padded embedding from {query_dim} to {index_dim} dimensions with zeros")
            
        return fixed_embedding
    
    def find_similar_tenders(self, query_tender: Dict[str, Any],
                            k: Optional[int] = None,
                            filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Find similar tenders for a query tender.

        Args:
            query_tender: Query tender dictionary (raw or preprocessed)
            k: Number of similar tenders to return (defaults to config.search_k)
            filters: Optional filters to apply to results

        Returns:
            Dictionary with search results
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before finding similar tenders")
        
        if k is None:
            k = self.config.search_k
        
        logger.info(f"Finding similar tenders (k={k})")
        
        # Preprocess query tender
        query_processed = preprocess_tender(query_tender)
        
        # Extract features for query
        query_features = self.feature_pipeline.transform([query_processed])[0]
        
        # Generate embedding for query
        if self.embeddings is not None and self.config.enable_similarity_learning:
            # Use similarity model to encode query
            query_embedding = self.similarity_learner.encode(query_features.reshape(1, -1))[0]
        else:
            # Use raw features
            query_embedding = query_features
        
        # If similarity_learner is missing but FAISS index exists, we may have dimension mismatch
        # Preemptively disable FAISS if we can't encode queries to match index dimension
        if self.is_index_built and self.search_engine is not None and self.similarity_learner is None:
            # Try to get index dimension
            index_dim = None
            if self.search_engine.index is not None:
                if hasattr(self.search_engine.index, 'index') and self.search_engine.index.index is not None:
                    if hasattr(self.search_engine.index.index, 'd'):
                        index_dim = self.search_engine.index.index.d
                elif hasattr(self.search_engine.index, 'config'):
                    index_dim = self.search_engine.index.config.dimension
                elif hasattr(self.search_engine, 'embedding_dim'):
                    index_dim = self.search_engine.embedding_dim
            
            if index_dim is not None and query_embedding.shape[0] != index_dim:
                logger.warning("similarity_learner missing and query dimension doesn't match FAISS index. "
                              "Disabling FAISS index to avoid dimension mismatch error.")
                self.is_index_built = False
        
        # Validate and fix embedding dimension if using FAISS
        if self.is_index_built and self.search_engine is not None:
            if not self._validate_embedding_dimension(query_embedding):
                # Try to fix dimension
                query_embedding = self._fix_embedding_dimension(query_embedding)
                
                # Check again
                if not self._validate_embedding_dimension(query_embedding):
                    logger.warning("Dimension mismatch persists, falling back to brute-force search")
                    self.is_index_built = False  # Temporarily disable FAISS
        
        # Search for similar tenders
        if self.is_index_built and self.search_engine is not None:
            # Use FAISS index for efficient search
            results = self._search_with_faiss(query_embedding, k, filters)
            
            # If FAISS returns empty results due to invalid indices, fall back to brute-force
            if not results and len(self.tenders) > 0:
                logger.warning("FAISS search returned empty results (likely due to dimension mismatch). "
                              "Falling back to brute-force search.")
                self.is_index_built = False  # Temporarily disable FAISS for future queries
                results = self._search_with_bruteforce(query_embedding, k, filters)
        else:
            # Use brute-force similarity computation
            results = self._search_with_bruteforce(query_embedding, k, filters)
        
        # Format results
        formatted_results = self._format_results(query_tender, results)
        
        return formatted_results
    
    def _search_with_faiss(self, query_embedding: np.ndarray, k: int,
                          filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Search for similar tenders using FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filters: Optional filters
            
        Returns:
            List of result dictionaries
        """
        # Reshape for search engine
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        search_results = self.search_engine.search_similar(
            query_embedding, k, include_metadata=True
        )[0]
        
        # Apply filters if provided
        if filters:
            search_results = self._apply_filters(search_results, filters)
        
        # Convert to internal format
        results = []
        invalid_indices = []
        
        for result in search_results:
            idx = result['index']
            distance = result['distance']
            
            # Validate index
            if idx < 0 or idx >= len(self.tenders):
                logger.warning(f"FAISS returned invalid index {idx} (tenders length: {len(self.tenders)}). "
                              f"Distance: {distance}. Skipping this result.")
                invalid_indices.append(idx)
                continue
            
            # Convert distance to similarity score
            similarity = self._distance_to_similarity(distance)
            
            if similarity >= self.config.min_similarity_threshold:
                results.append({
                    'index': idx,
                    'similarity': similarity,
                    'distance': distance,
                    'tender': self.tenders[idx],
                    'cluster': result['metadata'].get('cluster', -1) if result['metadata'] else -1
                })
        
        # If we got too many invalid indices, FAISS index might be corrupted or dimension mismatch
        if invalid_indices and len(invalid_indices) > len(search_results) / 2:
            logger.error(f"FAISS returned {len(invalid_indices)} invalid indices out of {len(search_results)}. "
                         f"FAISS index may be corrupted or dimension mismatch is severe. "
                         f"Disabling FAISS index for this query.")
            # Return empty results to trigger fallback in find_similar_tenders
            return []
        
        # Log if we have some invalid indices but not enough to disable FAISS
        elif invalid_indices:
            logger.warning(f"FAISS returned {len(invalid_indices)} invalid indices. "
                          f"Results may be incomplete.")
        
        return results
    
    def _search_with_bruteforce(self, query_embedding: np.ndarray, k: int,
                               filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Search for similar tenders using brute-force computation.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results
            filters: Optional filters
            
        Returns:
            List of result dictionaries
        """
        # Determine what to compare against
        if self.embeddings is not None:
            database_vectors = self.embeddings
        else:
            database_vectors = self.features
        
        # Compute similarities
        if self.config.faiss_config.metric_type == "cosine":
            # Cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            db_norms = np.linalg.norm(database_vectors, axis=1)
            
            if query_norm == 0:
                similarities = np.zeros(len(database_vectors))
            else:
                similarities = np.dot(database_vectors, query_embedding) / (db_norms * query_norm)
        else:
            # Euclidean distance (convert to similarity)
            distances = np.linalg.norm(database_vectors - query_embedding, axis=1)
            max_dist = np.max(distances) if len(distances) > 0 else 1
            similarities = 1.0 - distances / max_dist
        
        # Get top k indices
        top_indices = np.argsort(-similarities)[:k]
        
        # Build results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            
            if similarity >= self.config.min_similarity_threshold:
                results.append({
                    'index': idx,
                    'similarity': similarity,
                    'distance': 1.0 - similarity,  # Approximate
                    'tender': self.tenders[idx],
                    'cluster': int(self.cluster_labels[idx]) if self.cluster_labels is not None else -1
                })
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        return results
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply filters to search results.
        
        Args:
            results: Search results
            filters: Filter criteria
            
        Returns:
            Filtered results
        """
        filtered_results = []
        
        for result in results:
            tender = result['tender']
            include = True
            
            # Apply each filter
            for key, value in filters.items():
                if key in tender:
                    if isinstance(value, list):
                        # Check if tender value is in list
                        if tender[key] not in value:
                            include = False
                            break
                    else:
                        # Check exact match
                        if tender[key] != value:
                            include = False
                            break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score.
        
        Args:
            distance: Distance value
            
        Returns:
            Similarity score (0-1)
        """
        if self.config.faiss_config.metric_type == "cosine":
            # For cosine, distance is actually 1 - similarity in some cases
            # FAISS returns inner product for cosine, which is similarity
            return max(0.0, min(1.0, distance))
        else:
            # For Euclidean, convert distance to similarity
            # This is a simple conversion, can be improved
            return max(0.0, 1.0 - distance)
    
    def _format_results(self, query_tender: Dict[str, Any],
                       results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format search results according to output configuration.
        
        Args:
            query_tender: Original query tender
            results: Search results
            
        Returns:
            Formatted output dictionary
        """
        output = {
            'query_tender': query_tender if self.config.output_format['include_full_tenders'] else {},
            'similar_tenders': [],
            'metadata': {
                'total_tenders_searched': len(self.tenders),
                'results_returned': len(results),
                'algorithm': 'ml_retrospective_v1',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add cluster analysis if enabled
        if self.config.include_cluster_info and self.cluster_stats is not None:
            output['cluster_analysis'] = self.cluster_stats
        
        # Format each similar tender
        for i, result in enumerate(results):
            formatted = {
                'rank': i + 1,
                'similarity_score': result['similarity'],
                'distance': result.get('distance', 0.0),
                'cluster_id': result.get('cluster', -1)
            }
            
            # Add tender data
            if self.config.output_format['include_full_tenders']:
                formatted['tender'] = result['tender']
            else:
                # Include only essential fields
                essential_fields = ['Идентификационный код закупки (ИКЗ)', 
                                   'Наименование объекта закупки',
                                   'Начальная (максимальная) цена контракта',
                                   'Регион', 'Заказчик', 'Дата публикации']
                formatted['tender'] = {
                    k: result['tender'].get(k, '') for k in essential_fields
                    if k in result['tender']
                }
            
            # Add similarity breakdown if enabled
            if self.config.include_similarity_breakdown:
                formatted['similarity_breakdown'] = self._compute_similarity_breakdown(
                    query_tender, result['tender']
                )
            
            # Add explanation if enabled
            if self.config.output_format['include_explanation']:
                formatted['explanation'] = self._generate_explanation(
                    query_tender, result['tender'], result['similarity'], result.get('cluster', -1)
                )
            
            output['similar_tenders'].append(formatted)
        
        return output
    
    def _compute_similarity_breakdown(self, tender1: Dict[str, Any], 
                                     tender2: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute similarity breakdown by field.
        
        Args:
            tender1: First tender
            tender2: Second tender
            
        Returns:
            Dictionary with similarity scores by field
        """
        breakdown = {}
        
        # Text similarity for key fields
        text_fields = ['Наименование объекта закупки', 'Наименование закупки', 'Заказчик']
        for field in text_fields:
            if field in tender1 and field in tender2:
                text1 = tender1[field]
                text2 = tender2[field]
                
                if self.embedding_generator is not None:
                    # Compute embedding similarity
                    emb1 = self.embedding_generator.encode(text1)
                    emb2 = self.embedding_generator.encode(text2)
                    
                    if len(emb1) > 0 and len(emb2) > 0:
                        similarity = np.dot(emb1[0], emb2[0]) / (
                            np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
                        )
                        breakdown[f'{field}_similarity'] = float(similarity)
        
        # Categorical similarity
        cat_fields = ['Регион', 'Способ определения поставщика', 'Валюта']
        for field in cat_fields:
            if field in tender1 and field in tender2:
                similarity = 1.0 if tender1[field] == tender2[field] else 0.0
                breakdown[f'{field}_similarity'] = similarity
        
        # Numerical similarity (price)
        price_fields = ['Начальная (максимальная) цена контракта', 'Цена контракта']
        for field in price_fields:
            if field in tender1 and field in tender2:
                try:
                    from preprocessing import parse_russian_number
                    price1 = parse_russian_number(tender1[field])
                    price2 = parse_russian_number(tender2[field])
                    
                    if price1 > 0 and price2 > 0:
                        # Gaussian similarity
                        diff = abs(np.log(price1) - np.log(price2))
                        similarity = np.exp(-diff)
                        breakdown[f'{field}_similarity'] = float(similarity)
                except:
                    pass
        
        return breakdown
    
    def _generate_explanation(self, query_tender: Dict[str, Any],
                             result_tender: Dict[str, Any],
                             similarity: float, cluster_id: int) -> str:
        """
        Generate human-readable explanation for similarity.
        
        Args:
            query_tender: Query tender
            result_tender: Result tender
            similarity: Similarity score
            cluster_id: Cluster ID
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Base explanation
        if similarity > 0.8:
            explanations.append("Very high similarity")
        elif similarity > 0.6:
            explanations.append("High similarity")
        elif similarity > 0.4:
            explanations.append("Moderate similarity")
        else:
            explanations.append("Low similarity")
        
        # Cluster-based explanation
        if cluster_id >= 0 and self.cluster_stats is not None:
            explanations.append(f"Both tenders belong to cluster {cluster_id}")
        
        # Field-based explanations
        common_fields = []
        
        # Check region
        region1 = query_tender.get('Регион')
        region2 = result_tender.get('Регион')
        if region1 and region2 and region1 == region2:
            common_fields.append("same region")
        
        # Check procurement method
        method1 = query_tender.get('Способ определения поставщика')
        method2 = result_tender.get('Способ определения поставщика')
        if method1 and method2 and method1 == method2:
            common_fields.append("same procurement method")
        
        # Check customer (partial match)
        customer1 = query_tender.get('Заказчик')
        customer2 = result_tender.get('Заказчик')
        if customer1 and customer2:
            # Ensure both are strings
            if isinstance(customer1, str) and isinstance(customer2, str):
                if customer1 in customer2 or customer2 in customer1:
                    common_fields.append("similar customer")
        
        if common_fields:
            explanations.append(f"Shared characteristics: {', '.join(common_fields)}")
        
        return ". ".join(explanations) + "."
    
    def _get_cluster_distribution(self) -> Dict[int, int]:
        """
        Get distribution of tenders across clusters.
        
        Returns:
            Dictionary mapping cluster ID to count
        """
        if self.cluster_labels is None:
            return {}
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def analyze_clusters(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze and describe each cluster.
        
        Returns:
            Dictionary with cluster analysis
        """
        if self.cluster_analyzer is None or self.features is None:
            return {}
        
        # Get feature names (simplified)
        feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        
        # Analyze clusters
        analysis = self.cluster_analyzer.analyze_cluster_features(
            self.features, top_n=5
        )
        
        # Generate descriptions
        descriptions = self.cluster_analyzer.generate_cluster_descriptions(
            self.features
        )
        
        # Combine analysis with descriptions
        for cluster_id in analysis:
            if cluster_id in descriptions:
                analysis[cluster_id]['description'] = descriptions[cluster_id]
        
        return analysis
    
    def save(self, filepath: str):
        """
        Save trained analyzer to disk.

        Args:
            filepath: Path to save analyzer
        """
        import pickle
        
        # Save main analyzer with all necessary components
        data = {
            'version': '2.1',  # New version with tenders
            'config': self.config,
            'cluster_labels': self.cluster_labels,
            'cluster_stats': self.cluster_stats,
            'is_trained': self.is_trained,
            'is_index_built': self.is_index_built,
            # Save essential ML components
            'feature_pipeline': self.feature_pipeline,
            'similarity_learner': self.similarity_learner,
            'clustering_model': self.clustering_model,
            'cluster_analyzer': self.cluster_analyzer,
            'embedding_generator': self.embedding_generator,
            # Save tenders for search results (essential for FAISS index lookup)
            'tenders': self.tenders if hasattr(self, 'tenders') and self.tenders is not None else [],
            # Save embeddings if available (for consistency checks)
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'features_shape': self.features.shape if self.features is not None else None,
            # Save metadata for backward compatibility
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately if exists
        if self.is_index_built and self.search_engine is not None:
            index_file = filepath + ".faiss"
            self.search_engine.save(index_file)
        
        logger.info(f"Saved analyzer (v2.0) to {filepath}")
        logger.info(f"Saved components: feature_pipeline={self.feature_pipeline is not None}, "
                   f"similarity_learner={self.similarity_learner is not None}, "
                   f"clustering_model={self.clustering_model is not None}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MLRetrospectiveAnalyzer':
        """
        Load trained analyzer from disk.

        Args:
            filepath: Path to load analyzer from
            
        Returns:
            Loaded MLRetrospectiveAnalyzer instance
        """
        import pickle
        import os
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check version for backward compatibility
        version = data.get('version', '1.0')  # Default to old format
        
        if version == '1.0':
            # Old format: only basic data, need to initialize components
            analyzer = cls(data['config'])
            analyzer.cluster_labels = data['cluster_labels']
            analyzer.cluster_stats = data['cluster_stats']
            analyzer.is_trained = data['is_trained']
            analyzer.is_index_built = data['is_index_built']
            
            # Initialize components (they will be untrained)
            analyzer._init_components()
            
            # Load tenders if available (old format may not have them)
            analyzer.tenders = data.get('tenders', [])
            if analyzer.tenders:
                logger.info(f"Loaded {len(analyzer.tenders)} tenders from old format model")
            else:
                logger.warning("No tenders loaded from old format model - FAISS index may return invalid indices")
            
            logger.info(f"Loaded analyzer (v1.0) from {filepath}")
            logger.warning("Loaded old format model - similarity_learner and feature_pipeline are not trained!")
            
        else:
            # New format (v2.0+): restore all components
            analyzer = cls(data['config'])
            analyzer.cluster_labels = data['cluster_labels']
            analyzer.cluster_stats = data['cluster_stats']
            analyzer.is_trained = data['is_trained']
            analyzer.is_index_built = data['is_index_built']
            
            # Restore ML components
            analyzer.feature_pipeline = data.get('feature_pipeline')
            analyzer.similarity_learner = data.get('similarity_learner')
            analyzer.clustering_model = data.get('clustering_model')
            analyzer.cluster_analyzer = data.get('cluster_analyzer')
            analyzer.embedding_generator = data.get('embedding_generator')
            
            # Load tenders if available (version 2.1+)
            if version >= '2.1' or 'tenders' in data:
                analyzer.tenders = data.get('tenders', [])
                logger.info(f"Loaded {len(analyzer.tenders)} tenders for search results")
            else:
                analyzer.tenders = []
                logger.warning("No tenders loaded - FAISS index may return invalid indices")
            
            # Log restored components
            logger.info(f"Loaded analyzer (v{version}) from {filepath}")
            logger.info(f"Restored components: feature_pipeline={analyzer.feature_pipeline is not None}, "
                       f"similarity_learner={analyzer.similarity_learner is not None}, "
                       f"clustering_model={analyzer.clustering_model is not None}")
            
            # Check embedding dimensions consistency
            embeddings_shape = data.get('embeddings_shape')
            features_shape = data.get('features_shape')
            
            if embeddings_shape is not None:
                logger.info(f"Original embeddings shape: {embeddings_shape}")
            if features_shape is not None:
                logger.info(f"Original features shape: {features_shape}")
        
        # Load FAISS index if it exists
        index_file = filepath + ".faiss"
        if os.path.exists(index_file) and analyzer.config.enable_faiss_index:
            try:
                analyzer.search_engine = SimilaritySearchEngine.load(index_file)
                logger.info(f"Loaded FAISS index from {index_file}")
                
                # Verify index dimension matches expected embedding dimension
                if analyzer.search_engine.index is not None:
                    # Get index dimension from FAISS index or config
                    index_dim = None
                    if (hasattr(analyzer.search_engine.index, 'index') and
                        analyzer.search_engine.index.index is not None and
                        hasattr(analyzer.search_engine.index.index, 'd')):
                        index_dim = analyzer.search_engine.index.index.d
                    elif hasattr(analyzer.search_engine.index, 'config'):
                        index_dim = analyzer.search_engine.index.config.dimension
                    elif hasattr(analyzer.search_engine, 'embedding_dim'):
                        index_dim = analyzer.search_engine.embedding_dim
                    
                    if index_dim is not None:
                        logger.info(f"FAISS index dimension: {index_dim}")
                        
                        # If we have similarity_learner, check if its output dimension matches
                        if analyzer.similarity_learner is not None:
                            try:
                                # Try to get output dimension from similarity learner
                                # This depends on the implementation of SimilarityMetricLearner
                                if hasattr(analyzer.similarity_learner, 'output_dim'):
                                    learner_dim = analyzer.similarity_learner.output_dim
                                    if learner_dim != index_dim:
                                        logger.warning(f"Dimension mismatch: similarity_learner.output_dim={learner_dim}, "
                                                      f"FAISS index dimension={index_dim}")
                            except:
                                pass
                        else:
                            # similarity_learner is missing but FAISS index exists
                            # This likely means index was built on embeddings but we can't encode new queries
                            logger.warning("similarity_learner is missing but FAISS index exists. "
                                          "This may cause dimension mismatch. Disabling FAISS index for safety.")
                            analyzer.is_index_built = False
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
        
        return analyzer


def create_ml_retrospective_analyzer(config: Optional[MLRetrospectiveConfig] = None
                                    ) -> MLRetrospectiveAnalyzer:
    """
    Create a configured ML retrospective analyzer.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured MLRetrospectiveAnalyzer
    """
    return MLRetrospectiveAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("ML Retrospective Analysis Algorithm")
    print("=" * 50)
    
    # Create example tender data
    example_tenders = [
        {
            "Идентификационный код закупки (ИКЗ)": "123456",
            "Наименование объекта закупки": "Поставка компьютерной техники",
            "Наименование закупки": "Закупка компьютерного оборудования",
            "Заказчик": "Министерство образования",
            "Начальная (максимальная) цена контракта": "1 250 000,00",
            "Регион": "Москва",
            "Способ определения поставщика": "Электронный аукцион",
            "Дата публикации": "15.03.2024 10:30"
        },
        {
            "Идентификационный код закупки (ИКЗ)": "123457",
            "Наименование объекта закупки": "Поставка серверного оборудования",
            "Наименование закупки": "Закупка ИТ-инфраструктуры",
            "Заказчик": "Министерство цифрового развития",
            "Начальная (максимальная) цена контракта": "2 500 000,00",
            "Регион": "Москва",
            "Способ определения поставщика": "Электронный аукцион",
            "Дата публикации": "20.03.2024 14:00"
        },
        {
            "Идентификационный код закупки (ИКЗ)": "123458",
            "Наименование объекта закупки": "Ремонт административного здания",
            "Наименование закупки": "Капитальный ремонт",
            "Заказчик": "Администрация города",
            "Начальная (максимальная) цена контракта": "5 000 000,00",
            "Регион": "Санкт-Петербург",
            "Способ определения поставщика": "Конкурс",
            "Дата публикации": "10.03.2024 09:00"
        }
    ]
    
    # Create query tender
    query_tender = {
        "Наименование объекта закупки": "Закупка компьютерных комплектующих",
        "Заказчик": "Министерство образования",
        "Начальная (максимальная) цена контракта": "1 300 000,00",
        "Регион": "Москва",
        "Способ определения поставщика": "Электронный аукцион"
    }
    
    print(f"Example dataset: {len(example_tenders)} tenders")
    print(f"Query tender: {query_tender['Наименование объекта закупки']}")
    
    # Create and train analyzer
    print("\n1. Creating ML retrospective analyzer...")
    analyzer = create_ml_retrospective_analyzer()
    
    print("2. Training on example data...")
    analyzer.train(example_tenders)
    
    print("3. Finding similar tenders...")
    results = analyzer.find_similar_tenders(query_tender, k=2)
    
    print("\n4. Results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    print("\n5. Cluster analysis:")
    cluster_analysis = analyzer.analyze_clusters()
    for cluster_id, analysis in cluster_analysis.items():
        print(f"  Cluster {cluster_id}: {analysis.get('description', 'No description')}")