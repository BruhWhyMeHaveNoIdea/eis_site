"""
FAISS-based similarity search for efficient retrieval of similar tenders.

This module provides FAISS (Facebook AI Similarity Search) integration
for efficient nearest neighbor search in high-dimensional embedding spaces.
FAISS enables fast similarity search even with millions of tenders.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import pickle
import os
import faiss

logger = logging.getLogger(__name__)


@dataclass
class FaissConfig:
    """Configuration for FAISS index."""
    
    # Index type configuration
    index_type: str = "IVF"  # "Flat", "IVF", "HNSW", "IVFPQ"
    metric_type: str = "cosine"  # "cosine", "l2", "ip" (inner product)
    dimension: int = 64  # Dimension of embeddings
    
    # IVF-specific parameters
    nlist: int = 100  # Number of Voronoi cells (IVF)
    nprobe: int = 10  # Number of cells to probe at query time
    
    # HNSW-specific parameters
    hnsw_m: int = 32  # Number of neighbors in HNSW graph
    hnsw_ef_construction: int = 200  # Construction time/accuracy trade-off
    hnsw_ef_search: int = 128  # Search time/accuracy trade-off
    
    # Product Quantization parameters
    pq_m: int = 8  # Number of subquantizers (PQ)
    pq_nbits: int = 8  # Bits per subquantizer
    
    # Training configuration
    training_samples: int = 10000  # Number of samples for training
    training_fraction: float = 0.1  # Fraction of data to use for training
    
    # Performance configuration
    use_gpu: bool = False  # Use GPU acceleration if available
    gpu_id: int = 0  # GPU device ID
    
    # Search configuration
    search_batch_size: int = 1024  # Batch size for search
    normalize_vectors: bool = True  # Normalize vectors for cosine similarity


class FaissIndex:
    """
    FAISS-based similarity index for efficient nearest neighbor search.
    
    This class provides a wrapper around FAISS indices with support for
    various index types, metric spaces, and batch operations.
    """
    
    def __init__(self, config: Optional[FaissConfig] = None):
        """
        Initialize FAISS index.
        
        Args:
            config: FAISS configuration
        """
        self.config = config or FaissConfig()
        self.index = None
        self.is_trained = False
        self.vectors = None
        self.metadata = []
        
        # Initialize index based on configuration
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index based on configuration."""
        dim = self.config.dimension
        metric = self.config.metric_type
        
        # Set metric type
        if metric == "l2":
            metric_type = faiss.METRIC_L2
        elif metric == "cosine":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "ip":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unsupported metric type: {metric}")
        
        # Create index based on type
        if self.config.index_type == "Flat":
            # Exact search index
            if metric == "cosine":
                # For cosine similarity, we need to normalize vectors
                # Use IndexFlatIP with normalized vectors
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexFlatL2(dim)
        
        elif self.config.index_type == "IVF":
            # Inverted file index for approximate search
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, self.config.nlist, metric_type
            )
        
        elif self.config.index_type == "IVFPQ":
            # Product quantization for memory efficiency
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, dim, self.config.nlist,
                self.config.pq_m, self.config.pq_nbits
            )
        
        elif self.config.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, metric_type)
            self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self.index.hnsw.efSearch = self.config.hnsw_ef_search
        
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # Move to GPU if configured
        if self.config.use_gpu:
            self._move_to_gpu()
        
        logger.info(f"Initialized FAISS index: {self.config.index_type}, "
                   f"dim={dim}, metric={metric}")
    
    def _move_to_gpu(self):
        """Move index to GPU if available."""
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.config.gpu_id, self.index)
            logger.info(f"Moved index to GPU {self.config.gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}. Using CPU.")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        
        Args:
            vectors: Input vectors
            
        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def add(self, vectors: np.ndarray, metadata: Optional[List[Any]] = None,
            train_if_needed: bool = True):
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add, shape (n, dimension)
            metadata: Optional metadata for each vector
            train_if_needed: Train the index if not already trained
        """
        # Normalize vectors for cosine similarity if needed
        if self.config.normalize_vectors and self.config.metric_type == "cosine":
            vectors = self._normalize_vectors(vectors)
        
        # Train index if needed
        if train_if_needed and not self.is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.train(vectors)
        
        # Add vectors to index
        self.index.add(vectors.astype(np.float32))
        
        # Store metadata
        if metadata is not None:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([None] * len(vectors))
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def train(self, vectors: np.ndarray):
        """
        Train the index on sample vectors.
        
        Args:
            vectors: Training vectors
        """
        # Check if index needs training
        if not hasattr(self.index, 'is_trained') or self.index.is_trained:
            logger.info("Index already trained")
            return
        
        # Determine number of training samples
        n_train = min(self.config.training_samples, len(vectors))
        if n_train < self.config.training_samples:
            logger.warning(f"Only {n_train} samples available for training, "
                          f"requested {self.config.training_samples}")
        
        # Select training samples
        if n_train < len(vectors):
            indices = np.random.choice(len(vectors), n_train, replace=False)
            training_vectors = vectors[indices]
        else:
            training_vectors = vectors
        
        # Normalize if needed
        if self.config.normalize_vectors and self.config.metric_type == "cosine":
            training_vectors = self._normalize_vectors(training_vectors)
        
        # Train index
        logger.info(f"Training FAISS index on {len(training_vectors)} samples")
        self.index.train(training_vectors.astype(np.float32))
        self.is_trained = True
        
        logger.info("FAISS index training completed")
    
    def search(self, query_vectors: np.ndarray, k: int = 10,
               return_distances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of query vectors.
        
        Args:
            query_vectors: Query vectors, shape (n_queries, dimension)
            k: Number of nearest neighbors to return
            return_distances: Whether to return distances
        
        Returns:
            Tuple of (indices, distances) where:
            - indices: shape (n_queries, k) with indices of nearest neighbors
            - distances: shape (n_queries, k) with distances to nearest neighbors
        """
        # Normalize query vectors if needed
        if self.config.normalize_vectors and self.config.metric_type == "cosine":
            query_vectors = self._normalize_vectors(query_vectors)
        
        # Set nprobe for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.nprobe
        
        # Set efSearch for HNSW indices
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.config.hnsw_ef_search
        
        # Convert to float32 for FAISS
        query_vectors = query_vectors.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vectors, k)
        
        # Convert distances to similarities for cosine metric
        if self.config.metric_type == "cosine" and return_distances:
            # FAISS returns inner products for cosine similarity
            # Convert to cosine similarity (already normalized)
            similarities = distances
            # Optionally convert to distances if needed
            # distances = 1.0 - similarities
            return indices, similarities
        else:
            return indices, distances
    
    def search_with_metadata(self, query_vectors: np.ndarray, k: int = 10
                            ) -> List[List[Dict[str, Any]]]:
        """
        Search for nearest neighbors and return with metadata.
        
        Args:
            query_vectors: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            List of results for each query, where each result is a list of
            dictionaries with keys: 'index', 'distance', 'metadata'
        """
        indices, distances = self.search(query_vectors, k, return_distances=True)
        
        results = []
        for i in range(len(query_vectors)):
            query_results = []
            for j in range(k):
                idx = indices[i, j]
                if idx < 0 or idx >= len(self.metadata):
                    continue  # Invalid index
                
                result = {
                    'index': int(idx),
                    'distance': float(distances[i, j]),
                    'metadata': self.metadata[idx]
                }
                query_results.append(result)
            results.append(query_results)
        
        return results
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 10,
                     batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in batches for memory efficiency.
        
        Args:
            query_vectors: Query vectors
            k: Number of nearest neighbors
            batch_size: Batch size (defaults to config.search_batch_size)
            
        Returns:
            Tuple of (indices, distances) for all queries
        """
        if batch_size is None:
            batch_size = self.config.search_batch_size
        
        n_queries = len(query_vectors)
        all_indices = []
        all_distances = []
        
        for i in range(0, n_queries, batch_size):
            batch = query_vectors[i:i + batch_size]
            indices, distances = self.search(batch, k)
            all_indices.append(indices)
            all_distances.append(distances)
        
        if all_indices:
            indices = np.vstack(all_indices)
            distances = np.vstack(all_distances)
        else:
            indices = np.array([]).reshape(0, k)
            distances = np.array([]).reshape(0, k)
        
        return indices, distances
    
    def get_vector(self, index: int) -> Optional[np.ndarray]:
        """
        Get vector by index.
        
        Args:
            index: Vector index
            
        Returns:
            Vector if available, None otherwise
        """
        # FAISS doesn't have a direct get method for all index types
        # We'll need to reconstruct or store separately
        # For now, return None if we don't have stored vectors
        if self.vectors is not None and 0 <= index < len(self.vectors):
            return self.vectors[index]
        
        return None
    
    def size(self) -> int:
        """
        Get number of vectors in index.
        
        Returns:
            Number of vectors
        """
        return self.index.ntotal if self.index is not None else 0
    
    def save(self, filepath: str):
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index
        """
        # Save FAISS index
        faiss.write_index(self.index, filepath)
        
        # Save metadata separately
        metadata_file = filepath + ".metadata"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'config': self.config
            }, f)
        
        logger.info(f"Saved FAISS index to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FaissIndex':
        """
        Load index from disk.
        
        Args:
            filepath: Path to load index from
            
        Returns:
            Loaded FaissIndex instance
        """
        # Load FAISS index
        index = cls()
        index.index = faiss.read_index(filepath)
        index.is_trained = True
        
        # Update config dimension from loaded index
        if hasattr(index.index, 'd'):
            index.config.dimension = index.index.d
            logger.info(f"Updated config dimension to {index.index.d} from loaded index")
        
        # Load metadata
        metadata_file = filepath + ".metadata"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
            index.metadata = data['metadata']
            # Update config if present
            if 'config' in data:
                index.config = data['config']
        
        logger.info(f"Loaded FAISS index from {filepath}")
        return index


class SimilaritySearchEngine:
    """
    High-level similarity search engine using FAISS.
    
    This class provides a complete similarity search pipeline with
    embedding generation, indexing, and search capabilities.
    """
    
    def __init__(self, embedding_dim: int = 64,
                 index_type: str = "IVF",
                 metric: str = "cosine"):
        """
        Initialize similarity search engine.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type
            metric: Similarity metric
        """
        config = FaissConfig(
            dimension=embedding_dim,
            index_type=index_type,
            metric_type=metric
        )
        
        self.index = FaissIndex(config)
        self.embedding_dim = embedding_dim
    
    def build_index(self, embeddings: np.ndarray,
                    metadata: Optional[List[Any]] = None):
        """
        Build index from embeddings.
        
        Args:
            embeddings: Embedding vectors
            metadata: Optional metadata for each embedding
        """
        logger.info(f"Building index with {len(embeddings)} embeddings")
        
        # Validate embeddings
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Add to index
        self.index.add(embeddings, metadata)
    
    def search_similar(self, query_embeddings: np.ndarray, k: int = 10,
                       include_metadata: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embeddings: Query embeddings
            k: Number of similar items to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results for each query
        """
        if include_metadata:
            return self.index.search_with_metadata(query_embeddings, k)
        else:
            indices, distances = self.index.search(query_embeddings, k)
            
            # Convert to list format
            results = []
            for i in range(len(query_embeddings)):
                query_results = []
                for j in range(k):
                    idx = indices[i, j]
                    if idx < 0:
                        continue
                    
                    result = {
                        'index': int(idx),
                        'distance': float(distances[i, j])
                    }
                    query_results.append(result)
                results.append(query_results)
            
            return results
    
    def find_nearest_neighbors(self, query_embedding: np.ndarray, k: int = 10
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbors for a single query.
        
        Args:
            query_embedding: Query embedding (single vector)
            k: Number of neighbors
            
        Returns:
            Tuple of (indices, distances)
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        indices, distances = self.index.search(query_embedding, k)
        return indices[0], distances[0]
    
    def save(self, filepath: str):
        """
        Save search engine to disk.
        
        Args:
            filepath: Path to save
        """
        self.index.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimilaritySearchEngine':
        """
        Load search engine from disk.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded SimilaritySearchEngine instance
        """
        index = FaissIndex.load(filepath)
        
        engine = cls(
            embedding_dim=index.config.dimension,
            index_type=index.config.index_type,
            metric=index.config.metric_type
        )
        engine.index = index
        engine.embedding_dim = index.config.dimension
        
        return engine


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate example embeddings
    np.random.seed(42)
    n_vectors = 1000
    dim = 64
    
    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Create metadata
    metadata = [{"id": i, "description": f"tender_{i}"} for i in range(n_vectors)]
    
    print("Creating FAISS index...")
    config = FaissConfig(
        dimension=dim,
        index_type="IVF",
        metric_type="cosine",
        nlist=50,
        nprobe=5
    )
    
    index = FaissIndex(config)
    
    print("Adding embeddings to index...")
    index.add(embeddings, metadata)
    
    print(f"Index size: {index.size()}")
    
    # Create query
    query = np.random.randn(1, dim).astype(np.float32)
    
    print("Searching for similar vectors...")
    indices, distances = index.search(query, k=5)
    
    print(f"Query results:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        print(f"  {i+1}. Index {idx}, Distance {dist:.4f}, "
              f"Metadata: {metadata[idx] if idx < len(metadata) else 'N/A'}")
    
    # Test with search engine
    print("\nTesting SimilaritySearchEngine...")
    engine = SimilaritySearchEngine(embedding_dim=dim)
    engine.build_index(embeddings, metadata)
    
    results = engine.search_similar(query, k=3, include_metadata=True)
    
    print(f"Engine results:")
    for i, result in enumerate(results[0]):
        print(f"  {i+1}. Index {result['index']}, Distance {result['distance']:.4f}, "
              f"Metadata: {result['metadata']}")