"""
Embedding generation for Russian text using lightweight models.

This module provides functionality to generate text embeddings for
Russian procurement tender data using Sentence Transformers.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
import warnings

# Try to import sentence-transformers, but provide fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "sentence-transformers not installed. "
        "Install with: pip install sentence-transformers"
    )

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not installed. "
        "Install with: pip install torch"
    )


class EmbeddingGenerator:
    """
    Generator for text embeddings using pre-trained multilingual models.
    
    Uses Sentence Transformers with multilingual MiniLM model for balance
    of accuracy and speed (384 dimensions).
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the Sentence Transformers model to use
            device: Device to run model on ('cpu', 'cuda', 'mps')
            cache_folder: Folder to cache downloaded models
        """
        self.model_name = model_name
        self.model = None
        self.device = device
        self.cache_folder = cache_folder
        self.dimension = 384  # Default for multilingual MiniLM
        
        # Handle "auto" device string
        if self.device == "auto":
            self.device = None
        
        # Auto-detect device if not specified
        if self.device is None:
            if TORCH_AVAILABLE:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = 'cpu'
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Sentence Transformers model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder
            )
            # Get actual dimension from model
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.dimension = self.model.get_sentence_embedding_dimension()
            logging.info(f"Loaded model '{self.model_name}' on device '{self.device}'")
            logging.info(f"Embedding dimension: {self.dimension}")
        except Exception as e:
            logging.error(f"Failed to load model '{self.model_name}': {e}")
            raise
    
    def encode(self, 
               texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = False,
               normalize_embeddings: bool = True) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if self.model is None:
            self._initialize_model()
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Handle empty texts
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        # Clean texts (remove None, convert to string)
        cleaned_texts = []
        for text in texts:
            if text is None:
                cleaned_texts.append("")
            elif not isinstance(text, str):
                cleaned_texts.append(str(text))
            else:
                cleaned_texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings
        )
        
        return embeddings
    
    def encode_tender_text_fields(self,
                                 tender: Dict[str, Any],
                                 text_fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for specific text fields of a tender.
        
        Args:
            tender: Tender dictionary
            text_fields: List of field names to embed
            
        Returns:
            Dictionary mapping field names to their embeddings
        """
        if text_fields is None:
            text_fields = [
                'Наименование объекта закупки_cleaned',
                'Наименование закупки_cleaned',
                'Заказчик_cleaned',
                'Требования к участникам_cleaned',
                'Критерии оценки заявок_cleaned'
            ]
        
        embeddings = {}
        
        for field in text_fields:
            text = tender.get(field, "")
            if not text:
                # Use original field if cleaned version not found
                orig_field = field.replace('_cleaned', '')
                text = tender.get(orig_field, "")
            
            field_embeddings = self.encode(text)
            embeddings[field] = field_embeddings[0] if len(field_embeddings) > 0 else np.zeros(self.dimension)
        
        return embeddings
    
    def get_tender_composite_embedding(self,
                                      tender: Dict[str, Any],
                                      text_field_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create a composite embedding for a tender by combining multiple field embeddings.
        
        Args:
            tender: Preprocessed tender dictionary
            text_field_weights: Weights for each text field embedding
            
        Returns:
            Composite embedding vector
        """
        if text_field_weights is None:
            text_field_weights = {
                'Наименование объекта закупки_cleaned': 0.4,
                'Наименование закупки_cleaned': 0.3,
                'Заказчик_cleaned': 0.2,
                'Требования к участникам_cleaned': 0.05,
                'Критерии оценки заявок_cleaned': 0.05
            }
        
        # Get embeddings for each text field
        field_embeddings = self.encode_tender_text_fields(tender, list(text_field_weights.keys()))
        
        # Weighted combination
        composite = np.zeros(self.dimension)
        total_weight = 0.0
        
        for field, weight in text_field_weights.items():
            if field in field_embeddings:
                composite += field_embeddings[field] * weight
                total_weight += weight
        
        # Normalize if any weights were applied
        if total_weight > 0:
            composite = composite / total_weight
        
        return composite


class CachedEmbeddingGenerator:
    """
    Embedding generator with caching to avoid recomputing embeddings.
    
    Useful when processing the same texts multiple times.
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: Optional[str] = None):
        """
        Initialize cached embedding generator.
        
        Args:
            model_name: Name of the Sentence Transformers model
            device: Device to run model on
        """
        self.generator = EmbeddingGenerator(model_name, device)
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def encode(self, 
               texts: Union[str, List[str]],
               use_cache: bool = True,
               **kwargs) -> np.ndarray:
        """
        Generate embeddings with optional caching.
        
        Args:
            texts: Single text string or list of texts
            use_cache: Whether to use cache
            **kwargs: Additional arguments to pass to encode method
            
        Returns:
            numpy array of embeddings
        """
        if not use_cache:
            return self.generator.encode(texts, **kwargs)
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text is None:
                text = ""
            
            cache_key = hash(text)  # Simple hash for caching
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
                self.hits += 1
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
                self.misses += 1
        
        # Encode texts not in cache
        if texts_to_encode:
            new_embeddings = self.generator.encode(texts_to_encode, **kwargs)
            
            # Store in cache and add to results
            for idx, (text, emb) in enumerate(zip(texts_to_encode, new_embeddings)):
                cache_key = hash(text)
                self.cache[cache_key] = emb
                
                # Insert at correct position
                target_idx = text_indices[idx]
                # We need to insert into embeddings list at correct position
                # Build complete embeddings list
                pass
            
            # Simpler approach: just recompute all if any misses
            # For simplicity, we'll just recompute when there are misses
            # and update cache
            all_embeddings = self.generator.encode(texts, **kwargs)
            
            # Update cache
            for text, emb in zip(texts, all_embeddings):
                cache_key = hash(text)
                self.cache[cache_key] = emb
            
            return all_embeddings
        else:
            # All from cache
            return np.array(embeddings)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(1, self.hits + self.misses)
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# Default global generator for convenience
_default_generator = None


def get_default_generator() -> EmbeddingGenerator:
    """
    Get or create default embedding generator.
    
    Returns:
        Shared EmbeddingGenerator instance
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = EmbeddingGenerator()
    return _default_generator


def encode_text(text: Union[str, List[str]], **kwargs) -> np.ndarray:
    """
    Convenience function to encode text using default generator.
    
    Args:
        text: Text or list of texts to encode
        **kwargs: Additional arguments to pass to encode method
        
    Returns:
        numpy array of embeddings
    """
    generator = get_default_generator()
    return generator.encode(text, **kwargs)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if dependencies are available
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: sentence-transformers not installed.")
        print("Example will show mock embeddings.")
        
        # Create mock generator for demonstration
        class MockGenerator:
            def __init__(self):
                self.dimension = 384
            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.randn(len(texts), self.dimension)
        
        generator = MockGenerator()
    else:
        generator = EmbeddingGenerator()
    
    # Example texts in Russian
    texts = [
        "Поставка компьютерной техники для государственных учреждений",
        "Ремонт административных зданий и сооружений",
        "Закупка медицинского оборудования для больницы"
    ]
    
    print("Generating embeddings for example texts...")
    embeddings = generator.encode(texts)
    
    print(f"Number of texts: {len(texts)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Show similarity between first and other texts
    if len(texts) >= 2:
        from numpy.linalg import norm
        emb1 = embeddings[0]
        for i, (text, emb) in enumerate(zip(texts[1:], embeddings[1:]), 1):
            similarity = np.dot(emb1, emb) / (norm(emb1) * norm(emb))
            print(f"Similarity between text 1 and text {i+1}: {similarity:.4f}")