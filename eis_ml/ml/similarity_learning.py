"""
Contrastive learning for similarity metric learning in tender data.

This module implements contrastive learning approaches to learn a similarity
metric that captures semantic relationships between tenders. The learned
metric can be used for more accurate similarity search and clustering.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveLearningConfig:
    """Configuration for contrastive learning."""
    
    # Model architecture
    input_dim: int = 512                    # Dimension of input features
    hidden_dims: List[int] = None           # Hidden layer dimensions
    embedding_dim: int = 64                 # Output embedding dimension
    dropout_rate: float = 0.2               # Dropout rate
    
    # Training configuration
    batch_size: int = 32                    # Training batch size
    learning_rate: float = 1e-3             # Learning rate
    weight_decay: float = 1e-4              # Weight decay (L2 regularization)
    epochs: int = 50                        # Number of training epochs
    margin: float = 1.0                     # Margin for contrastive loss
    temperature: float = 0.07               # Temperature for contrastive loss
    
    # Pair generation
    positive_pair_strategy: str = "same_cluster"  # "same_cluster", "knn", "random"
    negative_pair_strategy: str = "different_cluster"  # "different_cluster", "random"
    pairs_per_sample: int = 2               # Number of pairs to generate per sample
    hard_negative_mining: bool = True       # Use hard negative mining
    
    # Validation
    validation_split: float = 0.2           # Fraction of data for validation
    early_stopping_patience: int = 10       # Early stopping patience
    
    # Device
    device: str = "auto"                    # "auto", "cpu", "cuda", "mps"
    
    def __post_init__(self):
        """Set default values if None."""
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with positive and negative pairs.
    
    This dataset generates pairs of samples for contrastive learning:
    - Positive pairs: Similar samples (e.g., from same cluster)
    - Negative pairs: Dissimilar samples (e.g., from different clusters)
    """
    
    def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
                 config: Optional[ContrastiveLearningConfig] = None):
        """
        Initialize the contrastive dataset.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Cluster labels or similarity labels (optional)
            config: Configuration for pair generation
        """
        self.features = features
        self.labels = labels
        self.config = config or ContrastiveLearningConfig()
        self.pairs = []
        
        # Generate pairs
        self._generate_pairs()
    
    def _generate_pairs(self):
        """Generate positive and negative pairs for contrastive learning."""
        n_samples = len(self.features)
        
        # If labels are provided, use them for pair generation
        if self.labels is not None:
            self._generate_pairs_with_labels()
        else:
            # Use KNN or random strategies
            self._generate_pairs_without_labels()
    
    def _generate_pairs_with_labels(self):
        """Generate pairs using cluster labels."""
        n_samples = len(self.features)
        labels = self.labels
        
        # Group samples by label
        label_to_indices = {}
        for i, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)
        
        # Generate positive pairs (same cluster)
        positive_pairs = []
        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue
            
            # Generate pairs within the same cluster
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    positive_pairs.append((indices[i], indices[j], 1))
        
        # Generate negative pairs (different clusters)
        negative_pairs = []
        all_labels = list(label_to_indices.keys())
        
        for i in range(n_samples):
            label_i = labels[i]
            
            # Find samples from different clusters
            other_labels = [l for l in all_labels if l != label_i]
            if not other_labels:
                continue
            
            # Select random label
            label_j = random.choice(other_labels)
            j = random.choice(label_to_indices[label_j])
            negative_pairs.append((i, j, 0))
        
        # Balance positive and negative pairs
        n_pairs_per_sample = self.config.pairs_per_sample
        n_positive = min(len(positive_pairs), n_samples * n_pairs_per_sample // 2)
        n_negative = min(len(negative_pairs), n_samples * n_pairs_per_sample // 2)
        
        # Randomly select pairs
        if positive_pairs:
            selected_positive = random.sample(positive_pairs, min(n_positive, len(positive_pairs)))
        else:
            selected_positive = []
        
        if negative_pairs:
            selected_negative = random.sample(negative_pairs, min(n_negative, len(negative_pairs)))
        else:
            selected_negative = []
        
        self.pairs = selected_positive + selected_negative
        random.shuffle(self.pairs)
        
        logger.info(f"Generated {len(self.pairs)} pairs ({len(selected_positive)} positive, {len(selected_negative)} negative)")
    
    def _generate_pairs_without_labels(self):
        """Generate pairs using KNN or random strategies."""
        n_samples = len(self.features)
        n_pairs = n_samples * self.config.pairs_per_sample
        
        # Use KNN for positive pairs if strategy is "knn"
        if self.config.positive_pair_strategy == "knn":
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=min(6, n_samples))
            knn.fit(self.features)
            
            # Get nearest neighbors for each point
            distances, indices = knn.kneighbors(self.features)
            
            positive_pairs = []
            for i in range(n_samples):
                # Skip self
                neighbors = indices[i][1:min(4, len(indices[i]))]
                for j in neighbors:
                    positive_pairs.append((i, j, 1))
        else:
            # Random positive pairs
            positive_pairs = []
            for _ in range(n_pairs // 2):
                i, j = random.sample(range(n_samples), 2)
                positive_pairs.append((i, j, 1))
        
        # Generate negative pairs
        negative_pairs = []
        for _ in range(n_pairs // 2):
            i, j = random.sample(range(n_samples), 2)
            negative_pairs.append((i, j, 0))
        
        self.pairs = positive_pairs + negative_pairs
        random.shuffle(self.pairs)
        
        logger.info(f"Generated {len(self.pairs)} pairs without labels")
    
    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of samples and their similarity label.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Tuple of (sample1, sample2, label) where label is 1 for similar, 0 for dissimilar
        """
        i, j, label = self.pairs[idx]
        
        sample1 = torch.FloatTensor(self.features[i])
        sample2 = torch.FloatTensor(self.features[j])
        label_tensor = torch.FloatTensor([label])
        
        return sample1, sample2, label_tensor


class SimilarityModel(nn.Module):
    """
    Neural network for learning similarity embeddings.
    
    This model projects high-dimensional features into a lower-dimensional
    embedding space where similar samples are close and dissimilar samples
    are far apart.
    """
    
    def __init__(self, config: ContrastiveLearningConfig):
        """
        Initialize the similarity model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Build network layers
        layers = []
        input_dim = config.input_dim
        
        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim
        
        # Output embedding layer
        layers.append(nn.Linear(input_dim, config.embedding_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.network(x)
    
    def encode(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Encode features into embedding space.
        
        Args:
            features: Input feature matrix
            batch_size: Batch size for encoding
            
        Returns:
            Embedding matrix
        """
        self.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch).to(next(self.parameters()).device)
                batch_embeddings = self.forward(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for similarity learning.
    
    This loss encourages similar pairs to have small distances and
    dissimilar pairs to have distances greater than a margin.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: Embeddings of first samples
            embedding2: Embeddings of second samples
            label: Similarity labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Loss value
        """
        # Euclidean distance between embeddings
        distance = torch.norm(embedding1 - embedding2, dim=1)
        
        # Contrastive loss
        loss_similar = label.flatten() * distance.pow(2)
        loss_dissimilar = (1 - label.flatten()) * torch.clamp(self.margin - distance, min=0).pow(2)
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for similarity learning.
    
    This loss uses triplets (anchor, positive, negative) and encourages
    the distance between anchor and positive to be smaller than the
    distance between anchor and negative by a margin.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive (similar) embeddings
            negative: Negative (dissimilar) embeddings
            
        Returns:
            Loss value
        """
        pos_distance = torch.norm(anchor - positive, dim=1)
        neg_distance = torch.norm(anchor - negative, dim=1)
        
        losses = torch.relu(pos_distance - neg_distance + self.margin)
        return torch.mean(losses)


class ContrastiveLearner:
    """
    Main class for contrastive learning of similarity metrics.
    
    This class handles training, evaluation, and inference for learning
    similarity embeddings using contrastive learning approaches.
    """
    
    def __init__(self, config: Optional[ContrastiveLearningConfig] = None):
        """
        Initialize the contrastive learner.
        
        Args:
            config: Configuration for contrastive learning
        """
        self.config = config or ContrastiveLearningConfig()
        self.model = None
        self.loss_history = {"train": [], "val": []}
    
    def build_model(self, input_dim: Optional[int] = None) -> SimilarityModel:
        """
        Build the similarity model.
        
        Args:
            input_dim: Input feature dimension (overrides config if provided)
            
        Returns:
            Similarity model
        """
        if input_dim is not None:
            self.config.input_dim = input_dim
        
        self.model = SimilarityModel(self.config)
        self.model.to(self.config.device)
        
        logger.info(f"Built similarity model with input_dim={self.config.input_dim}, "
                   f"embedding_dim={self.config.embedding_dim}")
        logger.info(f"Model architecture: {self.model}")
        
        return self.model
    
    def train(self, features: np.ndarray, labels: Optional[np.ndarray] = None,
              val_features: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None) -> SimilarityModel:
        """
        Train the similarity model using contrastive learning.
        
        Args:
            features: Training feature matrix
            labels: Training labels for pair generation (optional)
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            
        Returns:
            Trained similarity model
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(features.shape[1])
        
        # Create datasets
        train_dataset = ContrastiveDataset(features, labels, self.config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Create validation dataset if provided
        val_loader = None
        if val_features is not None:
            val_dataset = ContrastiveDataset(val_features, val_labels, self.config)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Initialize loss and optimizer
        criterion = ContrastiveLoss(margin=self.config.margin)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                sample1, sample2, labels = batch
                sample1 = sample1.to(self.config.device)
                sample2 = sample2.to(self.config.device)
                labels = labels.to(self.config.device)
                
                # Forward pass
                embedding1 = self.model(sample1)
                embedding2 = self.model(sample2)
                
                # Compute loss
                loss = criterion(embedding1, embedding2, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / max(train_batches, 1)
            self.loss_history["train"].append(avg_train_loss)
            
            # Validation phase
            avg_val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        sample1, sample2, labels = batch
                        sample1 = sample1.to(self.config.device)
                        sample2 = sample2.to(self.config.device)
                        labels = labels.to(self.config.device)
                        
                        embedding1 = self.model(sample1)
                        embedding2 = self.model(sample2)
                        
                        loss = criterion(embedding1, embedding2, labels)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / max(val_batches, 1)
                self.loss_history["val"].append(avg_val_loss)
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Log progress
            if val_loader is not None:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                           f"val_loss={avg_val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
        
        # Load best model if validation was used
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        logger.info("Training completed")
        return self.model
    
    def encode(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Encode features using the trained model.
        
        Args:
            features: Feature matrix to encode
            batch_size: Batch size for encoding
            
        Returns:
            Embedding matrix
        """
        if self.model is None:
            raise ValueError("Model must be trained before encoding")
        
        return self.model.encode(features, batch_size)
    
    def compute_similarity_matrix(self, features: np.ndarray, 
                                 metric: str = "cosine") -> np.ndarray:
        """
        Compute similarity matrix between all pairs of samples.
        
        Args:
            features: Feature matrix
            metric: Similarity metric ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        embeddings = self.encode(features)
        n_samples = len(embeddings)
        
        if metric == "cosine":
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / np.where(norms > 0, norms, 1)
            
            # Compute cosine similarity
            similarity = np.dot(normalized, normalized.T)
        
        elif metric == "euclidean":
            # Convert Euclidean distance to similarity
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings, embeddings, metric='euclidean')
            max_dist = np.max(distances)
            similarity = 1.0 - distances / max_dist if max_dist > 0 else np.ones_like(distances)
        
        elif metric == "dot":
            # Dot product similarity
            similarity = np.dot(embeddings, embeddings.T)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def find_similar(self, query_features: np.ndarray, 
                    reference_features: np.ndarray,
                    k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar samples to query samples.
        
        Args:
            query_features: Query feature matrix
            reference_features: Reference feature matrix to search in
            k: Number of similar samples to return
            
        Returns:
            Tuple of (indices, similarities) for each query
        """
        query_embeddings = self.encode(query_features)
        reference_embeddings = self.encode(reference_features)
        
        # Compute cosine similarities
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        ref_norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        
        query_normalized = query_embeddings / np.where(query_norms > 0, query_norms, 1)
        ref_normalized = reference_embeddings / np.where(ref_norms > 0, ref_norms, 1)
        
        similarities = np.dot(query_normalized, ref_normalized.T)
        
        # Get top k similarities for each query
        if k > len(reference_features):
            k = len(reference_features)
        
        top_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
        
        return top_indices, top_similarities
    
    def save(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history
        }, filepath)
        
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ContrastiveLearner':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded ContrastiveLearner instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        learner = cls(checkpoint['config'])
        learner.build_model()
        learner.model.load_state_dict(checkpoint['model_state_dict'])
        learner.loss_history = checkpoint['loss_history']
        
        logger.info(f"Loaded model from {filepath}")
        return learner


class SimilarityMetricLearner:
    """
    High-level interface for similarity metric learning.
    
    This class provides a simplified interface for training and using
    similarity models with various loss functions and strategies.
    """
    
    def __init__(self, method: str = "contrastive", **kwargs):
        """
        Initialize the similarity metric learner.
        
        Args:
            method: Learning method ("contrastive", "triplet", "siamese")
            **kwargs: Additional configuration parameters
        """
        self.method = method
        
        # Set default config
        config_kwargs = {
            'input_dim': kwargs.get('input_dim', 512),
            'embedding_dim': kwargs.get('embedding_dim', 64),
            'batch_size': kwargs.get('batch_size', 32),
            'epochs': kwargs.get('epochs', 50),
            'learning_rate': kwargs.get('learning_rate', 1e-3),
        }
        
        self.config = ContrastiveLearningConfig(**config_kwargs)
        self.learner = ContrastiveLearner(self.config)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SimilarityMetricLearner':
        """
        Fit the similarity model.
        
        Args:
            X: Feature matrix
            y: Labels for supervised pair generation (optional)
            
        Returns:
            self
        """
        self.learner.train(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to similarity embeddings.
        
        Args:
            X: Feature matrix
            
        Returns:
            Embedding matrix
        """
        return self.learner.encode(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Args:
            X: Feature matrix
            y: Labels (optional)
            
        Returns:
            Embedding matrix
        """
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    
    # Create synthetic features with 3 clusters
    features = np.random.randn(n_samples, n_features)
    labels = np.zeros(n_samples, dtype=int)
    
    # Create cluster structure
    features[:70] += 2.0
    labels[:70] = 0
    
    features[70:140] -= 1.5
    labels[70:140] = 1
    
    features[140:] += np.array([1.0] * 50 + [-1.0] * 50)
    labels[140:] = 2
    
    print("Creating contrastive learner...")
    config = ContrastiveLearningConfig(
        input_dim=n_features,
        embedding_dim=32,
        epochs=20,
        batch_size=16
    )
    
    learner = ContrastiveLearner(config)
    
    print("Training similarity model...")
    model = learner.train(features, labels)
    
    print("Encoding features...")
    embeddings = learner.encode(features)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Computing similarity matrix...")
    similarity_matrix = learner.compute_similarity_matrix(features[:10])
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    print("Finding similar samples...")
    query = features[:3]
    indices, similarities = learner.find_similar(query, features, k=3)
    
    for i in range(len(query)):
        print(f"Query {i}: most similar samples {indices[i]} with similarities {similarities[i]}")