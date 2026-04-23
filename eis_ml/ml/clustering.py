"""
UMAP + HDBSCAN clustering implementation for unsupervised tender grouping.

This module provides density-based clustering of procurement tenders using
UMAP for dimensionality reduction and HDBSCAN for clustering. The approach
is designed to handle high-dimensional feature vectors while identifying
natural clusters of similar tenders.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import warnings

# Try to import optional dependencies
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not installed. Install with: pip install hdbscan")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for UMAP + HDBSCAN clustering."""
    
    # UMAP configuration
    umap_n_components: int = 50  # Reduced dimension size
    umap_n_neighbors: int = 15   # Local neighborhood size
    umap_min_dist: float = 0.1   # Minimum distance between points
    umap_metric: str = 'cosine'  # Distance metric
    umap_random_state: int = 42  # Random seed for reproducibility
    
    # HDBSCAN configuration
    hdbscan_min_cluster_size: int = 5      # Minimum cluster size
    hdbscan_min_samples: Optional[int] = None  # Minimum samples in neighborhood
    hdbscan_cluster_selection_epsilon: float = 0.0  # DBSCAN-like epsilon
    hdbscan_cluster_selection_method: str = 'eom'  # 'eom' or 'leaf'
    hdbscan_metric: str = 'euclidean'      # Distance metric for HDBSCAN
    hdbscan_allow_single_cluster: bool = False  # Allow single cluster
    
    # Preprocessing configuration
    preprocess_with_pca: bool = True       # Apply PCA before UMAP for very high-dim data
    pca_n_components: Optional[int] = 100  # PCA components (None for auto)
    scale_features: bool = True            # Standardize features before clustering
    
    # Performance configuration
    n_jobs: int = -1                       # Number of parallel jobs (-1 for all)
    memory: Optional[str] = None           # Memory caching for UMAP
    
    # Output configuration
    assign_outliers_to_nearest: bool = True  # Assign outliers to nearest cluster
    outlier_threshold: float = 0.5          # Probability threshold for outlier assignment


class TenderClustering:
    """
    UMAP + HDBSCAN clustering for tender data.
    
    This class implements a two-stage clustering approach:
    1. Dimensionality reduction with UMAP (optionally preceded by PCA)
    2. Density-based clustering with HDBSCAN
    
    The method is particularly suitable for high-dimensional data
    with varying density clusters and noise/outlier detection.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the clustering model.
        
        Args:
            config: Clustering configuration
        """
        self.config = config or ClusteringConfig()
        self.umap_reducer = None
        self.hdbscan_clusterer = None
        self.pca_reducer = None
        self.scaler = None
        self.is_fitted = False
        
        # Check dependencies
        self._check_dependencies()
        
        # Initialize components
        self._init_components()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not UMAP_AVAILABLE:
            logger.error("UMAP is required but not installed. Install with: pip install umap-learn")
        if not HDBSCAN_AVAILABLE:
            logger.error("HDBSCAN is required but not installed. Install with: pip install hdbscan")
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is required but not installed. Install with: pip install scikit-learn")
    
    def _init_components(self):
        """Initialize UMAP, HDBSCAN, and preprocessing components."""
        # Initialize UMAP
        if UMAP_AVAILABLE:
            self.umap_reducer = umap.UMAP(
                n_components=self.config.umap_n_components,
                n_neighbors=self.config.umap_n_neighbors,
                min_dist=self.config.umap_min_dist,
                metric=self.config.umap_metric,
                random_state=self.config.umap_random_state,
                n_jobs=self.config.n_jobs,
                verbose=False
            )
        
        # Initialize HDBSCAN
        if HDBSCAN_AVAILABLE:
            self.hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
                cluster_selection_epsilon=self.config.hdbscan_cluster_selection_epsilon,
                cluster_selection_method=self.config.hdbscan_cluster_selection_method,
                metric=self.config.hdbscan_metric,
                allow_single_cluster=self.config.hdbscan_allow_single_cluster,
                core_dist_n_jobs=self.config.n_jobs
            )
        
        # Initialize PCA if needed
        if self.config.preprocess_with_pca and SKLEARN_AVAILABLE:
            self.pca_reducer = PCA(
                n_components=self.config.pca_n_components,
                random_state=self.config.umap_random_state
            )
        
        # Initialize scaler if needed
        if self.config.scale_features and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
    
    def fit(self, features: np.ndarray) -> 'TenderClustering':
        """
        Fit the clustering model to feature data.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            self: Fitted clustering model
        """
        logger.info(f"Fitting clustering model on {features.shape[0]} samples with {features.shape[1]} features")
        
        # Adjust PCA n_components if necessary
        if self.pca_reducer is not None and self.config.preprocess_with_pca:
            n_samples, n_features = features.shape
            max_components = min(n_samples, n_features)
            if self.config.pca_n_components is not None and self.config.pca_n_components > max_components:
                logger.warning(
                    f"PCA n_components={self.config.pca_n_components} exceeds "
                    f"min(n_samples, n_features)={max_components}. "
                    f"Reducing to {max_components}."
                )
                # Update PCA reducer with new n_components
                from sklearn.decomposition import PCA
                self.pca_reducer = PCA(
                    n_components=max_components,
                    random_state=self.config.umap_random_state
                )
                self.config.pca_n_components = max_components
        
        # Adjust UMAP n_components if necessary
        if self.umap_reducer is not None:
            n_samples, _ = features.shape
            # Disable UMAP if too few samples
            if n_samples < 5:
                logger.warning(
                    f"Too few samples ({n_samples}) for UMAP. Disabling UMAP reduction."
                )
                self.umap_reducer = None
            else:
                # Adjust n_components
                if self.config.umap_n_components >= n_samples:
                    logger.warning(
                        f"UMAP n_components={self.config.umap_n_components} >= n_samples={n_samples}. "
                        f"Reducing to {n_samples - 1}."
                    )
                    # UMAP doesn't allow n_components >= n_samples, need to recreate
                    import umap
                    self.umap_reducer = umap.UMAP(
                        n_components=max(1, n_samples - 1),
                        n_neighbors=self.config.umap_n_neighbors,
                        min_dist=self.config.umap_min_dist,
                        metric=self.config.umap_metric,
                        random_state=self.config.umap_random_state,
                        n_jobs=self.config.n_jobs,
                        verbose=False
                    )
                    self.config.umap_n_components = max(1, n_samples - 1)
                
                # Adjust n_neighbors if necessary
                if self.config.umap_n_neighbors >= n_samples:
                    new_neighbors = max(2, n_samples - 1)
                    logger.warning(
                        f"UMAP n_neighbors={self.config.umap_n_neighbors} >= n_samples={n_samples}. "
                        f"Reducing to {new_neighbors}."
                    )
                    # Recreate UMAP with adjusted n_neighbors
                    import umap
                    self.umap_reducer = umap.UMAP(
                        n_components=self.config.umap_n_components,
                        n_neighbors=new_neighbors,
                        min_dist=self.config.umap_min_dist,
                        metric=self.config.umap_metric,
                        random_state=self.config.umap_random_state,
                        n_jobs=self.config.n_jobs,
                        verbose=False
                    )
                    self.config.umap_n_neighbors = new_neighbors
        
        # Preprocess features
        processed_features = self._preprocess_features(features, fit=True)
        
        # Apply UMAP dimensionality reduction
        if self.umap_reducer is not None:
            logger.info(f"Applying UMAP reduction to {self.config.umap_n_components} dimensions")
            umap_embeddings = self.umap_reducer.fit_transform(processed_features)
            logger.info(f"UMAP embeddings shape: {umap_embeddings.shape}")
        else:
            # Fallback: use original features (or PCA if available)
            if self.pca_reducer is not None and hasattr(self.pca_reducer, 'components_'):
                umap_embeddings = self.pca_reducer.transform(processed_features)
            else:
                umap_embeddings = processed_features
            logger.warning("UMAP not available, using original/PCA features for clustering")
        
        # Apply HDBSCAN clustering
        if self.hdbscan_clusterer is not None:
            logger.info("Applying HDBSCAN clustering")
            self.hdbscan_clusterer.fit(umap_embeddings)
            
            # Get clustering results
            labels = self.hdbscan_clusterer.labels_
            probabilities = self.hdbscan_clusterer.probabilities_
            
            # Process outliers if configured
            if self.config.assign_outliers_to_nearest:
                labels = self._assign_outliers_to_nearest(labels, umap_embeddings)
            
            # Store results
            self.labels_ = labels
            self.probabilities_ = probabilities
            self.umap_embeddings_ = umap_embeddings
            self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            
            logger.info(f"Found {self.n_clusters_} clusters")
            logger.info(f"Cluster sizes: {self._get_cluster_sizes(labels)}")
            logger.info(f"Outliers (noise): {np.sum(labels == -1)}")
        else:
            logger.error("HDBSCAN not available, cannot perform clustering")
            self.labels_ = np.zeros(features.shape[0], dtype=int)
            self.probabilities_ = np.ones(features.shape[0])
            self.umap_embeddings_ = umap_embeddings
            self.n_clusters_ = 0
        
        self.is_fitted = True
        logger.info("Clustering model fitted successfully")
        
        return self
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster labels.
        
        Args:
            features: Feature matrix
            
        Returns:
            labels: Cluster labels for each sample
        """
        self.fit(features)
        return self.labels_
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Note: HDBSCAN doesn't have a direct predict method for new points.
        This implementation uses approximate nearest neighbor assignment.
        
        Args:
            features: Feature matrix for new samples
            
        Returns:
            labels: Predicted cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess features (using fitted scaler/PCA)
        processed_features = self._preprocess_features(features, fit=False)
        
        # Transform with UMAP
        if self.umap_reducer is not None:
            umap_embeddings = self.umap_reducer.transform(processed_features)
        else:
            if self.pca_reducer is not None and hasattr(self.pca_reducer, 'components_'):
                umap_embeddings = self.pca_reducer.transform(processed_features)
            else:
                umap_embeddings = processed_features
        
        # Assign to nearest cluster using nearest neighbor in UMAP space
        labels = self._predict_with_nn(umap_embeddings)
        
        return labels
    
    def _preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features with scaling and optional PCA.
        
        Args:
            features: Input feature matrix
            fit: Whether to fit the preprocessing components
            
        Returns:
            Processed feature matrix
        """
        processed = features.copy()
        
        # Scale features
        if self.scaler is not None:
            if fit:
                processed = self.scaler.fit_transform(processed)
            else:
                processed = self.scaler.transform(processed)
        
        # Apply PCA for initial dimensionality reduction
        if self.pca_reducer is not None and self.config.preprocess_with_pca:
            if fit:
                processed = self.pca_reducer.fit_transform(processed)
                explained_variance = np.sum(self.pca_reducer.explained_variance_ratio_)
                logger.info(f"PCA explained variance: {explained_variance:.3f}")
            else:
                processed = self.pca_reducer.transform(processed)
        
        return processed
    
    def _assign_outliers_to_nearest(self, labels: np.ndarray, 
                                   embeddings: np.ndarray) -> np.ndarray:
        """
        Assign outlier points (-1) to their nearest cluster.
        
        Args:
            labels: Original cluster labels with -1 for outliers
            embeddings: UMAP embeddings
            
        Returns:
            Updated labels with outliers assigned to nearest clusters
        """
        if not self.config.assign_outliers_to_nearest:
            return labels
        
        # Find outlier indices
        outlier_indices = np.where(labels == -1)[0]
        cluster_indices = np.where(labels != -1)[0]
        
        if len(outlier_indices) == 0 or len(cluster_indices) == 0:
            return labels
        
        # For each outlier, find nearest non-outlier point
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(embeddings[cluster_indices])
        
        distances, nn_indices = knn.kneighbors(embeddings[outlier_indices])
        
        # Assign outlier to cluster of its nearest neighbor
        updated_labels = labels.copy()
        for i, outlier_idx in enumerate(outlier_indices):
            nearest_idx = cluster_indices[nn_indices[i][0]]
            updated_labels[outlier_idx] = labels[nearest_idx]
        
        logger.info(f"Assigned {len(outlier_indices)} outliers to nearest clusters")
        
        return updated_labels
    
    def _predict_with_nn(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels using nearest neighbor in UMAP space.
        
        Args:
            embeddings: UMAP embeddings of new points
            
        Returns:
            Predicted cluster labels
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest training point for each new point
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(self.umap_embeddings_)
        
        distances, indices = knn.kneighbors(embeddings)
        
        # Assign label of nearest training point
        labels = self.labels_[indices.flatten()]
        
        return labels
    
    def _get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """
        Get size of each cluster.
        
        Args:
            labels: Cluster labels
            
        Returns:
            Dictionary mapping cluster ID to size
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_cluster_centroids(self) -> Dict[int, np.ndarray]:
        """
        Calculate centroids of each cluster in UMAP space.
        
        Returns:
            Dictionary mapping cluster ID to centroid vector
        """
        if not hasattr(self, 'umap_embeddings_') or not hasattr(self, 'labels_'):
            raise ValueError("Model must be fitted before getting centroids")
        
        centroids = {}
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip outliers
            
            cluster_points = self.umap_embeddings_[self.labels_ == label]
            centroids[label] = np.mean(cluster_points, axis=0)
        
        return centroids
    
    def get_cluster_statistics(self, features: Optional[np.ndarray] = None) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for each cluster.
        
        Args:
            features: Original feature matrix (optional)
            
        Returns:
            Dictionary with cluster statistics
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Model must be fitted before getting statistics")
        
        stats = {}
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip outliers
            
            cluster_mask = self.labels_ == label
            cluster_size = np.sum(cluster_mask)
            
            # Basic statistics
            cluster_stats = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(self.labels_) * 100)
            }
            
            # Add UMAP centroid if available
            if hasattr(self, 'umap_embeddings_'):
                cluster_points = self.umap_embeddings_[cluster_mask]
                cluster_stats['centroid'] = np.mean(cluster_points, axis=0).tolist()
                cluster_stats['diameter'] = float(np.max(
                    np.linalg.norm(cluster_points - cluster_stats['centroid'], axis=1)
                ))
            
            # Add feature statistics if features provided
            if features is not None and cluster_size > 0:
                cluster_features = features[cluster_mask]
                cluster_stats['feature_means'] = np.mean(cluster_features, axis=0).tolist()
                cluster_stats['feature_stds'] = np.std(cluster_features, axis=0).tolist()
            
            stats[label] = cluster_stats
        
        return stats
    
    def get_outlier_indices(self) -> np.ndarray:
        """
        Get indices of points classified as outliers (noise).
        
        Returns:
            Array of outlier indices
        """
        if not hasattr(self, 'labels_'):
            return np.array([])
        
        return np.where(self.labels_ == -1)[0]
    
    def visualize_clusters(self, save_path: Optional[str] = None, 
                          method: str = 'umap') -> Optional[Any]:
        """
        Create visualization of clusters (requires matplotlib).
        
        Args:
            save_path: Path to save the visualization
            method: Visualization method ('umap', 'tsne', 'pca')
            
        Returns:
            matplotlib figure if not saving to file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            logger.warning("Matplotlib not installed, cannot create visualization")
            return None
        
        if not hasattr(self, 'umap_embeddings_') or not hasattr(self, 'labels_'):
            logger.warning("Model must be fitted before visualization")
            return None
        
        # Further reduce to 2D for visualization
        if method == 'umap' and self.umap_embeddings_.shape[1] > 2:
            # Use UMAP to reduce to 2D
            visualizer = umap.UMAP(n_components=2, random_state=42)
            vis_embeddings = visualizer.fit_transform(self.umap_embeddings_)
        elif method == 'tsne' and SKLEARN_AVAILABLE:
            # Use t-SNE for visualization
            vis_embeddings = TSNE(n_components=2, random_state=42).fit_transform(self.umap_embeddings_)
        elif method == 'pca' and SKLEARN_AVAILABLE:
            # Use PCA for visualization
            vis_embeddings = PCA(n_components=2).fit_transform(self.umap_embeddings_)
        else:
            # Use first two dimensions
            vis_embeddings = self.umap_embeddings_[:, :2]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique labels and colors
        unique_labels = np.unique(self.labels_)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Plot outliers in gray
                mask = self.labels_ == label
                ax.scatter(vis_embeddings[mask, 0], vis_embeddings[mask, 1], 
                          c='gray', alpha=0.5, s=20, label='Outliers')
            else:
                mask = self.labels_ == label
                ax.scatter(vis_embeddings[mask, 0], vis_embeddings[mask, 1], 
                          c=[color], alpha=0.7, s=30, label=f'Cluster {label}')
        
        ax.set_title(f'Tender Clusters (n={self.n_clusters_})')
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Visualization saved to {save_path}")
            return None
        else:
            return fig
    
    def save(self, filepath: str):
        """
        Save the fitted clustering model to disk.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved clustering model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TenderClustering':
        """
        Load a fitted clustering model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded TenderClustering instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded clustering model from {filepath}")
        return model


class ClusterAnalyzer:
    """
    Analyzer for interpreting and explaining clusters.
    
    This class provides methods to analyze cluster characteristics,
    identify key features distinguishing clusters, and generate
    human-readable descriptions of clusters.
    """
    
    def __init__(self, clustering_model: TenderClustering, 
                 feature_names: Optional[List[str]] = None):
        """
        Initialize cluster analyzer.
        
        Args:
            clustering_model: Fitted TenderClustering model
            feature_names: Names of features (for interpretation)
        """
        self.model = clustering_model
        self.feature_names = feature_names
    
    def analyze_cluster_features(self, features: np.ndarray, 
                                top_n: int = 5) -> Dict[int, Dict[str, Any]]:
        """
        Analyze which features are most distinctive for each cluster.
        
        Args:
            features: Original feature matrix
            top_n: Number of top features to identify
            
        Returns:
            Dictionary with feature analysis per cluster
        """
        if not hasattr(self.model, 'labels_'):
            raise ValueError("Clustering model must be fitted")
        
        analysis = {}
        unique_labels = np.unique(self.model.labels_)
        
        # Calculate global feature means
        global_means = np.mean(features, axis=0)
        global_stds = np.std(features, axis=0)
        
        for label in unique_labels:
            if label == -1:
                continue
            
            cluster_mask = self.model.labels_ == label
            cluster_features = features[cluster_mask]
            
            # Calculate cluster means
            cluster_means = np.mean(cluster_features, axis=0)
            
            # Calculate z-scores (how many standard deviations from global mean)
            with np.errstate(divide='ignore', invalid='ignore'):
                z_scores = np.where(global_stds > 0, 
                                   (cluster_means - global_means) / global_stds,
                                   0)
            
            # Get top features with highest absolute z-scores
            abs_z_scores = np.abs(z_scores)
            top_indices = np.argsort(abs_z_scores)[-top_n:][::-1]
            
            # Prepare feature descriptions
            top_features = []
            for idx in top_indices:
                feature_name = f"feature_{idx}" if self.feature_names is None or idx >= len(self.feature_names) else self.feature_names[idx]
                top_features.append({
                    'name': feature_name,
                    'z_score': float(z_scores[idx]),
                    'cluster_mean': float(cluster_means[idx]),
                    'global_mean': float(global_means[idx]),
                    'importance': float(abs_z_scores[idx])
                })
            
            analysis[label] = {
                'top_features': top_features,
                'cluster_mean_vector': cluster_means.tolist(),
                'size': int(np.sum(cluster_mask))
            }
        
        return analysis
    
    def generate_cluster_descriptions(self, features: np.ndarray,
                                     feature_categories: Optional[Dict[str, List[int]]] = None) -> Dict[int, str]:
        """
        Generate human-readable descriptions of clusters.
        
        Args:
            features: Original feature matrix
            feature_categories: Dictionary mapping category names to feature indices
            
        Returns:
            Dictionary mapping cluster ID to description
        """
        # Get feature analysis
        feature_analysis = self.analyze_cluster_features(features, top_n=3)
        
        descriptions = {}
        
        for label, analysis in feature_analysis.items():
            if label == -1:
                descriptions[label] = "Outliers (noise points)"
                continue
            
            # Build description from top features
            top_features = analysis['top_features']
            desc_parts = []
            
            for feat in top_features:
                name = feat['name']
                z_score = feat['z_score']
                cluster_mean = feat['cluster_mean']
                
                # Create descriptive phrase based on z-score
                if abs(z_score) > 2.0:
                    strength = "very high" if z_score > 0 else "very low"
                elif abs(z_score) > 1.0:
                    strength = "high" if z_score > 0 else "low"
                else:
                    strength = "slightly above" if z_score > 0 else "slightly below"
                
                if z_score > 0:
                    desc_parts.append(f"{strength} {name}")
                else:
                    desc_parts.append(f"{strength} {name}")
            
            # Combine into full description
            size = analysis['size']
            description = f"Cluster {label} ({size} tenders): Characterized by "
            description += ", ".join(desc_parts[:2])  # Use top 2 features
            
            if len(desc_parts) > 2:
                description += f", and {len(desc_parts) - 2} other distinctive features"
            
            descriptions[label] = description
        
        return descriptions


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate example data
    np.random.seed(42)
    n_samples = 100
    n_features = 200
    
    # Create synthetic feature matrix with 3 clusters
    features = np.random.randn(n_samples, n_features)
    
    # Add cluster structure
    features[:30] += 2.0  # Cluster 0
    features[30:60] -= 1.5  # Cluster 1
    features[60:90] += np.array([1.0] * 100 + [-1.0] * 100)  # Cluster 2
    # Remaining 10 points are noise
    
    print("Creating clustering model...")
    config = ClusteringConfig(
        umap_n_components=10,
        hdbscan_min_cluster_size=5
    )
    clustering = TenderClustering(config)
    
    print("Fitting clustering model...")
    labels = clustering.fit_predict(features)
    
    print(f"Number of clusters found: {clustering.n_clusters_}")
    print(f"Cluster labels: {np.unique(labels)}")
    
    # Get cluster statistics
    stats = clustering.get_cluster_statistics(features)
    print(f"\nCluster statistics:")
    for cluster_id, cluster_stats in stats.items():
        print(f"  Cluster {cluster_id}: size={cluster_stats['size']}, percentage={cluster_stats['percentage']:.1f}%")
    
    # Analyze clusters
    if clustering.n_clusters_ > 0:
        analyzer = ClusterAnalyzer(clustering)
        descriptions = analyzer.generate_cluster_descriptions(features)
        print(f"\nCluster descriptions:")
        for cluster_id, description in descriptions.items():
            print(f"  {description}")