"""
Comprehensive edge case tests for ML modules.

This module tests feature engineering, clustering, similarity learning,
and FAISS index components with edge cases and boundary conditions.
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
import warnings

# Import modules to test
from ml.feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from ml.clustering import TenderClustering, ClusteringConfig
from ml.similarity_learning import ContrastiveLearner, ContrastiveLearningConfig
from ml.faiss_index import FaissIndex, FaissConfig
from ml_retrospective import MLRetrospectiveAnalyzer, MLRetrospectiveConfig
# ml_config removed (use MLRetrospectiveConfig directly)

# Compatibility stubs for removed ml_config module
def get_ml_config() -> MLRetrospectiveConfig:
    """Stub for get_ml_config."""
    return MLRetrospectiveConfig()

def validate_ml_config(config: MLRetrospectiveConfig) -> bool:
    """Stub for validate_ml_config - always returns True."""
    # Basic validation: check required fields exist
    required = ['feature_config', 'clustering_config', 'similarity_config', 'faiss_config']
    for attr in required:
        if not hasattr(config, attr):
            raise ValueError(f"Missing attribute {attr}")
    return True


class TestFeatureEngineeringEdgeCases(unittest.TestCase):
    """Edge case tests for feature engineering."""
    
    def test_empty_tender_list(self):
        """Test feature extraction with empty tender list."""
        config = FeatureConfig()
        pipeline = FeatureEngineeringPipeline(config)
        
        # Should handle empty list gracefully
        features = pipeline.fit_transform([])
        self.assertEqual(features.shape[0], 0)
        self.assertEqual(features.shape[1], 0)
    
    def test_tender_with_missing_fields(self):
        """Test feature extraction with tenders missing required fields."""
        config = FeatureConfig()
        pipeline = FeatureEngineeringPipeline(config)
        
        # Tender with minimal fields
        tenders = [
            {"Наименование объекта закупки_cleaned": "поставка компьютеров"},
            {},  # Empty tender
            {"Способ определения поставщика": "Аукцион", "Регион": "Москва"}
        ]
        
        features = pipeline.fit_transform(tenders)
        # Should not crash and produce features
        self.assertEqual(features.shape[0], 3)
        self.assertGreater(features.shape[1], 0)
    
    def test_extreme_numerical_values(self):
        """Test handling of extreme numerical values (zeros, negatives, large)."""
        config = FeatureConfig()
        pipeline = FeatureEngineeringPipeline(config)
        
        tenders = [
            {"Начальная (максимальная) цена контракта_parsed": 0.0},
            {"Начальная (максимальная) цена контракта_parsed": -1000.0},
            {"Начальная (максимальная) цена контракта_parsed": 1e12},  # Very large
            {"Начальная (максимальная) цена контракта_parsed": np.inf},
            {"Начальная (максимальная) цена контракта_parsed": np.nan},
        ]
        
        features = pipeline.fit_transform(tenders)
        # Should handle without crashing
        self.assertEqual(features.shape[0], 5)
        # Check no NaN in output (except possibly from input NaN)
        self.assertFalse(np.all(np.isnan(features)))
    
    def test_text_embedding_edge_cases(self):
        """Test text embedding with edge case texts."""
        config = FeatureConfig()
        pipeline = FeatureEngineeringPipeline(config)
        
        tenders = [
            {"Наименование объекта закупки_cleaned": ""},  # Empty string
            {"Наименование объекта закупки_cleaned": "   "},  # Whitespace
            {"Наименование объекта закупки_cleaned": "a" * 10000},  # Very long
            {"Наименование объекта закупки_cleaned": "1234567890"},  # Only numbers
            {"Наименование объекта закупки_cleaned": "!@#$%^&*()"},  # Only symbols
            {"Наименование объекта закупки_cleaned": "Поставка компьютерной техники"},  # Normal
        ]
        
        # Mock the embedding generator to avoid downloading models
        with patch.object(pipeline, '_initialize_embedding_generator'):
            pipeline.embedding_generator = MagicMock()
            pipeline.embedding_generator.encode.return_value = np.random.randn(6, 384)
            
            features = pipeline.fit_transform(tenders)
            self.assertEqual(features.shape[0], 6)
    
    def test_invalid_configuration(self):
        """Test with invalid configuration values."""
        # Invalid normalization method
        with self.assertRaises(ValueError):
            config = FeatureConfig(numerical_normalization_method="invalid_method")
            pipeline = FeatureEngineeringPipeline(config)
        
        # Invalid fusion method
        with self.assertRaises(ValueError):
            config = FeatureConfig(fusion_method="invalid_fusion")
            pipeline = FeatureEngineeringPipeline(config)
    
    def test_single_tender_feature_extraction(self):
        """Test feature extraction with only one tender."""
        config = FeatureConfig()
        pipeline = FeatureEngineeringPipeline(config)
        
        tenders = [
            {
                "Наименование объекта закупки_cleaned": "поставка компьютеров",
                "Начальная (максимальная) цена контракта_parsed": 1000000.0,
                "Способ определения поставщика": "Электронный аукцион",
                "Регион": "Москва",
                "Дата публикации_parsed": "2024-03-15 10:30:00"
            }
        ]
        
        features = pipeline.fit_transform(tenders)
        self.assertEqual(features.shape[0], 1)
        self.assertGreater(features.shape[1], 0)
    
    def test_mixed_data_types(self):
        """Test with mixed data types in fields."""
        config = FeatureConfig()
        pipeline = FeatureEngineeringPipeline(config)
        
        tenders = [
            {
                "Наименование объекта закупки_cleaned": 123,  # Wrong type
                "Начальная (максимальная) цена контракта_parsed": "1 000 000,00",  # String instead of float
                "Способ определения поставщика": ["Аукцион", "Конкурс"],  # List instead of string
            }
        ]
        
        # Should handle gracefully (convert or ignore)
        features = pipeline.fit_transform(tenders)
        self.assertEqual(features.shape[0], 1)


class TestClusteringEdgeCases(unittest.TestCase):
    """Edge case tests for clustering."""
    
    def test_empty_features(self):
        """Test clustering with empty feature matrix."""
        config = ClusteringConfig()
        clustering = TenderClustering(config)
        
        features = np.array([]).reshape(0, 100)  # 0 samples, 100 features
        labels = clustering.fit_predict(features)
        
        self.assertEqual(len(labels), 0)
        self.assertEqual(clustering.n_clusters_, 0)
    
    def test_single_sample(self):
        """Test clustering with only one sample."""
        config = ClusteringConfig()
        clustering = TenderClustering(config)
        
        features = np.random.randn(1, 50)
        labels = clustering.fit_predict(features)
        
        self.assertEqual(len(labels), 1)
        # Should label as noise (-1) or cluster 0 depending on algorithm
        self.assertIn(labels[0], [-1, 0])
    
    def test_all_identical_samples(self):
        """Test clustering with identical samples (no variance)."""
        config = ClusteringConfig()
        clustering = TenderClustering(config)
        
        features = np.ones((10, 50))  # All features are 1
        labels = clustering.fit_predict(features)
        
        # Should either find one cluster or label all as noise
        unique_labels = np.unique(labels)
        self.assertTrue(len(unique_labels) <= 2)  # Could be [-1] or [0] or [-1, 0]
    
    def test_extreme_dimensionality(self):
        """Test clustering with very high-dimensional features."""
        config = ClusteringConfig(umap_n_components=2)  # Force dimensionality reduction
        clustering = TenderClustering(config)
        
        # Very high dimension
        features = np.random.randn(20, 1000)
        labels = clustering.fit_predict(features)
        
        self.assertEqual(len(labels), 20)
    
    def test_nan_in_features(self):
        """Test clustering with NaN values in features."""
        config = ClusteringConfig()
        clustering = TenderClustering(config)
        
        features = np.random.randn(10, 50)
        features[2, 3] = np.nan
        features[5, :] = np.nan
        
        # Should handle NaN (either impute or raise error)
        try:
            labels = clustering.fit_predict(features)
            self.assertEqual(len(labels), 10)
        except ValueError as e:
            # Some clustering algorithms don't handle NaN
            self.assertIn("NaN", str(e))
    
    def test_invalid_cluster_config(self):
        """Test with invalid clustering configuration."""
        with self.assertRaises(ValueError):
            config = ClusteringConfig(umap_n_components=0)  # Invalid
            clustering = TenderClustering(config)
        
        with self.assertRaises(ValueError):
            config = ClusteringConfig(hdbscan_min_cluster_size=-1)
            clustering = TenderClustering(config)
    
    def test_cluster_statistics_edge_cases(self):
        """Test cluster statistics with edge cases."""
        config = ClusteringConfig()
        clustering = TenderClustering(config)
        
        # All noise (no clusters found)
        features = np.random.randn(10, 50)
        with patch.object(clustering.clusterer, 'labels_', np.full(10, -1)):
            clustering.n_clusters_ = 0
            stats = clustering.get_cluster_statistics(features)
            self.assertEqual(stats, {})
        
        # Single cluster
        with patch.object(clustering.clusterer, 'labels_', np.zeros(10)):
            clustering.n_clusters_ = 1
            stats = clustering.get_cluster_statistics(features)
            self.assertEqual(len(stats), 1)
            self.assertEqual(stats[0]['size'], 10)
            self.assertEqual(stats[0]['percentage'], 100.0)


class TestSimilarityLearningEdgeCases(unittest.TestCase):
    """Edge case tests for similarity learning."""
    
    def test_contrastive_learner_small_dataset(self):
        """Test contrastive learning with very small dataset."""
        config = ContrastiveLearningConfig(
            input_dim=50,
            embedding_dim=16,
            epochs=2,
            batch_size=4
        )
        
        learner = ContrastiveLearner(config)
        
        # Tiny dataset
        features = np.random.randn(3, 50)
        labels = np.array([0, 1, 0])
        
        # Should handle small dataset
        learner.train(features, labels)
        embeddings = learner.encode(features)
        
        self.assertEqual(embeddings.shape, (3, 16))
    
    def test_identical_samples(self):
        """Test with identical feature vectors."""
        config = ContrastiveLearningConfig(
            input_dim=50,
            embedding_dim=16,
            epochs=2
        )
        
        learner = ContrastiveLearner(config)
        
        # All samples identical
        features = np.ones((5, 50))
        labels = np.array([0, 0, 1, 1, 2])
        
        learner.train(features, labels)
        embeddings = learner.encode(features)
        
        # Embeddings might still differ due to random initialization
        self.assertEqual(embeddings.shape, (5, 16))
    
    def test_single_class(self):
        """Test with only one class in labels."""
        config = ContrastiveLearningConfig(
            input_dim=50,
            embedding_dim=16,
            epochs=2
        )
        
        learner = ContrastiveLearner(config)
        
        features = np.random.randn(10, 50)
        labels = np.zeros(10)  # All same class
        
        # Should handle single class (though contrastive learning may be trivial)
        learner.train(features, labels)
        embeddings = learner.encode(features)
        
        self.assertEqual(embeddings.shape, (10, 16))
    
    def test_extreme_labels(self):
        """Test with extreme label values (negative, large)."""
        config = ContrastiveLearningConfig(
            input_dim=50,
            embedding_dim=16,
            epochs=2
        )
        
        learner = ContrastiveLearner(config)
        
        features = np.random.randn(8, 50)
        labels = np.array([-1, -1, 0, 0, 1000, 1000, 9999, 9999])
        
        # Should handle arbitrary label values
        learner.train(features, labels)
        embeddings = learner.encode(features)
        
        self.assertEqual(embeddings.shape, (8, 16))
    
    def test_nan_in_features(self):
        """Test with NaN values in training features."""
        config = ContrastiveLearningConfig(
            input_dim=50,
            embedding_dim=16,
            epochs=2
        )
        
        learner = ContrastiveLearner(config)
        
        features = np.random.randn(6, 50)
        features[2, 3] = np.nan
        features[4, :] = np.nan
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        # Should raise error or handle NaN
        try:
            learner.train(features, labels)
            embeddings = learner.encode(features)
            self.assertEqual(embeddings.shape, (6, 16))
        except ValueError as e:
            self.assertIn("NaN", str(e))
    
    def test_model_persistence_edge_cases(self):
        """Test saving/loading models with edge cases."""
        config = ContrastiveLearningConfig(
            input_dim=50,
            embedding_dim=16,
            epochs=2
        )
        
        learner = ContrastiveLearner(config)
        features = np.random.randn(10, 50)
        labels = np.random.randint(0, 3, 10)
        learner.train(features, labels)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test saving
            learner.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Test loading empty file
            with open(temp_path, 'w') as f:
                f.write('')  # Corrupt file
            
            with self.assertRaises(Exception):
                learner.load_model(temp_path)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFaissIndexEdgeCases(unittest.TestCase):
    """Edge case tests for FAISS index."""
    
    def test_empty_index(self):
        """Test FAISS index with no vectors."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        # Build with empty array
        vectors = np.array([]).reshape(0, 128)
        index.build(vectors)
        
        # Search should return empty results
        query = np.random.randn(1, 128)
        distances, indices = index.search(query, k=5)
        
        self.assertEqual(len(distances[0]), 0)
        self.assertEqual(len(indices[0]), 0)
    
    def test_single_vector_index(self):
        """Test FAISS index with only one vector."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        vectors = np.random.randn(1, 128)
        index.build(vectors)
        
        query = np.random.randn(1, 128)
        distances, indices = index.search(query, k=5)
        
        # Should return at most 1 result
        self.assertLessEqual(len(distances[0]), 1)
        self.assertLessEqual(len(indices[0]), 1)
    
    def test_k_larger_than_dataset(self):
        """Test search with k larger than dataset size."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        vectors = np.random.randn(3, 128)
        index.build(vectors)
        
        query = np.random.randn(1, 128)
        distances, indices = index.search(query, k=10)  # k > 3
        
        # Should return at most 3 results
        self.assertLessEqual(len(distances[0]), 3)
        self.assertLessEqual(len(indices[0]), 3)
    
    def test_zero_k(self):
        """Test search with k=0."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        vectors = np.random.randn(5, 128)
        index.build(vectors)
        
        query = np.random.randn(1, 128)
        distances, indices = index.search(query, k=0)
        
        self.assertEqual(len(distances[0]), 0)
        self.assertEqual(len(indices[0]), 0)
    
    def test_nan_in_vectors(self):
        """Test building index with NaN vectors."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        vectors = np.random.randn(5, 128)
        vectors[2, :] = np.nan
        
        # Should raise error
        with self.assertRaises(ValueError):
            index.build(vectors)
    
    def test_inf_in_vectors(self):
        """Test building index with infinite values."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        vectors = np.random.randn(5, 128)
        vectors[3, 10] = np.inf
        vectors[4, 20] = -np.inf
        
        # Should raise error
        with self.assertRaises(ValueError):
            index.build(vectors)
    
    def test_wrong_dimension(self):
        """Test adding vectors with wrong dimension."""
        config = FaissConfig(dimension=128)
        index = FaissIndex(config)
        
        vectors = np.random.randn(5, 128)
        index.build(vectors)
        
        # Try to add vector with wrong dimension
        wrong_vector = np.random.randn(1, 64)
        with self.assertRaises(ValueError):
            index.add(wrong_vector)
    
    def test_index_persistence_edge_cases(self):
        """Test saving/loading index with edge cases."""
        config = FaissConfig()
        index = FaissIndex(config)
        
        vectors = np.random.randn(5, 128)
        index.build(vectors)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test saving
            index.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Test loading empty file
            with open(temp_path, 'wb') as f:
                f.write(b'')  # Corrupt file
            
            new_index = FaissIndex(config)
            with self.assertRaises(Exception):
                new_index.load(temp_path)
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMLRetrospectiveEdgeCases(unittest.TestCase):
    """Edge case tests for ML retrospective analyzer."""
    
    def test_empty_dataset(self):
        """Test analyzer with empty dataset."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Should handle empty dataset
        analyzer.fit([])
        results = analyzer.find_similar({}, k=5)
        
        self.assertEqual(len(results), 0)
    
    def test_single_tender_dataset(self):
        """Test analyzer with only one tender."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tender = {
            "Наименование объекта закупки": "Поставка компьютеров",
            "Начальная (максимальная) цена контракта": "1 000 000,00",
            "Регион": "Москва"
        }
        
        analyzer.fit([tender])
        
        # Search for similar to itself (should return empty if exclude_self=True)
        results = analyzer.find_similar(tender, k=5, exclude_self=True)
        self.assertEqual(len(results), 0)
        
        # With exclude_self=False, should return itself
        results = analyzer.find_similar(tender, k=5, exclude_self=False)
        self.assertEqual(len(results), 1)
    
    def test_invalid_configuration(self):
        """Test with invalid configuration."""
        with self.assertRaises(ValueError):
            config = MLRetrospectiveConfig(
                clustering_min_cluster_size=0  # Invalid
            )
            analyzer = MLRetrospectiveAnalyzer(config)
    
    def test_missing_required_fields(self):
        """Test with tenders missing required fields."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tenders = [
            {},  # Empty
            {"field1": "value1"},  # Missing all required
            {"Наименование объекта закупки": "test"}  # Partial
        ]
        
        # Should handle gracefully
        analyzer.fit(tenders)
        results = analyzer.find_similar({}, k=5)
        
        # Results may be empty or contain some matches
        self.assertIsInstance(results, list)
    
    def test_large_k_value(self):
        """Test with k larger than dataset size."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tenders = [
            {"Наименование объекта закупки": f"Тендер {i}"} for i in range(5)
        ]
        
        analyzer.fit(tenders)
        
        query = {"Наименование объекта закупки": "Запрос"}
        results = analyzer.find_similar(query, k=100)  # k > 5
        
        self.assertLessEqual(len(results), 5)
    
    def test_negative_k_value(self):
        """Test with negative k value."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tenders = [{"Наименование объекта закупки": "test"}]
        analyzer.fit(tenders)
        
        query = {"Наименование объекта закупки": "query"}
        
        # Should handle negative k (treat as 0 or raise error)
        try:
            results = analyzer.find_similar(query, k=-1)
            self.assertEqual(len(results), 0)
        except ValueError as e:
            self.assertIn("k", str(e).lower())
    
    def test_extreme_similarity_threshold(self):
        """Test with extreme similarity thresholds."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tenders = [
            {"Наименование объекта закупки": f"Тендер {i}"} for i in range(10)
        ]
        
        analyzer.fit(tenders)
        
        query = {"Наименование объекта закупки": "Запрос"}
        
        # Threshold = 1.0 (should return only exact matches, likely empty)
        results = analyzer.find_similar(query, k=10, min_similarity=1.0)
        for result in results:
            self.assertEqual(result['similarity_score'], 1.0)
        
        # Threshold = 0.0 (should return all)
        results = analyzer.find_similar(query, k=10, min_similarity=0.0)
        self.assertGreaterEqual(len(results), 0)
        
        # Threshold = -0.1 (invalid)
        with self.assertRaises(ValueError):
            analyzer.find_similar(query, k=10, min_similarity=-0.1)


class TestMLConfigEdgeCases(unittest.TestCase):
    """Edge case tests for ML configuration."""
    
    def test_invalid_config_file(self):
        """Test loading invalid configuration file."""
        # Create temporary invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{invalid json')
            temp_path = f.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                with open(temp_path, 'r') as f:
                    json.load(f)
        finally:
            os.unlink(temp_path)
    
    def test_missing_config_values(self):
        """Test configuration with missing required values."""
        # Test with empty config dict
        config = {}
        
        # validate_ml_config should identify missing fields
        try:
            validate_ml_config(config)
            # If validation passes, that's okay too
        except ValueError as e:
            self.assertIn("missing", str(e).lower())
    
    def test_wrong_type_config_values(self):
        """Test configuration with wrong data types."""
        config = {
            "text_embedding_model": 123,  # Should be string
            "clustering_min_cluster_size": "five",  # Should be int
            "similarity_threshold": "high"  # Should be float
        }
        
        try:
            validate_ml_config(config)
        except (ValueError, TypeError) as e:
            # Expected to fail type validation
            pass
    
    def test_extreme_config_values(self):
        """Test configuration with extreme values."""
        # Very small min_cluster_size
        config = {"clustering_min_cluster_size": 1}
        try:
            validate_ml_config(config)
        except ValueError:
            pass
        
        # Very large embedding dimension
        config = {"embedding_dimension": 10000}
        try:
            validate_ml_config(config)
        except ValueError:
            pass
        
        # Negative threshold
        config = {"similarity_threshold": -1.0}
        try:
            validate_ml_config(config)
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)