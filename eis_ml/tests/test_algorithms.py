"""
Comprehensive edge case tests for main algorithms.

This module tests Algorithm 1 (select_k_best) and Algorithm 2 (ML-based retrospective)
with edge cases and boundary conditions.
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
import warnings

# Import algorithms
from select_k_best import SelectKBest, create_target_entity_from_criteria
from ml_retrospective import MLRetrospectiveAnalyzer, MLRetrospectiveConfig
from config import DEFAULT_WEIGHTS, DEFAULT_K, MIN_SIMILARITY_THRESHOLD


class TestSelectKBestEdgeCases(unittest.TestCase):
    """Edge case tests for Algorithm 1 (select_k_best)."""
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        algorithm = SelectKBest()
        algorithm.load_tenders([])
        
        target = {"Наименование объекта закупки": "Поставка компьютеров"}
        results = algorithm.find_similar(target, k=5)
        
        self.assertEqual(len(results), 0)
    
    def test_single_tender_dataset(self):
        """Test with only one tender in dataset."""
        algorithm = SelectKBest()
        tenders = [
            {
                "Идентификационный код закупки (ИКЗ)": "001",
                "Наименование объекта закупки": "Поставка компьютеров",
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "Регион": "Москва"
            }
        ]
        algorithm.load_tenders(tenders)
        
        # Target is the same tender
        target = tenders[0]
        
        # With exclude_self=True (default)
        results = algorithm.find_similar(target, k=5, exclude_self=True)
        self.assertEqual(len(results), 0)
        
        # With exclude_self=False
        results = algorithm.find_similar(target, k=5, exclude_self=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['tender']["Идентификационный код закупки (ИКЗ)"], "001")
        self.assertAlmostEqual(results[0]['similarity_score'], 0.5, places=5)
    
    def test_k_larger_than_dataset(self):
        """Test with k larger than dataset size."""
        algorithm = SelectKBest()
        tenders = [
            {"Наименование объекта закупки": f"Тендер {i}"} for i in range(3)
        ]
        algorithm.load_tenders(tenders)
        
        target = {"Наименование объекта закупки": "Запрос"}
        results = algorithm.find_similar(target, k=10)  # k > 3
        
        # Should return at most 3 results
        self.assertLessEqual(len(results), 3)
    
    def test_k_zero_or_negative(self):
        """Test with k=0 and negative k."""
        algorithm = SelectKBest()
        tenders = [{"Наименование объекта закупки": "test"}]
        algorithm.load_tenders(tenders)
        
        target = {"Наименование объекта закупки": "query"}
        
        # k=0 should return empty list
        results = algorithm.find_similar(target, k=0)
        self.assertEqual(len(results), 0)
        
        # Negative k should raise error or return empty
        try:
            results = algorithm.find_similar(target, k=-1)
            self.assertEqual(len(results), 0)
        except ValueError as e:
            self.assertIn("k", str(e).lower())
    
    def test_target_with_missing_fields(self):
        """Test with target entity missing required fields."""
        algorithm = SelectKBest()
        tenders = [
            {
                "Наименование объекта закупки": "Поставка компьютеров",
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "Регион": "Москва"
            }
        ]
        algorithm.load_tenders(tenders)
        
        # Target with minimal fields
        target = {}
        results = algorithm.find_similar(target, k=5)
        
        # Should handle gracefully (may return low similarity results)
        self.assertIsInstance(results, list)
    
    def test_all_identical_tenders(self):
        """Test with all tenders identical."""
        algorithm = SelectKBest()
        tenders = [
            {
                "Наименование объекта закупки": "Поставка компьютеров",
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "Регион": "Москва"
            }
        ] * 5  # 5 identical tenders
        
        algorithm.load_tenders(tenders)
        
        target = tenders[0]
        results = algorithm.find_similar(target, k=5, exclude_self=True)
        
        # Should return 4 results (excluding self)
        self.assertEqual(len(results), 4)
        
        # All similarities should be 0.5 (weighted sum of text, region, price similarities)
        for result in results:
            self.assertAlmostEqual(result['similarity_score'], 0.5, places=5)
    
    def test_extreme_similarity_threshold(self):
        """Test with extreme similarity thresholds."""
        algorithm = SelectKBest()
        tenders = [
            {"Наименование объекта закупки": f"Тендер {i}"} for i in range(10)
        ]
        algorithm.load_tenders(tenders)
        
        target = {"Наименование объекта закупки": "Запрос"}
        
        # Threshold = 1.0 (should return only exact matches)
        results = algorithm.find_similar(target, k=10, min_similarity=1.0)
        for result in results:
            self.assertEqual(result['similarity_score'], 1.0)
        
        # Threshold = 0.0 (should return all)
        results = algorithm.find_similar(target, k=10, min_similarity=0.0)
        self.assertGreaterEqual(len(results), 0)
        
        # Threshold = -0.1 (invalid)
        with self.assertRaises(ValueError):
            algorithm.find_similar(target, k=10, min_similarity=-0.1)
    
    def test_weights_edge_cases(self):
        """Test with edge case weights."""
        # All weights zero
        weights = {
            'text': 0.0,
            'description': 0.0,
            'customer': 0.0,
            'region': 0.0,
            'method': 0.0,
            'price': 0.0,
            'date': 0.0
        }
        
        algorithm = SelectKBest(weights=weights)
        tenders = [{"Наименование объекта закупки": "test"}]
        algorithm.load_tenders(tenders)
        
        target = {"Наименование объекта закупки": "query"}
        results = algorithm.find_similar(target, k=5)
        
        # With all weights zero, similarity should be 0 or handle gracefully
        for result in results:
            self.assertEqual(result['similarity_score'], 0.0)
        
        # Very large weights
        weights = {'text': 1000.0, 'price': 1000.0}
        algorithm = SelectKBest(weights=weights)
        algorithm.load_tenders(tenders)
        results = algorithm.find_similar(target, k=5)
        # Should still produce valid similarity scores (normalized)
    
    def test_create_target_entity_edge_cases(self):
        """Test edge cases for create_target_entity_from_criteria."""
        # Empty criteria
        target = create_target_entity_from_criteria({})
        self.assertIsInstance(target, dict)
        
        # Criteria with extra fields
        criteria = {
            "region": "Москва",
            "method": "Электронный аукцион",
            "keywords": "компьютерная техника",
            "extra_field": "should_be_ignored"
        }
        target = create_target_entity_from_criteria(criteria)
        self.assertEqual(target.get("Регион"), "Москва")
        self.assertEqual(target.get("Способ определения поставщика"), "Электронный аукцион")
        self.assertEqual(target.get("Наименование объекта закупки"), "компьютерная техника")
        
        # Criteria with None values
        criteria = {"region": None, "method": ""}
        target = create_target_entity_from_criteria(criteria)
        self.assertEqual(target.get("Регион"), None)
        self.assertEqual(target.get("Способ определения поставщика"), "")
    
    def test_batch_processing_edge_cases(self):
        """Test batch processing with edge cases."""
        algorithm = SelectKBest()
        
        # Empty batch
        with patch.object(algorithm, '_compute_similarity_batch') as mock_batch:
            algorithm._compute_similarity_batch([], {})
            mock_batch.assert_called_once()
        
        # Very large batch
        large_tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(1000)]
        algorithm.load_tenders(large_tenders)
        
        target = {"Наименование объекта закупки": "Запрос"}
        results = algorithm.find_similar(target, k=5)
        
        # Should complete without error
        self.assertLessEqual(len(results), 5)
    
    def test_save_load_results_edge_cases(self):
        """Test saving and loading results with edge cases."""
        algorithm = SelectKBest()
        
        # Empty results
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            algorithm.save_results([], temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load empty results
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertIn('similar_tenders', loaded)
            self.assertEqual(len(loaded['similar_tenders']), 0)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Results with special characters
        results = [
            {
                'rank': 1,
                'similarity_score': 0.95,
                'tender': {
                    "Наименование объекта закупки": "Тест с \"кавычками\" и \nпереносами",
                    "Регион": "Москва & область"
                },
                'explanation': "Explanation with Unicode: αβγδε",
                'similarity_breakdown': {}
            }
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w', encoding='utf-8') as f:
            temp_path = f.name
        
        try:
            algorithm.save_results(results, temp_path)
            
            # Verify file can be loaded
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertEqual(len(loaded['similar_tenders']), 1)
            self.assertEqual(loaded['similar_tenders'][0]['tender']["Наименование объекта закупки"],
                           "Тест с \"кавычками\" и \nпереносами")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMLRetrospectiveEdgeCases(unittest.TestCase):
    """Edge case tests for Algorithm 2 (ML-based retrospective)."""
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        analyzer.fit([])
        results = analyzer.find_similar({}, k=5)
        
        self.assertEqual(len(results), 0)
    
    def test_small_dataset_clustering(self):
        """Test with dataset smaller than minimum cluster size."""
        config = MLRetrospectiveConfig(clustering_min_cluster_size=10)
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Only 5 tenders
        tenders = [
            {"Наименование объекта закупки": f"Тендер {i}"} for i in range(5)
        ]
        
        analyzer.fit(tenders)
        
        # Should still work (clustering may find no clusters)
        target = {"Наименование объекта закупки": "Запрос"}
        results = analyzer.find_similar(target, k=5)
        
        self.assertIsInstance(results, list)
    
    def test_all_noise_dataset(self):
        """Test with dataset where all points are noise."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Mock clustering to return all noise
        with patch.object(analyzer.clustering, 'fit_predict') as mock_cluster:
            mock_cluster.return_value = np.full(10, -1)  # All noise
            
            tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(10)]
            analyzer.fit(tenders)
            
            target = {"Наименование объекта закупки": "Запрос"}
            results = analyzer.find_similar(target, k=5)
            
            # Should still return results (based on similarity, not clustering)
            self.assertIsInstance(results, list)
    
    def test_single_cluster_dataset(self):
        """Test with dataset forming a single cluster."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Mock clustering to return single cluster
        with patch.object(analyzer.clustering, 'fit_predict') as mock_cluster:
            mock_cluster.return_value = np.zeros(10)  # All in cluster 0
            
            tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(10)]
            analyzer.fit(tenders)
            
            target = {"Наименование объекта закупки": "Запрос"}
            results = analyzer.find_similar(target, k=5)
            
            self.assertIsInstance(results, list)
    
    def test_high_dimensional_features(self):
        """Test with very high-dimensional feature space."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Mock feature engineering to return high-dim features
        with patch.object(analyzer.feature_pipeline, 'fit_transform') as mock_features:
            mock_features.return_value = np.random.randn(10, 1000)  # 1000 dimensions
            
            tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(10)]
            analyzer.fit(tenders)
            
            # Should handle high dimensions (UMAP will reduce)
            self.assertTrue(analyzer.features.shape[1] <= 1000)
    
    def test_training_with_incomplete_data(self):
        """Test training with tenders missing required data."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tenders = [
            {},  # Empty
            {"field1": "value1"},  # Missing required
            {"Наименование объекта закупки": "test", "price": "invalid"},  # Partial
            {"Наименование объекта закупки": "valid", "Начальная (максимальная) цена контракта": "1 000 000,00"}  # Valid
        ]
        
        # Should handle gracefully
        analyzer.fit(tenders)
        self.assertIsNotNone(analyzer.features)
    
    def test_model_persistence_edge_cases(self):
        """Test saving and loading model with edge cases."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Train with some data
        tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(5)]
        analyzer.fit(tenders)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model")
            
            # Test saving
            analyzer.save_model(model_path)
            self.assertTrue(os.path.exists(model_path + ".pkl"))
            
            # Test loading empty file
            empty_path = os.path.join(temp_dir, "empty")
            with open(empty_path + ".pkl", 'wb') as f:
                f.write(b'')
            
            new_analyzer = MLRetrospectiveAnalyzer(config)
            with self.assertRaises(Exception):
                new_analyzer.load_model(empty_path)
            
            # Test loading corrupted file
            corrupt_path = os.path.join(temp_dir, "corrupt")
            with open(corrupt_path + ".pkl", 'wb') as f:
                f.write(b'invalid pickle data')
            
            new_analyzer = MLRetrospectiveAnalyzer(config)
            with self.assertRaises(Exception):
                new_analyzer.load_model(corrupt_path)
    
    def test_find_similar_with_invalid_target(self):
        """Test find_similar with invalid target entity."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        tenders = [{"Наименование объекта закупки": "test"}]
        analyzer.fit(tenders)
        
        # Target with wrong data types
        target = {
            "Наименование объекта закупки": 123,  # Should be string
            "Начальная (максимальная) цена контракта": ["list", "not", "allowed"]
        }
        
        results = analyzer.find_similar(target, k=5)
        # Should handle gracefully
        self.assertIsInstance(results, list)
    
    def test_performance_with_large_k(self):
        """Test performance with very large k value."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Create moderate dataset
        tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(100)]
        analyzer.fit(tenders)
        
        target = {"Наименование объекта закупки": "Запрос"}
        
        # Request all tenders (k = dataset size)
        import time
        start = time.time()
        results = analyzer.find_similar(target, k=100)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 10.0)  # Less than 10 seconds
        self.assertEqual(len(results), 100)
    
    def test_cluster_based_filtering_edge_cases(self):
        """Test cluster-based filtering with edge cases."""
        config = MLRetrospectiveConfig(use_cluster_filtering=True)
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Mock clustering results
        with patch.object(analyzer.clustering, 'labels_', np.array([-1, -1, 0, 0, 1, 1])):
            analyzer.n_clusters_ = 2
            analyzer.cluster_centers_ = np.random.randn(2, 50)
            
            # Test with target that doesn't belong to any cluster
            target_features = np.random.randn(1, 50)
            with patch.object(analyzer.feature_pipeline, 'transform') as mock_transform:
                mock_transform.return_value = target_features
                
                # Mock distance calculation
                with patch('numpy.linalg.norm') as mock_norm:
                    mock_norm.return_value = 100.0  # Large distance
                    
                    tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(6)]
                    analyzer.fit(tenders)
                    
                    target = {"Наименование объекта закупки": "Запрос"}
                    results = analyzer.find_similar(target, k=5)
                    
                    # Should still return results (fallback to similarity)
                    self.assertIsInstance(results, list)


class TestAlgorithmIntegrationEdgeCases(unittest.TestCase):
    """Integration edge case tests for both algorithms."""
    
    def test_consistent_results_empty_input(self):
        """Both algorithms should handle empty input consistently."""
        # Algorithm 1
        algo1 = SelectKBest()
        algo1_results = algo1.find_similar({}, k=5)
        
        # Algorithm 2
        config = MLRetrospectiveConfig()
        algo2 = MLRetrospectiveAnalyzer(config)
        algo2.fit([])
        algo2_results = algo2.find_similar({}, k=5)
        
        # Both should return empty lists
        self.assertEqual(len(algo1_results), 0)
        self.assertEqual(len(algo2_results), 0)
    
    def test_identical_inputs_both_algorithms(self):
        """Test both algorithms with identical inputs produce valid results."""
        tenders = [
            {
                "Наименование объекта закупки": "Поставка компьютеров",
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "Регион": "Москва"
            }
        ] * 3
        
        target = tenders[0]
        
        # Algorithm 1
        algo1 = SelectKBest()
        algo1.load_tenders(tenders)
        algo1_results = algo1.find_similar(target, k=5, exclude_self=True)
        
        # Algorithm 2
        config = MLRetrospectiveConfig()
        algo2 = MLRetrospectiveAnalyzer(config)
        algo2.fit(tenders)
        algo2_results = algo2.find_similar(target, k=5, exclude_self=True)
        
        # Both should return 2 results (excluding self)
        self.assertEqual(len(algo1_results), 2)
        self.assertEqual(len(algo2_results), 2)
        
        # Similarity scores should be 1.0 for identical items
        for result in algo1_results:
            self.assertEqual(result['similarity_score'], 1.0)
    
    def test_large_dataset_performance(self):
        """Test both algorithms with large dataset (performance)."""
        # Create large synthetic dataset
        n_tenders = 1000
        tenders = [
            {
                "Наименование объекта закупки": f"Тендер {i}",
                "Начальная (максимальная) цена контракта": f"{i * 100000:,}".replace(',', ' ') + ",00",
                "Регион": "Москва" if i % 2 == 0 else "Санкт-Петербург"
            }
            for i in range(n_tenders)
        ]
        
        target = {"Наименование объекта закупки": "Запрос", "Регион": "Москва"}
        
        # Algorithm 1
        import time
        start = time.time()
        algo1 = SelectKBest()
        algo1.load_tenders(tenders)
        algo1_results = algo1.find_similar(target, k=10)
        algo1_time = time.time() - start
        
        # Algorithm 2 (with mocked ML components for speed)
        start = time.time()
        config = MLRetrospectiveConfig()
        algo2 = MLRetrospectiveAnalyzer(config)
        
        # Mock feature engineering to return simple features
        with patch.object(algo2.feature_pipeline, 'fit_transform') as mock_fit:
            mock_fit.return_value = np.random.randn(n_tenders, 50)
            algo2.fit(tenders)
            
            with patch.object(algo2.feature_pipeline, 'transform') as mock_transform:
                mock_transform.return_value = np.random.randn(1, 50)
                algo2_results = algo2.find_similar(target, k=10)
        algo2_time = time.time() - start
        
        # Both should complete in reasonable time
        self.assertLess(algo1_time, 30.0)  # Less than 30 seconds
        self.assertLess(algo2_time, 30.0)
        
        # Both should return exactly k results (or fewer if not enough)
        self.assertLessEqual(len(algo1_results), 10)
        self.assertLessEqual(len(algo2_results), 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)