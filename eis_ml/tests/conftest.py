"""
Pytest fixtures for comprehensive test suite.

This module provides reusable fixtures for testing the retrospective analysis system,
including synthetic tender data, algorithm instances, and configuration objects.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import modules
from core.preprocessing import preprocess_tender
from core.embeddings import EmbeddingGenerator
from select_k_best import SelectKBest
from ml_retrospective import MLRetrospectiveAnalyzer, MLRetrospectiveConfig
from ml.feature_engineering import FeatureEngineeringPipeline, FeatureConfig
from ml.clustering import TenderClustering, ClusteringConfig
from ml.similarity_learning import ContrastiveLearner, ContrastiveLearningConfig
from ml.faiss_index import FaissIndex, FaissConfig


@pytest.fixture
def sample_tender():
    """Return a single sample tender with typical Russian procurement data."""
    return {
        "Идентификационный код закупки (ИКЗ)": "0373200004524000001",
        "Наименование объекта закупки": "Поставка компьютерной техники для государственных учреждений",
        "Наименование закупки": "Закупка компьютерного оборудования",
        "Начальная (максимальная) цена контракта": "1 250 000,00",
        "Дата публикации": "15.03.2024 10:30",
        "Дата окончания подачи заявок": "30.03.2024 18:00",
        "Способ определения поставщика": "Электронный аукцион",
        "Регион": "Москва",
        "Заказчик": "Министерство образования Российской Федерации",
        "Требования к участникам": "Наличие опыта выполнения аналогичных контрактов",
        "Критерии оценки заявок": "Цена, квалификация участника, сроки выполнения",
        "Валюта": "Рубль",
        "Источник финансирования": "Федеральный бюджет",
        "Категория": "Техника и оборудование"
    }


@pytest.fixture
def sample_tenders():
    """Return a list of sample tenders with variation."""
    base_tender = sample_tender()
    
    tenders = []
    for i in range(10):
        tender = base_tender.copy()
        tender["Идентификационный код закупки (ИКЗ)"] = f"TEST{i:03d}"
        tender["Наименование объекта закупки"] = f"Поставка оборудования типа {i}"
        tender["Начальная (максимальная) цена контракта"] = f"{1000000 + i * 50000:,}".replace(',', ' ').replace('.', ',') + ",00"
        tender["Дата публикации"] = f"{(i+1):02d}.03.2024 10:00"
        tender["Регион"] = "Москва" if i < 5 else "Санкт-Петербург"
        tender["Способ определения поставщика"] = "Электронный аукцион" if i < 7 else "Конкурс"
        tenders.append(tender)
    
    return tenders


@pytest.fixture
def edge_case_tenders():
    """Return tenders with edge case values for testing."""
    return [
        # Empty tender
        {},
        
        # Tender with only required fields
        {
            "Идентификационный код закупки (ИКЗ)": "MINIMAL",
            "Наименование объекта закупки": "Минимальный тендер"
        },
        
        # Tender with extreme values
        {
            "Идентификационный код закупки (ИКЗ)": "EXTREME",
            "Наименование объекта закупки": "A" * 1000,  # Very long
            "Начальная (максимальная) цена контракта": "9 999 999 999 999,99",  # Very large
            "Регион": "",  # Empty string
            "Дата публикации": "01.01.1970 00:00"  # Very old date
        },
        
        # Tender with special characters
        {
            "Идентификационный код закупки (ИКЗ)": "SPECIAL",
            "Наименование объекта закупки": "Тест с \"кавычками\", & амперсандами, <тегами>",
            "Начальная (максимальная) цена контракта": "1,5",  # Comma decimal
            "Регион": "Москва & область",
            "Заказчик": "ООО \"Рога и копыта\""
        },
        
        # Tender with Unicode
        {
            "Идентификационный код закупки (ИКЗ)": "UNICODE",
            "Наименование объекта закупки": "Поставка αβγδε equipment με Unicode",
            "Регион": "Санкт-Петербург",
            "Заказчик": "Компания © 2024"
        },
        
        # Tender with numeric-like text
        {
            "Идентификационный код закупки (ИКЗ)": "NUMTEXT",
            "Наименование объекта закупки": "1234567890",
            "Начальная (максимальная) цена контракта": "ноль рублей",  # Not a number
            "Регион": "77 регион"  # Number in text
        }
    ]


@pytest.fixture
def preprocessed_tenders(sample_tenders):
    """Return preprocessed sample tenders."""
    return [preprocess_tender(tender) for tender in sample_tenders]


@pytest.fixture
def mock_embedding_generator():
    """Return a mock embedding generator to avoid model downloads."""
    with patch.object(EmbeddingGenerator, '_initialize_model'):
        generator = EmbeddingGenerator()
        generator.model = MagicMock()
        generator.dimension = 384
        generator.encode = MagicMock(return_value=np.random.randn(10, 384))
        return generator


@pytest.fixture
def select_k_best_algorithm(mock_embedding_generator):
    """Return a SelectKBest instance with mock embeddings."""
    algorithm = SelectKBest(embedding_generator=mock_embedding_generator)
    return algorithm


@pytest.fixture
def ml_retrospective_analyzer():
    """Return an MLRetrospectiveAnalyzer with mocked ML components."""
    # Mock all heavy dependencies
    with patch.object(FeatureEngineeringPipeline, 'fit_transform') as mock_feat, \
         patch.object(TenderClustering, 'fit_predict') as mock_cluster, \
         patch.object(ContrastiveLearner, 'train') as mock_train, \
         patch.object(FaissIndex, 'build') as mock_faiss:
        
        # Setup mocks
        mock_feat.return_value = np.random.randn(10, 50)
        mock_cluster.return_value = np.random.randint(-1, 3, 10)
        mock_train.return_value = None
        
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        # Mock internal attributes
        analyzer.features = np.random.randn(10, 50)
        analyzer.clustering_labels = np.random.randint(-1, 3, 10)
        analyzer.n_clusters_ = 3
        
        return analyzer


@pytest.fixture
def feature_engineering_pipeline():
    """Return a FeatureEngineeringPipeline with mocked embeddings."""
    config = FeatureConfig()
    
    with patch.object(FeatureEngineeringPipeline, '_initialize_embedding_generator'):
        pipeline = FeatureEngineeringPipeline(config)
        pipeline.embedding_generator = MagicMock()
        pipeline.embedding_generator.encode.return_value = np.random.randn(5, 384)
        
        return pipeline


@pytest.fixture
def clustering_model():
    """Return a TenderClustering instance."""
    config = ClusteringConfig(
        umap_n_components=10,
        hdbscan_min_cluster_size=5
    )
    return TenderClustering(config)


@pytest.fixture
def faiss_index():
    """Return a FaissIndex instance."""
    config = FaissConfig(dimension=128)
    return FaissIndex(config)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {
            "tenders": [
                {
                    "Идентификационный код закупки (ИКЗ)": "TEMP001",
                    "Наименование объекта закупки": "Временный тендер",
                    "Начальная (максимальная) цена контракта": "500 000,00"
                }
            ]
        }
        json.dump(data, f, ensure_ascii=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_model_file():
    """Create a temporary file for model saving."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture(params=[0, 1, 5, 10, 100])
def dataset_size(request):
    """Parameterized fixture for different dataset sizes."""
    return request.param


@pytest.fixture(params=[0, 1, 5, 10, -1])
def k_value(request):
    """Parameterized fixture for different k values."""
    return request.param


@pytest.fixture(params=[0.0, 0.5, 0.9, 1.0, -0.1])
def similarity_threshold(request):
    """Parameterized fixture for different similarity thresholds."""
    return request.param


@pytest.fixture
def synthetic_features():
    """Generate synthetic feature matrix for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 100
    
    # Create features with some cluster structure
    features = np.random.randn(n_samples, n_features)
    
    # Add cluster structure
    features[:20] += 2.0  # Cluster 1
    features[20:35] -= 1.5  # Cluster 2
    features[35:45] += 0.5  # Cluster 3
    # Last 5 are noise
    
    return features


@pytest.fixture
def synthetic_labels():
    """Generate synthetic cluster labels."""
    labels = np.full(50, -1)  # Start as noise
    labels[:20] = 0  # Cluster 0
    labels[20:35] = 1  # Cluster 1
    labels[35:45] = 2  # Cluster 2
    # Last 5 remain as noise (-1)
    
    return labels


@pytest.fixture
def malformed_inputs():
    """Return various malformed inputs for security testing."""
    return [
        # Path traversal
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config",
        
        # SQL injection
        "'; DROP TABLE tenders; --",
        "' OR '1'='1",
        
        # Command injection
        "$(rm -rf /)",
        "| cat /etc/passwd",
        
        # Large inputs
        "A" * 1000000,  # 1MB string
        
        # Special characters
        "\x00\x01\x02\x03",  # Binary data
        "<script>alert('xss')</script>",
        
        # JSON injection
        '{"__class__": "os.system", "__args__": ["ls"]}',
    ]


@pytest.fixture
def datetime_range():
    """Return a range of datetime objects for temporal testing."""
    base_date = datetime(2024, 3, 15, 10, 30)
    return [
        base_date,
        base_date + timedelta(days=1),
        base_date + timedelta(days=7),
        base_date + timedelta(days=30),
        base_date + timedelta(days=365),
    ]


@pytest.fixture
def russian_number_strings():
    """Return Russian-formatted number strings for testing."""
    return [
        "1 250 000,00",  # Standard
        "500,50",  # Decimal
        "1000",  # No decimal
        "1,5",  # Comma decimal
        "0",  # Zero
        "ноль",  # Text zero
        "1 000 000 000,00",  # Large
        "-500,00",  # Negative
        "1.250.000,00",  # Dots as separators (non-standard)
    ]


@pytest.fixture
def russian_date_strings():
    """Return Russian date strings for testing."""
    return [
        "15.03.2024 10:30",  # Standard with time
        "15.03.2024",  # Without time
        "01.01.1970 00:00",  # Epoch
        "31.12.2024 23:59",  # End of year
        "invalid date",  # Invalid
        "2024-03-15 10:30:00",  # ISO format
        "15/03/2024",  # Wrong separator
    ]


# Parameterized test data for preprocessing
@pytest.fixture(params=[
    ("1 250 000,00", 1250000.0),
    ("500,50", 500.5),
    ("1000", 1000.0),
    ("", 0.0),
    ("ноль", 0.0),
    ("1,5", 1.5),
])
def number_parsing_test_case(request):
    """Parameterized fixture for number parsing tests."""
    return request.param


# Parameterized test data for similarity
@pytest.fixture(params=[
    # (vec1, vec2, expected_similarity)
    ([1, 0, 0], [1, 0, 0], 1.0),
    ([1, 0, 0], [0, 1, 0], 0.0),
    ([1, 0, 0], [-1, 0, 0], -1.0),
    ([0, 0, 0], [1, 0, 0], 0.0),
])
def cosine_similarity_test_case(request):
    """Parameterized fixture for cosine similarity tests."""
    vec1, vec2, expected = request.param
    return np.array(vec1), np.array(vec2), expected