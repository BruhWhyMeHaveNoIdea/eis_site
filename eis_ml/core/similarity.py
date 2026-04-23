"""
Similarity calculation functions for government procurement tender data.

This module provides specialized similarity metrics for different data types:
- Text similarity using embeddings
- Categorical similarity for regions, methods, etc.
- Numerical similarity for prices and financial data
- Temporal similarity for dates
- Composite weighted similarity scores

REFACTORED VERSION: Uses configurable field names, weights, and hierarchies
from config.settings instead of hardcoded values.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import re

from core.embeddings import EmbeddingGenerator, get_default_generator

# Import configuration system
try:
    from config.settings import get_config, get_field_mapping
    from core.preprocessing import get_region_hierarchy, get_procurement_method_groups
    _config = get_config()
except ImportError:
    # Fallback to old config module for backward compatibility
    import warnings
    warnings.warn("config.settings not found, using fallback configuration")
    from config import (
        DEFAULT_WEIGHTS, REGION_HIERARCHY, PROCUREMENT_METHOD_GROUPS,
        NUMERICAL_SIMILARITY_METHOD, TEMPORAL_SIMILARITY_METHOD,
        TEMPORAL_SCALE_DAYS
    )
    _config = None


def _get_config_value(key: str, default: Any = None) -> Any:
    """Get value from configuration system with fallback."""
    if _config:
        try:
            return _config.get(key, default)
        except (AttributeError, KeyError):
            pass
    return default


def _get_field_names(field_type: str) -> List[str]:
    """Get field names for a specific field type from configuration."""
    try:
        return get_field_mapping(field_type)
    except (NameError, AttributeError, KeyError):
        # Fallback to old constants
        if field_type == 'text':
            from config import TEXT_FIELDS
            return TEXT_FIELDS
        elif field_type == 'numeric':
            from config import NUMERIC_FIELDS
            return NUMERIC_FIELDS
        elif field_type == 'date':
            from config import DATE_FIELDS
            return DATE_FIELDS
        elif field_type == 'categorical':
            from config import CATEGORICAL_FIELDS
            return CATEGORICAL_FIELDS
        else:
            raise ValueError(f"Unknown field type: {field_type}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity in range [-1, 1] (typically [0, 1] for normalized vectors)
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector shapes don't match: {vec1.shape} vs {vec2.shape}")
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def text_similarity(text1: str, 
                   text2: str,
                   embedding_generator: Optional[EmbeddingGenerator] = None,
                   use_cache: bool = True) -> float:
    """
    Compute text similarity using embeddings.
    
    Args:
        text1: First text
        text2: Second text
        embedding_generator: Embedding generator instance
        use_cache: Whether to use caching for embeddings
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    if embedding_generator is None:
        embedding_generator = get_default_generator()
    
    # Generate embeddings
    embeddings = embedding_generator.encode([text1, text2])
    
    if len(embeddings) < 2:
        return 0.0
    
    # Compute cosine similarity
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    
    # Clip to [0, 1] range (cosine similarity can be negative for non-normalized vectors)
    return max(0.0, similarity)


def categorical_similarity(cat1: str, 
                         cat2: str,
                         hierarchical_mapping: Optional[Dict[str, List[str]]] = None) -> float:
    """
    Compute similarity between categorical values.
    
    Args:
        cat1: First categorical value
        cat2: Second categorical value
        hierarchical_mapping: Optional mapping for hierarchical categories
                             (e.g., region -> federal district)
        
    Returns:
        Similarity score: 1.0 for exact match, 0.5 for hierarchical match, 0.0 otherwise
    """
    if not cat1 or not cat2:
        return 0.0
    
    # Exact match
    if cat1 == cat2:
        return 1.0
    
    # Case-insensitive match
    if cat1.lower() == cat2.lower():
        return 1.0
    
    # Hierarchical similarity check
    if hierarchical_mapping:
        # Check if categories belong to same parent group
        for parent, children in hierarchical_mapping.items():
            if cat1 in children and cat2 in children:
                return 0.5  # Same parent group
    
    # Partial string match (for similar categories)
    if cat1 in cat2 or cat2 in cat1:
        return 0.3
    
    # No similarity
    return 0.0


def numerical_similarity(num1: Union[float, int, str],
                        num2: Union[float, int, str],
                        scale: float = 1000000.0,
                        method: str = 'gaussian') -> float:
    """
    Compute similarity between numerical values.
    
    Args:
        num1: First numerical value (can be string with Russian formatting)
        num2: Second numerical value (can be string with Russian formatting)
        scale: Scale parameter for similarity kernel
        method: Similarity method: 'gaussian', 'exponential', 'inverse'
        
    Returns:
        Similarity score between 0 and 1
    """
    # Parse if strings
    from core.preprocessing import parse_russian_number
    
    if isinstance(num1, str):
        num1 = parse_russian_number(num1)
    if isinstance(num2, str):
        num2 = parse_russian_number(num2)
    
    # Convert to float
    try:
        num1 = float(num1)
        num2 = float(num2)
    except (ValueError, TypeError):
        return 0.0
    
    # Absolute difference
    diff = abs(num1 - num2)
    
    # Handle zero scale
    if scale <= 0:
        scale = 1.0
    
    if method == 'gaussian':
        # Gaussian similarity kernel
        return np.exp(-(diff ** 2) / (2 * scale ** 2))
    
    elif method == 'exponential':
        # Exponential decay
        return np.exp(-diff / scale)
    
    elif method == 'inverse':
        # Inverse similarity
        return 1.0 / (1.0 + diff / scale)
    
    else:
        raise ValueError(f"Unknown numerical similarity method: {method}")


def temporal_similarity(date1: Union[str, datetime],
                       date2: Union[str, datetime],
                       time_scale: float = 30 * 24 * 3600,  # 30 days in seconds
                       method: str = 'exponential') -> float:
    """
    Compute temporal similarity between dates.
    
    Args:
        date1: First date (string in Russian format or datetime object)
        date2: Second date (string in Russian format or datetime object)
        time_scale: Time scale parameter in seconds
        method: Similarity method: 'exponential', 'gaussian', 'linear'
        
    Returns:
        Similarity score between 0 and 1
    """
    from core.preprocessing import parse_russian_date
    
    # Parse if strings
    if isinstance(date1, str):
        date1 = parse_russian_date(date1)
    if isinstance(date2, str):
        date2 = parse_russian_date(date2)
    
    # Check if dates are valid
    if date1 is None or date2 is None:
        return 0.0
    
    # Time difference in seconds
    diff_seconds = abs((date1 - date2).total_seconds())
    
    if method == 'exponential':
        # Exponential decay
        return np.exp(-diff_seconds / time_scale)
    
    elif method == 'gaussian':
        # Gaussian kernel
        return np.exp(-(diff_seconds ** 2) / (2 * time_scale ** 2))
    
    elif method == 'linear':
        # Linear decay (clipped at time_scale)
        if diff_seconds >= time_scale:
            return 0.0
        else:
            return 1.0 - (diff_seconds / time_scale)
    
    else:
        raise ValueError(f"Unknown temporal similarity method: {method}")


def region_similarity(region1: str, region2: str) -> float:
    """
    Specialized similarity for Russian regions.
    
    Args:
        region1: First region name
        region2: Second region name
        
    Returns:
        Similarity score between 0 and 1
    """
    if not region1 or not region2:
        return 0.0
    
    # Exact match
    if region1 == region2:
        return 1.0
    
    # Case-insensitive match
    if region1.lower() == region2.lower():
        return 1.0
    
    # Get region hierarchy from configuration
    try:
        region_hierarchy = get_region_hierarchy()
    except (NameError, AttributeError):
        # Fallback to old constant
        try:
            from config import REGION_HIERARCHY
            region_hierarchy = REGION_HIERARCHY
        except ImportError:
            region_hierarchy = {}
    
    # Check if regions are in the same federal district
    for district, regions in region_hierarchy.items():
        if region1 in regions and region2 in regions:
            return 0.5
    
    # Common abbreviations and variations (configurable in the future)
    region_variations = {
        'Москва': ['г. Москва', 'Москва г.', 'Москва город'],
        'Санкт-Петербург': ['г. Санкт-Петербург', 'Санкт-Петербург г.', 'СПб', 'Петербург'],
        'Московская область': ['МО', 'Московская обл.', 'Подмосковье'],
        'Ленинградская область': ['ЛО', 'Ленинградская обл.'],
    }
    
    # Check if regions are variations of the same region
    for base, variations in region_variations.items():
        if (region1 == base or region1 in variations) and (region2 == base or region2 in variations):
            return 0.8
    
    # No similarity
    return 0.0


def procurement_method_similarity(method1: str, method2: str) -> float:
    """
    Specialized similarity for procurement methods.
    
    Args:
        method1: First procurement method
        method2: Second procurement method
        
    Returns:
        Similarity score between 0 and 1
    """
    if not method1 or not method2:
        return 0.0
    
    # Exact match
    if method1 == method2:
        return 1.0
    
    # Get method groups from configuration
    try:
        method_groups = get_procurement_method_groups()
    except (NameError, AttributeError):
        # Fallback to old constant
        try:
            from config import PROCUREMENT_METHOD_GROUPS
            method_groups = PROCUREMENT_METHOD_GROUPS
        except ImportError:
            method_groups = {}
    
    # Check if methods are in the same group
    for group, methods in method_groups.items():
        if method1 in methods and method2 in methods:
            return 0.8
    
    # Partial match (configurable)
    # This uses direct string matching which should be replaced with embedding-based similarity
    # For now, keep as is but mark for future refactoring
    if any(word in method2 for word in method1.split()) or any(word in method1 for word in method2.split()):
        return 0.3
    
    return 0.0


def composite_similarity(tender1: Dict[str, Any],
                        tender2: Dict[str, Any],
                        weights: Optional[Dict[str, float]] = None,
                        embedding_generator: Optional[EmbeddingGenerator] = None) -> Dict[str, Any]:
    """
    Compute composite similarity score between two tenders.
    
    Args:
        tender1: First tender dictionary (preprocessed)
        tender2: Second tender dictionary (preprocessed)
        weights: Dictionary of weights for each similarity component
        embedding_generator: Embedding generator for text similarity
        
    Returns:
        Dictionary with overall score and breakdown
    """
    # Get default weights from configuration
    if weights is None:
        weights = _get_config_value('weights', {
            'text': 0.3,
            'region': 0.1,
            'method': 0.05,
            'price': 0.1,
            'date': 0.05,
            'customer': 0.15,
            'description': 0.2,
            'other': 0.05
        })
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    similarity_breakdown = {}
    
    # Get field names from configuration
    text_fields = _get_field_names('text')
    numeric_fields = _get_field_names('numeric')
    date_fields = _get_field_names('date')
    categorical_fields = _get_field_names('categorical')
    
    # 1. Text similarity (procurement name) - use first text field as primary
    primary_text_field = text_fields[0] if text_fields else 'Наименование объекта закупки'
    text1 = tender1.get(f'{primary_text_field}_cleaned', tender1.get(primary_text_field, ''))
    text2 = tender2.get(f'{primary_text_field}_cleaned', tender2.get(primary_text_field, ''))
    similarity_breakdown['text_similarity'] = text_similarity(
        text1, text2, embedding_generator
    )
    
    # 2. Region similarity
    region_field = 'Регион'  # Could be configurable
    region1 = tender1.get(region_field, '')
    region2 = tender2.get(region_field, '')
    similarity_breakdown['region_similarity'] = region_similarity(region1, region2)
    
    # 3. Procurement method similarity
    method_field = 'Способ определения поставщика'  # Could be configurable
    method1 = tender1.get(method_field, '')
    method2 = tender2.get(method_field, '')
    similarity_breakdown['method_similarity'] = procurement_method_similarity(method1, method2)
    
    # 4. Price similarity - use first numeric field as price
    primary_numeric_field = numeric_fields[0] if numeric_fields else 'Начальная (максимальная) цена контракта'
    price1 = tender1.get(f'{primary_numeric_field}_parsed', tender1.get(primary_numeric_field, '0'))
    price2 = tender2.get(f'{primary_numeric_field}_parsed', tender2.get(primary_numeric_field, '0'))
    
    # Determine scale based on price magnitude
    try:
        price1_val = float(price1) if not isinstance(price1, str) else float(parse_russian_number(price1))
        price2_val = float(price2) if not isinstance(price2, str) else float(parse_russian_number(price2))
        avg_price = (abs(price1_val) + abs(price2_val)) / 2
        scale = max(100000, avg_price * 0.5)  # Dynamic scale
    except (ValueError, TypeError):
        scale = 1000000.0
    
    # Get numerical similarity method from configuration
    numerical_method = _get_config_value('similarity.numerical.method', 'gaussian')
    similarity_breakdown['price_similarity'] = numerical_similarity(
        price1, price2, scale=scale, method=numerical_method
    )
    
    # 5. Date similarity (use first date field)
    primary_date_field = date_fields[0] if date_fields else 'Дата публикации'
    date1 = tender1.get(f'{primary_date_field}_parsed', tender1.get(primary_date_field, ''))
    date2 = tender2.get(f'{primary_date_field}_parsed', tender2.get(primary_date_field, ''))
    
    # Get temporal similarity parameters from configuration
    temporal_method = _get_config_value('similarity.temporal.method', 'exponential')
    temporal_scale_days = _get_config_value('similarity.temporal.scale_days', 90)
    time_scale = temporal_scale_days * 24 * 3600  # Convert days to seconds
    
    similarity_breakdown['date_similarity'] = temporal_similarity(
        date1, date2, time_scale=time_scale, method=temporal_method
    )
    
    # 6. Customer similarity
    customer_field = 'Заказчик'  # Could be configurable
    customer1 = tender1.get(f'{customer_field}_cleaned', tender1.get(customer_field, ''))
    customer2 = tender2.get(f'{customer_field}_cleaned', tender2.get(customer_field, ''))
    similarity_breakdown['customer_similarity'] = text_similarity(
        customer1, customer2, embedding_generator
    )
    
    # 7. Description similarity (use second text field if available)
    description_field = text_fields[1] if len(text_fields) > 1 else 'Наименование закупки'
    desc1 = tender1.get(f'{description_field}_cleaned', tender1.get(description_field, ''))
    desc2 = tender2.get(f'{description_field}_cleaned', tender2.get(description_field, ''))
    similarity_breakdown['description_similarity'] = text_similarity(
        desc1, desc2, embedding_generator
    )
    
    # 8. Other categorical similarities
    other_similarities = []
    
    # Process other categorical fields (excluding region and method already handled)
    other_categorical_fields = [f for f in categorical_fields 
                               if f not in ['Регион', 'Способ определения поставщика']]
    
    for field in other_categorical_fields:
        val1 = tender1.get(field, '')
        val2 = tender2.get(field, '')
        if val1 and val2:
            # Use categorical similarity for other fields
            sim = categorical_similarity(val1, val2)
            other_similarities.append(sim)
    
    # Average of other similarities
    similarity_breakdown['other_similarity'] = (
        np.mean(other_similarities) if other_similarities else 0.0
    )
    
    # Compute weighted overall score
    overall_score = 0.0
    for component, weight in weights.items():
        component_key = f'{component}_similarity'
        if component_key in similarity_breakdown:
            overall_score += similarity_breakdown[component_key] * weight
        elif component == 'other' and 'other_similarity' in similarity_breakdown:
            overall_score += similarity_breakdown['other_similarity'] * weight
    
    # Ensure score is in [0, 1]
    overall_score = max(0.0, min(1.0, overall_score))
    
    return {
        'overall_score': overall_score,
        'breakdown': similarity_breakdown,
        'weights': weights
    }


def batch_similarity(tender: Dict[str, Any],
                    tenders: List[Dict[str, Any]],
                    weights: Optional[Dict[str, float]] = None,
                    embedding_generator: Optional[EmbeddingGenerator] = None,
                    n_jobs: int = 1) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """
    Compute similarity between a target tender and a list of tenders.
    
    Args:
        tender: Target tender
        tenders: List of tenders to compare against
        weights: Similarity weights
        embedding_generator: Embedding generator
        n_jobs: Number of parallel jobs (currently not implemented)
        
    Returns:
        List of tuples (score, tender, breakdown) sorted by score descending
    """
    results = []
    
    # Get ID field from configuration
    id_field = _get_config_value('field_mappings.id_field', 'Идентификационный код закупки (ИКЗ)')
    
    for other_tender in tenders:
        # Skip comparing tender to itself (based on ID if available)
        tender_id1 = tender.get(id_field, '')
        tender_id2 = other_tender.get(id_field, '')
        
        if tender_id1 and tender_id2 and tender_id1 == tender_id2:
            continue
        
        similarity_result = composite_similarity(
            tender, other_tender, weights, embedding_generator
        )
        
        results.append((
            similarity_result['overall_score'],
            other_tender,
            similarity_result['breakdown']
        ))
    
    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    
    return results


# ============================================================================
# Configuration-aware helper functions
# ============================================================================

def get_default_weights() -> Dict[str, float]:
    """
    Get default similarity weights from configuration.
    
    Returns:
        Dictionary of component weights
    """
    return _get_config_value('weights', {
        'text': 0.3,
        'region': 0.1,
        'method': 0.05,
        'price': 0.1,
        'date': 0.05,
        'customer': 0.15,
        'description': 0.2,
        'other': 0.05
    })


def get_numerical_similarity_params() -> Dict[str, Any]:
    """
    Get numerical similarity parameters from configuration.
    
    Returns:
        Dictionary with method, scale_auto, scale_factor
    """
    return _get_config_value('similarity.numerical', {
        'method': 'gaussian',
        'scale_auto': True,
        'scale_factor': 0.5
    })


def get_temporal_similarity_params() -> Dict[str, Any]:
    """
    Get temporal similarity parameters from configuration.
    
    Returns:
        Dictionary with method, scale_days
    """
    return _get_config_value('similarity.temporal', {
        'method': 'exponential',
        'scale_days': 90
    })


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    from core.preprocessing import preprocess_tender
    
    # Example tenders
    tender1 = {
        "Наименование объекта закупки": "Поставка компьютерной техники",
        "Начальная (максимальная) цена контракта": "1 250 000,00",
        "Дата публикации": "15.03.2024 10:30",
        "Способ определения поставщика": "Электронный аукцион",
        "Регион": "Москва",
        "Заказчик": "Министерство образования",
        "Наименование закупки": "Закупка компьютерного оборудования",
        "Валюта": "RUB",
        "Источник финансирования": "Федеральный бюджет"
    }
    
    tender2 = {
        "Наименование объекта закупки": "Поставка оргтехники и компьютеров",
        "Начальная (максимальная) цена контракта": "1 100 000,00",
        "Дата публикации": "20.03.2024 14:00",
        "Способ определения поставщика": "Электронный аукцион",
        "Регион": "Московская область",
        "Заказчик": "Министерство образования и науки",
        "Наименование закупки": "Приобретение компьютерной техники для учреждения",
        "Валюта": "RUB",
        "Источник финансирования": "Федеральный бюджет"
    }
    
    # Preprocess tenders
    tender1_processed = preprocess_tender(tender1)
    tender2_processed = preprocess_tender(tender2)
    
    print("Tender 1 processed keys:", list(tender1_processed.keys())[:10])
    print("Tender 2 processed keys:", list(tender2_processed.keys())[:10])
    
    # Compute similarity
    similarity_result = composite_similarity(tender1_processed, tender2_processed)
    
    print("\nSimilarity Analysis:")
    print(f"Overall score: {similarity_result['overall_score']:.4f}")
    print("\nBreakdown:")
    for component, score in similarity_result['breakdown'].items():
        print(f"  {component}: {score:.4f}")
    
    # Test individual similarity functions
    print("\nIndividual similarity tests:")
    print(f"Region similarity: {region_similarity('Москва', 'Московская область'):.4f}")
    print(f"Method similarity: {procurement_method_similarity('Электронный аукцион', 'Аукцион'):.4f}")
    
    price_sim = numerical_similarity("1 000 000,00", "1 200 000,00", scale=500000)
    print(f"Price similarity: {price_sim:.4f}")
    
    date_sim = temporal_similarity("15.03.2024 10:30", "20.03.2024 14:00", time_scale=30*24*3600)
    print(f"Date similarity: {date_sim:.4f}")
    
    # Show configuration values
    print(f"\nConfiguration check:")
    print(f"Default weights: {get_default_weights()}")
    print(f"Numerical params: {get_numerical_similarity_params()}")
    print(f"Temporal params: {get_temporal_similarity_params()}")