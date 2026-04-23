"""
Preprocessing utilities for government procurement tender data.

This module handles cleaning, normalization, and feature extraction from
Russian-language tender JSON data with specific formatting conventions.

REFACTORED VERSION: Uses configurable field names from config.settings
instead of hardcoded values. All field mappings are now configurable.
"""

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Import configuration system
try:
    from config.settings import get_config, get_field_mapping
    _config = get_config()
except ImportError:
    # Fallback to old config module for backward compatibility
    import warnings
    warnings.warn("config.settings not found, using fallback configuration")
    from config import (
        TEXT_FIELDS, NUMERIC_FIELDS, DATE_FIELDS, CATEGORICAL_FIELDS,
        TEXT_FIELD_WEIGHTS
    )
    _config = None


def _get_field_names(field_type: str) -> List[str]:
    """Get field names for a specific field type from configuration."""
    if _config:
        try:
            return get_field_mapping(field_type)
        except (AttributeError, KeyError):
            pass
    
    # Fallback to old constants
    if field_type == 'text':
        return TEXT_FIELDS if 'TEXT_FIELDS' in globals() else []
    elif field_type == 'numeric':
        return NUMERIC_FIELDS if 'NUMERIC_FIELDS' in globals() else []
    elif field_type == 'date':
        return DATE_FIELDS if 'DATE_FIELDS' in globals() else []
    elif field_type == 'categorical':
        return CATEGORICAL_FIELDS if 'CATEGORICAL_FIELDS' in globals() else []
    else:
        raise ValueError(f"Unknown field type: {field_type}")


def parse_russian_number(num_str: str) -> float:
    """
    Parse Russian-formatted number string to float.
    
    Russian format uses spaces as thousand separators and comma as decimal separator.
    Example: "1 250 000,00" -> 1250000.0
    
    Args:
        num_str: String containing formatted number
        
    Returns:
        Float representation of the number
        
    Raises:
        ValueError: If the string cannot be parsed
    """
    if not num_str or not isinstance(num_str, str):
        return 0.0
    
    # Remove all spaces (thousand separators) and replace comma with dot
    cleaned = num_str.strip().replace(' ', '').replace(',', '.')
    
    # Handle empty strings after cleaning
    if not cleaned:
        return 0.0
    
    try:
        return float(cleaned)
    except ValueError:
        # Try to extract numeric part using regex
        match = re.search(r'[-+]?\d*[\.,]?\d+', cleaned)
        if match:
            cleaned_num = match.group().replace(',', '.')
            return float(cleaned_num)
        else:
            return 0.0


def parse_russian_date(date_str: str) -> Optional[datetime]:
    """
    Parse Russian date string in format ДД.ММ.ГГГГ ЧЧ:ММ.
    
    Args:
        date_str: Date string in format "ДД.ММ.ГГГГ ЧЧ:ММ"
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    # Handle empty strings
    if not date_str:
        return None
    
    try:
        return datetime.strptime(date_str, "%d.%m.%Y %H:%M")
    except ValueError:
        # Try alternative formats
        try:
            # Try without time
            return datetime.strptime(date_str, "%d.%m.%Y")
        except ValueError:
            # Try with different separators
            date_str = re.sub(r'[^\d\.\:\s]', '', date_str)
            patterns = [
                "%d.%m.%Y %H:%M",
                "%d.%m.%Y",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]
            for pattern in patterns:
                try:
                    return datetime.strptime(date_str, pattern)
                except ValueError:
                    continue
            return None


def clean_russian_text(text: str, 
                       lowercase: bool = True,
                       remove_punctuation: bool = True,
                       remove_stopwords: bool = False) -> str:
    """
    Clean and normalize Russian text.
    
    Args:
        text: Input text in Russian
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        remove_stopwords: Remove common Russian stopwords (basic implementation)
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove punctuation if requested
    if remove_punctuation:
        # Keep some punctuation that might be meaningful in procurement context
        # like numbers with commas, hyphens in compound words
        text = re.sub(r'[^\w\s\d\-\.,]', ' ', text)
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
    
    # Basic Russian stopwords list (can be expanded)
    if remove_stopwords:
        stopwords = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
            'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
            'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
            'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж',
            'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
            'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее',
            'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
            'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше',
            'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
            'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им',
            'более', 'всегда', 'конечно', 'всю', 'между'
        }
        words = text.split()
        words = [w for w in words if w not in stopwords]
        text = ' '.join(words)
    
    return text.strip()


def normalize_numerical_value(value: Union[str, int, float], 
                             method: str = 'minmax',
                             stats: Optional[Dict[str, Any]] = None) -> float:
    """
    Normalize numerical value using specified method.
    
    Args:
        value: Numerical value (can be string with Russian formatting)
        method: Normalization method: 'minmax', 'zscore', 'log', 'none'
        stats: Dictionary with precomputed statistics (min, max, mean, std)
               Required for 'minmax' and 'zscore' methods
               
    Returns:
        Normalized float value
    """
    # Parse if string
    if isinstance(value, str):
        num_value = parse_russian_number(value)
    else:
        num_value = float(value)
    
    if method == 'none':
        return num_value
    
    if method == 'log':
        # Apply log transformation (add 1 to handle zeros)
        return np.log1p(max(0, num_value))
    
    if stats is None:
        # If no stats provided, return raw value
        return num_value
    
    if method == 'minmax':
        min_val = stats.get('min', 0)
        max_val = stats.get('max', 1)
        if max_val - min_val == 0:
            return 0.0
        return (num_value - min_val) / (max_val - min_val)
    
    if method == 'zscore':
        mean_val = stats.get('mean', 0)
        std_val = stats.get('std', 1)
        if std_val == 0:
            return 0.0
        return (num_value - mean_val) / std_val
    
    return num_value


def extract_date_features(date_str: str, 
                         reference_date: Optional[datetime] = None) -> Dict[str, float]:
    """
    Extract temporal features from date string.
    
    Args:
        date_str: Date string in Russian format
        reference_date: Reference date for computing time deltas
        
    Returns:
        Dictionary with extracted features:
        - timestamp: Unix timestamp
        - day_of_week: 0-6 (Monday=0)
        - month: 1-12
        - quarter: 1-4
        - year: year
        - days_from_reference: days from reference date (if provided)
    """
    dt = parse_russian_date(date_str)
    if dt is None:
        return {
            'timestamp': 0.0,
            'day_of_week': 0.0,
            'month': 0.0,
            'quarter': 0.0,
            'year': 0.0,
            'days_from_reference': 0.0
        }
    
    features = {
        'timestamp': dt.timestamp(),
        'day_of_week': float(dt.weekday()),  # Monday=0, Sunday=6
        'month': float(dt.month),
        'quarter': float((dt.month - 1) // 3 + 1),
        'year': float(dt.year)
    }
    
    if reference_date:
        delta = dt - reference_date
        features['days_from_reference'] = delta.total_seconds() / (24 * 3600)
    else:
        features['days_from_reference'] = 0.0
    
    return features


def preprocess_tender(tender: Dict[str, Any],
                     text_fields: Optional[List[str]] = None,
                     numeric_fields: Optional[List[str]] = None,
                     date_fields: Optional[List[str]] = None,
                     categorical_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Preprocess a single tender object.
    
    Args:
        tender: Raw tender dictionary
        text_fields: List of field names containing text to clean
        numeric_fields: List of field names containing numeric values
        date_fields: List of field names containing dates
        categorical_fields: List of field names containing categorical values
        
    Returns:
        Preprocessed tender dictionary with additional processed fields
    """
    # Use provided fields or get from configuration
    if text_fields is None:
        text_fields = _get_field_names('text')
    
    if numeric_fields is None:
        numeric_fields = _get_field_names('numeric')
    
    if date_fields is None:
        date_fields = _get_field_names('date')
    
    if categorical_fields is None:
        categorical_fields = _get_field_names('categorical')
    
    processed = tender.copy()
    
    # Process text fields
    for field in text_fields:
        if field in tender and tender[field]:
            processed[f'{field}_cleaned'] = clean_russian_text(tender[field])
        else:
            processed[f'{field}_cleaned'] = ''
    
    # Process numeric fields
    for field in numeric_fields:
        if field in tender and tender[field]:
            processed[f'{field}_parsed'] = parse_russian_number(tender[field])
        else:
            processed[f'{field}_parsed'] = 0.0
    
    # Process date fields
    for field in date_fields:
        if field in tender and tender[field]:
            dt = parse_russian_date(tender[field])
            processed[f'{field}_parsed'] = dt
            if dt:
                processed[f'{field}_features'] = extract_date_features(tender[field])
            else:
                processed[f'{field}_features'] = {}
        else:
            processed[f'{field}_parsed'] = None
            processed[f'{field}_features'] = {}
    
    # Process categorical fields (just ensure they exist)
    for field in categorical_fields:
        if field not in processed:
            processed[field] = ''
    
    return processed


def compute_field_statistics(tenders: List[Dict[str, Any]], 
                            numeric_fields: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for normalization of numeric fields.
    
    Args:
        tenders: List of preprocessed tender dictionaries
        numeric_fields: List of numeric field names to compute stats for
        
    Returns:
        Dictionary mapping field names to statistics (min, max, mean, std)
    """
    if numeric_fields is None:
        # Get numeric fields from configuration and add '_parsed' suffix
        base_fields = _get_field_names('numeric')
        numeric_fields = [f'{field}_parsed' for field in base_fields]
    
    stats = {}
    
    for field in numeric_fields:
        values = []
        for tender in tenders:
            if field in tender:
                value = tender[field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if values:
            stats[field] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) if len(values) > 1 else 1.0,
                'count': len(values)
            }
        else:
            stats[field] = {
                'min': 0.0,
                'max': 1.0,
                'mean': 0.0,
                'std': 1.0,
                'count': 0
            }
    
    return stats


def load_tenders_from_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load tenders from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of tender dictionaries
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Some JSON files might have a root object with tenders inside
            for key in ['tenders', 'data', 'results']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        else:
            return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return []


# ============================================================================
# Configuration-aware helper functions
# ============================================================================

def get_text_field_weights() -> Dict[str, float]:
    """
    Get text field weights for embedding generation from configuration.
    
    Returns:
        Dictionary mapping cleaned field names to weights
    """
    try:
        if _config:
            return _config.get('embedding.text_field_weights', {})
    except (AttributeError, KeyError):
        pass
    
    # Fallback to old constant
    try:
        from config import TEXT_FIELD_WEIGHTS
        return TEXT_FIELD_WEIGHTS
    except ImportError:
        return {
            'Наименование объекта закупки_cleaned': 0.4,
            'Наименование закупки_cleaned': 0.3,
            'Заказчик_cleaned': 0.2,
            'Требования к участникам_cleaned': 0.05,
            'Критерии оценки заявок_cleaned': 0.05
        }


def get_region_hierarchy() -> Dict[str, List[str]]:
    """
    Get region hierarchy for categorical similarity from configuration.
    
    Returns:
        Dictionary mapping federal districts to regions
    """
    try:
        if _config:
            return _config.get('similarity.categorical.hierarchical_mappings.region', {})
    except (AttributeError, KeyError):
        pass
    
    # Fallback to old constant
    try:
        from config import REGION_HIERARCHY
        return REGION_HIERARCHY
    except ImportError:
        return {}


def get_procurement_method_groups() -> Dict[str, List[str]]:
    """
    Get procurement method groups for categorical similarity from configuration.
    
    Returns:
        Dictionary mapping method groups to method names
    """
    try:
        if _config:
            return _config.get('similarity.categorical.hierarchical_mappings.procurement_method', {})
    except (AttributeError, KeyError):
        pass
    
    # Fallback to old constant
    try:
        from config import PROCUREMENT_METHOD_GROUPS
        return PROCUREMENT_METHOD_GROUPS
    except ImportError:
        return {}


# ============================================================================
# PII Masking for Logging
# ============================================================================

def mask_pii(value: str, field_name: str = "") -> str:
    """
    Mask potentially sensitive information for logging.
    
    Args:
        value: The value to mask
        field_name: Name of the field (helps determine masking strategy)
        
    Returns:
        Masked string (e.g., "***" for sensitive fields, truncated for others)
    """
    if not isinstance(value, str):
        value = str(value)
    
    # List of field names that may contain PII (customize based on your data)
    pii_field_patterns = [
        "заказчик", "customer", "организатор", "организация",
        "икз", "идентификационный", "номер", "телефон", "email",
        "адрес", "контакт", "фио", "паспорт", "инн", "огрн"
    ]
    
    field_lower = field_name.lower()
    
    # Check if field contains PII patterns
    is_pii = any(pattern in field_lower for pattern in pii_field_patterns)
    
    if is_pii and value.strip():
        # For PII fields, mask entire value
        if len(value) <= 5:
            return "***"
        else:
            # Show first 2 and last 2 characters with masking in between
            return f"{value[:2]}***{value[-2:]}"
    else:
        # For non-PII fields, truncate if too long
        if len(value) > 50:
            return f"{value[:47]}..."
        return value


def mask_tender_for_logging(tender: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a masked version of a tender dictionary safe for logging.
    
    Args:
        tender: Original tender dictionary
        
    Returns:
        Masked tender dictionary with PII fields obscured
    """
    masked = {}
    for key, value in tender.items():
        if isinstance(value, str):
            masked[key] = mask_pii(value, key)
        elif isinstance(value, (int, float)):
            masked[key] = value
        elif isinstance(value, dict):
            masked[key] = mask_tender_for_logging(value)
        elif isinstance(value, list):
            masked[key] = [mask_tender_for_logging(item) if isinstance(item, dict)
                          else mask_pii(str(item), key) for item in value]
        else:
            masked[key] = str(value)
    return masked


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    example_tender = {
        "Наименование объекта закупки": "Поставка компьютерной техники",
        "Начальная (максимальная) цена контракта": "1 250 000,00",
        "Дата публикации": "15.03.2024 10:30",
        "Способ определения поставщика": "Электронный аукцион",
        "Регион": "Москва"
    }
    
    processed = preprocess_tender(example_tender)
    print("Example preprocessing:")
    print(f"Original price: {example_tender['Начальная (максимальная) цена контракта']}")
    print(f"Parsed price: {processed['Начальная (максимальная) цена контракта_parsed']}")
    print(f"Cleaned text: {processed['Наименование объекта закупки_cleaned']}")
    print(f"Date features: {processed.get('Дата публикации_features', {})}")
    
    # Demonstrate configuration access
    print(f"\nConfiguration check:")
    print(f"Text fields: {_get_field_names('text')[:3]}...")
    print(f"Text field weights: {list(get_text_field_weights().keys())}")