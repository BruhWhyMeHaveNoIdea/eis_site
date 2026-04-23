"""
Algorithm 1: Simple select_k_best for retrospective analysis of government procurement tenders.

This module implements the main algorithm for finding the k most similar tenders
to a target entity based on composite similarity scores using embeddings.

REFACTORED VERSION: Uses configurable field names, weights, and parameters
from config.settings instead of hardcoded values.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

from core.preprocessing import preprocess_tender, load_tenders_from_json, compute_field_statistics
from core.embeddings import EmbeddingGenerator, get_default_generator
from core.similarity import composite_similarity, batch_similarity

# Import configuration system
try:
    from config.settings import get_config, get_field_mapping
    _config = get_config()
except ImportError:
    # Fallback to old config module for backward compatibility
    import warnings
    warnings.warn("config.settings not found, using fallback configuration")
    from config import (
        DEFAULT_WEIGHTS, DEFAULT_K, MIN_SIMILARITY_THRESHOLD, EXCLUDE_SELF,
        BATCH_SIZE, N_JOBS, OUTPUT_FORMAT, ESSENTIAL_FIELDS
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


class SelectKBest:
    """
    Implementation of Algorithm 1: Simple select_k_best.
    
    Finds the k most similar tenders to a target entity based on
    composite similarity scores using embeddings and specialized
    similarity metrics for different data types.
    """
    
    def __init__(self,
                 embedding_generator: Optional[EmbeddingGenerator] = None,
                 weights: Optional[Dict[str, float]] = None,
                 preprocess_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the select_k_best algorithm.
        
        Args:
            embedding_generator: Embedding generator for text similarity
            weights: Weights for similarity components
            preprocess_config: Configuration for preprocessing
        """
        self.embedding_generator = embedding_generator or get_default_generator()
        
        # Get default weights from configuration
        if weights is None:
            weights = _get_config_value('weights', {
                'text': 0.3,           # Procurement name
                'description': 0.2,    # Procurement description
                'customer': 0.15,      # Customer name
                'region': 0.1,         # Region
                'method': 0.05,        # Procurement method
                'price': 0.1,          # Price
                'date': 0.05,          # Date
                'other': 0.05          # Other categorical fields
            })
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.weights = weights
        
        self.preprocess_config = preprocess_config or {}
        self.tenders: List[Dict[str, Any]] = []
        self.processed_tenders: List[Dict[str, Any]] = []
        self.field_stats: Dict[str, Dict[str, float]] = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_tenders(self, 
                    tenders: Union[str, List[Dict[str, Any]]],
                    preprocess: bool = True) -> None:
        """
        Load tenders from file or list.
        
        Args:
            tenders: Path to JSON file or list of tender dictionaries
            preprocess: Whether to preprocess tenders after loading
        """
        if isinstance(tenders, str):
            self.logger.info(f"Loading tenders from file: {tenders}")
            self.tenders = load_tenders_from_json(tenders)
        else:
            self.logger.info(f"Loading {len(tenders)} tenders from list")
            self.tenders = tenders
        
        if preprocess:
            self.preprocess_tenders()
    
    def preprocess_tenders(self) -> None:
        """Preprocess all loaded tenders."""
        self.logger.info(f"Preprocessing {len(self.tenders)} tenders...")
        
        self.processed_tenders = []
        for i, tender in enumerate(self.tenders):
            processed = preprocess_tender(tender, **self.preprocess_config)
            self.processed_tenders.append(processed)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"  Processed {i + 1} tenders")
        
        # Compute field statistics for normalization
        self.field_stats = compute_field_statistics(self.processed_tenders)
        
        self.logger.info(f"Preprocessing complete. Processed {len(self.processed_tenders)} tenders.")
    
    def find_similar(self,
                    target_tender: Dict[str, Any],
                    k: int = 10,
                    exclude_self: bool = True,
                    min_similarity: float = 0.0,
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find k most similar tenders to target tender.
        
        Args:
            target_tender: Target tender dictionary
            k: Number of similar tenders to return
            exclude_self: Whether to exclude the target tender itself (if in dataset)
            min_similarity: Minimum similarity score threshold
            filters: Optional filters to apply before similarity computation
            
        Returns:
            List of similar tenders with metadata
        """
        start_time = time.time()
        
        # Preprocess target tender
        target_processed = preprocess_tender(target_tender, **self.preprocess_config)
        
        # Apply filters if specified
        candidates = self._filter_tenders(filters) if filters else self.processed_tenders
        
        # Exclude target tender if it's in the dataset
        if exclude_self:
            # Get ID field from configuration
            id_field = _get_config_value('field_mappings.id_field', 'Идентификационный код закупки (ИКЗ)')
            target_id = target_tender.get(id_field, '')
            if target_id:
                # Exclude by ID match
                candidates = [
                    t for t in candidates
                    if t.get(id_field, '') != target_id
                ]
            else:
                # No ID field, exclude by exact dictionary equality (deep match)
                # This is less efficient but handles edge cases
                target_processed_json = json.dumps(target_processed, sort_keys=True)
                # Exclude only the first matching tender to mimic "self" exclusion
                excluded = False
                filtered = []
                for t in candidates:
                    if not excluded and json.dumps(t, sort_keys=True) == target_processed_json:
                        excluded = True  # skip this one
                        continue
                    filtered.append(t)
                candidates = filtered
        
        self.logger.info(f"Finding {k} most similar tenders from {len(candidates)} candidates...")
        
        # Compute similarities
        similarity_results = batch_similarity(
            target_processed,
            candidates,
            weights=self.weights,
            embedding_generator=self.embedding_generator
        )
        
        # Apply minimum similarity threshold
        if min_similarity > 0:
            similarity_results = [
                r for r in similarity_results 
                if r[0] >= min_similarity
            ]
        
        # Take top k
        top_k = similarity_results[:k]
        
        # Prepare results
        results = []
        for rank, (score, tender, breakdown) in enumerate(top_k, 1):
            result = {
                'rank': rank,
                'similarity_score': float(score),
                'tender': tender,
                'similarity_breakdown': breakdown,
                'explanation': self._generate_explanation(score, breakdown)
            }
            results.append(result)
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        self.logger.info(f"Found {len(results)} similar tenders in {elapsed_time:.2f} ms")
        
        return results
    
    def _filter_tenders(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter tenders based on criteria.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered list of tenders
        """
        filtered = self.processed_tenders
        
        for field, condition in filters.items():
            if isinstance(condition, dict):
                # Complex condition (range, list, etc.)
                if 'min' in condition and 'max' in condition:
                    # Numeric range
                    min_val = condition['min']
                    max_val = condition['max']
                    filtered = [
                        t for t in filtered
                        if field in t and min_val <= t[field] <= max_val
                    ]
                elif 'in' in condition:
                    # Value in list
                    values = condition['in']
                    filtered = [
                        t for t in filtered
                        if field in t and t[field] in values
                    ]
            else:
                # Simple equality
                filtered = [
                    t for t in filtered
                    if field in t and t[field] == condition
                ]
        
        return filtered
    
    def _generate_explanation(self, 
                            score: float, 
                            breakdown: Dict[str, float]) -> str:
        """
        Generate human-readable explanation for similarity score.
        
        Args:
            score: Overall similarity score
            breakdown: Similarity breakdown by component
            
        Returns:
            Explanation string
        """
        # Find top contributing factors
        factors = []
        for component, comp_score in breakdown.items():
            component_name = component.replace('_similarity', '')
            weight = self.weights.get(component_name, 0.0)
            if weight > 0 and comp_score > 0.7:
                factors.append(f"high {component_name} similarity ({comp_score:.2f})")
            elif weight > 0 and comp_score < 0.3:
                factors.append(f"low {component_name} similarity ({comp_score:.2f})")
        
        if score >= 0.8:
            strength = "very strong"
        elif score >= 0.6:
            strength = "strong"
        elif score >= 0.4:
            strength = "moderate"
        elif score >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        if factors:
            factors_str = ", ".join(factors[:3])  # Limit to top 3 factors
            explanation = f"{strength} similarity (score: {score:.3f}) due to {factors_str}"
        else:
            explanation = f"{strength} similarity (score: {score:.3f})"
        
        return explanation
    
    def find_similar_to_query(self,
                             query: Dict[str, Any],
                             k: int = 10,
                             **kwargs) -> Dict[str, Any]:
        """
        Find similar tenders to a query and return formatted results.
        
        Args:
            query: Query dictionary with target tender and optional parameters
            k: Number of similar tenders to return
            **kwargs: Additional arguments passed to find_similar
            
        Returns:
            Formatted results in the specified output format
        """
        target_tender = query.get('target_tender', {})
        parameters = query.get('parameters', {})
        
        # Override defaults with query parameters
        k = parameters.get('k', k)
        weights = parameters.get('weights', self.weights)
        filters = parameters.get('filters', None)
        min_similarity = parameters.get('min_similarity', 0.0)
        
        # Find similar tenders
        similar_tenders = self.find_similar(
            target_tender=target_tender,
            k=k,
            min_similarity=min_similarity,
            filters=filters,
            **kwargs
        )
        
        # Prepare output
        output = {
            'query_tender': target_tender,
            'similar_tenders': similar_tenders,
            'metadata': {
                'total_tenders_searched': len(self.processed_tenders),
                'k_requested': k,
                'k_returned': len(similar_tenders),
                'algorithm': 'select_k_best_v1',
                'weights_used': weights,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return output
    
    def save_results(self, 
                    results: Dict[str, Any], 
                    filepath: str,
                    include_full_tenders: bool = True) -> None:
        """
        Save similarity results to JSON file.
        
        Args:
            results: Results dictionary from find_similar_to_query
            filepath: Path to save JSON file
            include_full_tenders: Whether to include full tender objects in output
        """
        # Create a copy to avoid modifying original
        output = results.copy()
        
        if not include_full_tenders:
            # Get essential fields from configuration
            essential_fields = _get_config_value('output.essential_fields', [
                'Идентификационный код закупки (ИКЗ)',
                'Наименование объекта закупки',
                'Начальная (максимальная) цена контракта',
                'Регион',
                'Заказчик',
                'Дата публикации'
            ])
            
            # Remove full tender objects to reduce file size
            for item in output.get('similar_tenders', []):
                if 'tender' in item:
                    # Keep only essential fields
                    item['tender'] = {
                        k: v for k, v in item['tender'].items()
                        if k in essential_fields
                    }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def evaluate_ranking(self,
                        target_tender: Dict[str, Any],
                        relevant_tender_ids: List[str],
                        k: int = 10) -> Dict[str, float]:
        """
        Evaluate ranking quality using standard metrics.
        
        Args:
            target_tender: Target tender
            relevant_tender_ids: List of IDs of truly relevant tenders
            k: Number of results to consider
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Find similar tenders
        results = self.find_similar(target_tender, k=len(self.processed_tenders))
        
        # Get ID field from configuration
        id_field = _get_config_value('field_mappings.id_field', 'Идентификационный код закупки (ИКЗ)')
        
        # Extract IDs of returned tenders
        returned_ids = []
        for result in results:
            tender = result['tender']
            tender_id = tender.get(id_field, '')
            if tender_id:
                returned_ids.append(tender_id)
        
        # Calculate metrics
        metrics = {}
        
        # Precision@k
        top_k_ids = returned_ids[:k]
        relevant_in_top_k = [tid for tid in top_k_ids if tid in relevant_tender_ids]
        metrics[f'precision@{k}'] = len(relevant_in_top_k) / k if k > 0 else 0.0
        
        # Recall@k
        metrics[f'recall@{k}'] = len(relevant_in_top_k) / len(relevant_tender_ids) if relevant_tender_ids else 0.0
        
        # Mean Reciprocal Rank (MRR)
        for rank, tid in enumerate(returned_ids, 1):
            if tid in relevant_tender_ids:
                metrics['mrr'] = 1.0 / rank
                break
        else:
            metrics['mrr'] = 0.0
        
        # Average Precision
        precision_values = []
        num_relevant_found = 0
        for rank, tid in enumerate(returned_ids, 1):
            if tid in relevant_tender_ids:
                num_relevant_found += 1
                precision_at_rank = num_relevant_found / rank
                precision_values.append(precision_at_rank)
        
        if precision_values:
            metrics['average_precision'] = sum(precision_values) / len(relevant_tender_ids)
        else:
            metrics['average_precision'] = 0.0
        
        return metrics


def create_target_entity_from_criteria(criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a target entity from search criteria.
    
    Args:
        criteria: Dictionary with search criteria
        
    Returns:
        Target tender dictionary
    """
    target = {}
    
    # Get field mappings from configuration
    field_mappings = _get_config_value('field_mappings', {})
    
    # Default field mapping (fallback)
    default_field_mapping = {
        'region': 'Регион',
        'method': 'Способ определения поставщика',
        'customer': 'Заказчик',
        'min_price': 'Начальная (максимальная) цена контракта',
        'max_price': 'Начальная (максимальная) цена контракта',
        'date_from': 'Дата публикации',
        'date_to': 'Дата публикации',
        'keywords': 'Наименование объекта закупки'
    }
    
    # Use configured field names if available
    field_mapping = {
        'region': field_mappings.get('region_field', default_field_mapping['region']),
        'method': field_mappings.get('method_field', default_field_mapping['method']),
        'customer': field_mappings.get('customer_field', default_field_mapping['customer']),
        'min_price': field_mappings.get('price_field', default_field_mapping['min_price']),
        'max_price': field_mappings.get('price_field', default_field_mapping['max_price']),
        'date_from': field_mappings.get('date_fields', [default_field_mapping['date_from']])[0],
        'date_to': field_mappings.get('date_fields', [default_field_mapping['date_to']])[0],
        'keywords': field_mappings.get('title_field', default_field_mapping['keywords'])
    }
    
    for crit_key, crit_value in criteria.items():
        if crit_key in field_mapping:
            tender_field = field_mapping[crit_key]
            target[tender_field] = crit_value
    
    return target


# ============================================================================
# Configuration-aware helper functions
# ============================================================================

def get_default_algorithm_params() -> Dict[str, Any]:
    """
    Get default algorithm parameters from configuration.
    
    Returns:
        Dictionary with default_k, min_similarity_threshold, exclude_self, batch_size, n_jobs
    """
    return {
        'default_k': _get_config_value('algorithm.default_k', 10),
        'min_similarity_threshold': _get_config_value('algorithm.min_similarity_threshold', 0.0),
        'exclude_self': _get_config_value('algorithm.exclude_self', True),
        'batch_size': _get_config_value('algorithm.batch_size', 32),
        'n_jobs': _get_config_value('algorithm.n_jobs', 1)
    }


def get_output_config() -> Dict[str, Any]:
    """
    Get output configuration from settings.
    
    Returns:
        Dictionary with output format settings
    """
    return _get_config_value('output', {
        'include_full_tenders': True,
        'include_breakdown': True,
        'include_explanation': True,
        'include_metadata': True,
        'pretty_print': True,
        'indent': 2
    })


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=== Select K Best Algorithm Example ===")
    
    # Create algorithm instance
    algorithm = SelectKBest()
    
    # Example 1: Create synthetic tenders for demonstration
    print("\n1. Creating synthetic tenders for demonstration...")
    synthetic_tenders = []
    for i in range(5):
        tender = {
            "Идентификационный код закупки (ИКЗ)": f"0123456789012345678{i}",
            "Наименование объекта закупки": f"Поставка оборудования {i}",
            "Наименование закупки": f"Закупка оборудования для проекта {i}",
            "Начальная (максимальная) цена контракта": f"{100000 + i * 50000:,}".replace(',', ' ').replace('.', ',') + ",00",
            "Дата публикации": f"15.0{i+1}.2024 10:30",
            "Способ определения поставщика": "Электронный аукцион",
            "Регион": "Москва" if i < 3 else "Санкт-Петербург",
            "Заказчик": f"Министерство {['образования', 'здравоохранения', 'транспорта'][i % 3]}",
            "Валюта": "RUB",
            "Источник финансирования": "Федеральный бюджет"
        }
        synthetic_tenders.append(tender)
    
    # Load synthetic tenders
    algorithm.load_tenders(synthetic_tenders)
    
    # Example 2: Find similar tenders
    print("\n2. Finding similar tenders...")
    target_tender = synthetic_tenders[0]
    
    similar = algorithm.find_similar(target_tender, k=3)
    
    print(f"\nTop {len(similar)} similar tenders to target:")
    for result in similar:
        print(f"  Rank {result['rank']}: Score {result['similarity_score']:.4f}")
        print(f"    Explanation: {result['explanation']}")
        tender = result['tender']
        print(f"    Tender: {tender.get('Наименование объекта закупки', 'N/A')}")
        print(f"    Region: {tender.get('Регион', 'N/A')}")
        print()
    
    # Example 3: Query-based search
    print("\n3. Query-based search...")
    query = {
        "target_tender": target_tender,
        "parameters": {
            "k": 2,
            "weights": {
                "text": 0.4,
                "region": 0.2,
                "price": 0.2,
                "date": 0.1,
                "other": 0.1
            }
        }
    }
    
    results = algorithm.find_similar_to_query(query)
    print(f"Found {len(results['similar_tenders'])} similar tenders")
    print(f"Metadata: {results['metadata']}")
    
    # Example 4: Create target from criteria
    print("\n4. Creating target from criteria...")
    criteria = {
        "region": "Москва",
        "method": "Электронный аукцион",
        "keywords": "компьютерная техника"
    }
    
    target_from_criteria = create_target_entity_from_criteria(criteria)
    print(f"Target from criteria: {target_from_criteria}")
    
    # Show configuration values
    print("\n5. Configuration check:")
    print(f"Default algorithm params: {get_default_algorithm_params()}")
    print(f"Output config: {get_output_config()}")
    
    print("\n=== Example complete ===")