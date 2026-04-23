#!/usr/bin/env python3
"""
Simple price predictor for government procurement tenders.

This module implements a simple price prediction algorithm based on
similar tender prices using various aggregation methods (mean, median,
weighted_median, etc.).

It works in conjunction with MLRetrospectiveAnalyzer to find similar
tenders and predict prices for new tenders.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SimplePricePredictor:
    """
    Simple price predictor that uses similar tenders to predict prices.
    
    Attributes:
        analyzer: MLRetrospectiveAnalyzer instance for finding similar tenders
        method: Prediction method ('mean', 'median', 'weighted_median', 'min', 'max')
    """
    
    def __init__(self, analyzer, method: str = 'weighted_median'):
        """
        Initialize the predictor.
        
        Args:
            analyzer: MLRetrospectiveAnalyzer instance
            method: Prediction method ('mean', 'median', 'weighted_median', 'min', 'max')
        """
        self.analyzer = analyzer
        self.method = method
        
        # Validate method
        valid_methods = ['mean', 'median', 'weighted_median', 'min', 'max', 'mode']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    def predict(self, tender: Dict[str, Any], k: int = 10) -> Dict[str, Any]:
        """
        Predict price for a tender based on similar tenders.
        
        Args:
            tender: New tender data
            k: Number of similar tenders to consider
            
        Returns:
            Dictionary with prediction results:
            - predicted_price: Predicted price (float or None)
            - method: Method used
            - confidence: Confidence score (0-1)
            - price_range: (min_price, max_price) of similar tenders
            - similar_tenders_used: Number of similar tenders used
            - similar_tenders_total: Total similar tenders found
            - similar_tenders: List of similar tenders with details
        """
        if not self.analyzer:
            raise ValueError("Analyzer not initialized")
        
        # Find similar tenders
        try:
            results = self.analyzer.find_similar_tenders(tender, k=k)
        except Exception as e:
            logger.error(f"Error finding similar tenders: {e}")
            # Fallback to empty results
            results = {'similar_tenders': []}
        
        similar_tenders = results.get('similar_tenders', [])
        
        # Extract prices from similar tenders
        prices = []
        valid_similar_tenders = []
        
        for tender_info in similar_tenders:
            tender_data = tender_info.get('tender', {})
            price = self._extract_price(tender_data)
            if price is not None:
                prices.append(price)
                valid_similar_tenders.append(tender_info)
        
        # Prepare result structure
        result = {
            'predicted_price': None,
            'method': self.method,
            'confidence': 0.0,
            'price_range': (None, None),
            'similar_tenders_used': len(prices),
            'similar_tenders_total': len(similar_tenders),
            'similar_tenders': valid_similar_tenders,
            'metadata': {
                'prices_extracted': len(prices),
                'prediction_method': self.method
            }
        }
        
        if not prices:
            logger.warning("No valid prices found in similar tenders")
            return result
        
        # Calculate price range
        min_price = min(prices)
        max_price = max(prices)
        result['price_range'] = (min_price, max_price)
        
        # Calculate predicted price based on method
        predicted_price = self._calculate_price(prices, valid_similar_tenders)
        result['predicted_price'] = predicted_price
        
        # Calculate confidence (simple heuristic based on number of similar tenders and price spread)
        if len(prices) >= 3:
            # More tenders = higher confidence
            count_confidence = min(len(prices) / 10, 1.0)
            
            # Lower spread = higher confidence
            if max_price > 0:
                spread_ratio = (max_price - min_price) / max_price
                spread_confidence = max(0, 1.0 - spread_ratio)
            else:
                spread_confidence = 0.5
                
            # Combined confidence
            confidence = 0.7 * count_confidence + 0.3 * spread_confidence
        else:
            confidence = 0.3 * (len(prices) / 3)
        
        result['confidence'] = min(max(confidence, 0.0), 1.0)
        
        return result
    
    def _extract_price(self, tender: Dict[str, Any]) -> Optional[float]:
        """
        Extract price from tender data.
        
        Args:
            tender: Tender dictionary
            
        Returns:
            Price as float or None if not found/invalid
        """
        # Try different possible price field names
        price_fields = [
            'Начальная (максимальная) цена контракта',
            'Начальная цена контракта',
            'Максимальная цена контракта',
            'Цена контракта',
            'price',
            'contract_price',
            'initial_price'
        ]
        
        for field in price_fields:
            if field in tender:
                try:
                    price = tender[field]
                    # Handle string with commas, spaces, etc.
                    if isinstance(price, str):
                        # Remove non-numeric characters except dots and commas
                        import re
                        price = re.sub(r'[^\d.,]', '', price)
                        # Replace comma with dot if needed
                        if ',' in price and '.' not in price:
                            price = price.replace(',', '.')
                        elif ',' in price and '.' in price:
                            # Handle European format: 1.234,56 -> 1234.56
                            if price.rfind(',') > price.rfind('.'):
                                price = price.replace('.', '').replace(',', '.')
                            else:
                                price = price.replace(',', '')
                    
                    price_float = float(price)
                    if price_float > 0:
                        return price_float
                except (ValueError, TypeError, AttributeError):
                    continue
        
        return None
    
    def _calculate_price(self, prices: List[float], similar_tenders: List[Dict]) -> float:
        """
        Calculate predicted price using the selected method.
        
        Args:
            prices: List of prices from similar tenders
            similar_tenders: List of similar tender info (for weights)
            
        Returns:
            Predicted price
        """
        if not prices:
            return 0.0
        
        if self.method == 'mean':
            return np.mean(prices)
        
        elif self.method == 'median':
            return np.median(prices)
        
        elif self.method == 'weighted_median':
            # Weight by similarity score
            similarities = []
            for tender_info in similar_tenders:
                similarity = tender_info.get('similarity_score', 0.5)
                similarities.append(similarity)
            
            # Normalize similarities to sum to 1
            if sum(similarities) > 0:
                weights = [s / sum(similarities) for s in similarities]
            else:
                weights = [1.0 / len(similarities)] * len(similarities)
            
            # Calculate weighted median
            sorted_indices = np.argsort(prices)
            sorted_prices = [prices[i] for i in sorted_indices]
            sorted_weights = [weights[i] for i in sorted_indices]
            
            cumsum = 0
            for i, w in enumerate(sorted_weights):
                cumsum += w
                if cumsum >= 0.5:
                    return sorted_prices[i]
            
            # Fallback to regular median
            return np.median(prices)
        
        elif self.method == 'min':
            return min(prices)
        
        elif self.method == 'max':
            return max(prices)
        
        elif self.method == 'mode':
            # Simple mode approximation (binning)
            if len(prices) == 1:
                return prices[0]
            
            # Use histogram to find bin with most values
            hist, bin_edges = np.histogram(prices, bins='auto')
            max_bin_idx = np.argmax(hist)
            mode_price = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
            return mode_price
        
        else:
            # Default to median
            return np.median(prices)


def test_predictor():
    """Simple test function for the predictor."""
    # Mock analyzer for testing
    class MockAnalyzer:
        def find_similar_tenders(self, tender, k=10):
            return {
                'similar_tenders': [
                    {
                        'tender': {'Начальная (максимальная) цена контракта': '1000000'},
                        'similarity_score': 0.9
                    },
                    {
                        'tender': {'Начальная (максимальная) цена контракта': '1500000'},
                        'similarity_score': 0.8
                    },
                    {
                        'tender': {'Начальная (максимальная) цена контракта': '800000'},
                        'similarity_score': 0.7
                    }
                ]
            }
    
    analyzer = MockAnalyzer()
    predictor = SimplePricePredictor(analyzer, method='weighted_median')
    
    test_tender = {'Наименование объекта закупки': 'Test tender'}
    result = predictor.predict(test_tender, k=5)
    
    print("Test prediction result:")
    for key, value in result.items():
        if key != 'similar_tenders':
            print(f"  {key}: {value}")
    
    return result


if __name__ == '__main__':
    test_predictor()