"""
Training script for ML-based retrospective analysis model.

This script provides a command-line interface and programmatic API for
training the ML retrospective analysis model on tender data.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from ml_retrospective import MLRetrospectiveAnalyzer, MLRetrospectiveConfig
from core.preprocessing import load_tenders_from_json

def get_ml_config() -> MLRetrospectiveConfig:
    """Get default ML configuration (compatibility wrapper)."""
    return MLRetrospectiveConfig()


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_training_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load training data from JSON file.
    
    Args:
        data_path: Path to JSON file with tender data
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        List of tender dictionaries
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading training data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load tenders
    tenders = load_tenders_from_json(data_path)
    
    if not tenders:
        raise ValueError(f"No tender data found in {data_path}")
    
    logger.info(f"Loaded {len(tenders)} tenders")
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(tenders):
        logger.info(f"Limiting to {max_samples} samples for testing")
        tenders = tenders[:max_samples]
    
    return tenders


def create_analyzer_from_config(config: MLRetrospectiveConfig) -> MLRetrospectiveAnalyzer:
    """
    Create ML retrospective analyzer from configuration.
    
    Args:
        config: ML configuration (MLRetrospectiveConfig)
        
    Returns:
        Configured MLRetrospectiveAnalyzer
    """
    return MLRetrospectiveAnalyzer(config)


def train_model(tenders: List[Dict[str, Any]],
                config: Optional[MLRetrospectiveConfig] = None,
                output_dir: str = "./models") -> MLRetrospectiveAnalyzer:
    """
    Train ML retrospective analysis model.
    
    Args:
        tenders: Training tender data
        config: ML configuration (uses default if None)
        output_dir: Directory to save trained model
        
    Returns:
        Trained MLRetrospectiveAnalyzer
    """
    logger = logging.getLogger(__name__)
    
    # Use default config if not provided
    if config is None:
        config = get_ml_config()
    
    logger.info("Creating ML retrospective analyzer")
    analyzer = create_analyzer_from_config(config)
    
    # Train the model
    logger.info("Starting model training")
    analyzer.train(tenders)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(output_dir, "ml_retrospective_model.pkl")
    analyzer.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save configuration
    config_path = os.path.join(output_dir, "ml_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    logger.info(f"Configuration saved to {config_path}")
    
    # Generate and save cluster analysis
    if analyzer.cluster_stats is not None:
        cluster_analysis = analyzer.analyze_clusters()
        # Convert int64 keys to int for JSON serialization
        cluster_analysis = {int(k): v for k, v in cluster_analysis.items()}
        cluster_path = os.path.join(output_dir, "cluster_analysis.json")
        with open(cluster_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Cluster analysis saved to {cluster_path}")
    
    # Generate training report
    report = generate_training_report(analyzer, len(tenders))
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Training report saved to {report_path}")
    
    return analyzer


def generate_training_report(analyzer: MLRetrospectiveAnalyzer, 
                            n_tenders: int) -> Dict[str, Any]:
    """
    Generate training report with model statistics.
    
    Args:
        analyzer: Trained analyzer
        n_tenders: Number of tenders used for training
        
    Returns:
        Training report dictionary
    """
    report = {
        'training_date': str(np.datetime64('now')),
        'n_tenders': n_tenders,
        'model_type': 'ML Retrospective Analyzer',
        'components_enabled': {
            'clustering': analyzer.config.enable_clustering,
            'similarity_learning': analyzer.config.enable_similarity_learning,
            'faiss_index': analyzer.config.enable_faiss_index
        }
    }
    
    # Add feature information
    if analyzer.features is not None:
        report['feature_dimensions'] = analyzer.feature_pipeline.get_feature_dimensions()
        report['feature_matrix_shape'] = list(analyzer.features.shape)
    
    # Add clustering information
    if analyzer.cluster_labels is not None:
        unique_clusters = np.unique(analyzer.cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_outliers = np.sum(analyzer.cluster_labels == -1)
        
        report['clustering'] = {
            'n_clusters': int(n_clusters),
            'n_outliers': int(n_outliers),
            'cluster_distribution': {int(k): int(v) for k, v in analyzer._get_cluster_distribution().items()}
        }
    
    # Add embedding information
    if analyzer.embeddings is not None:
        report['embeddings'] = {
            'shape': list(analyzer.embeddings.shape),
            'embedding_dim': analyzer.config.similarity_config.embedding_dim
        }
    
    # Add index information
    if analyzer.is_index_built and analyzer.search_engine is not None:
        report['faiss_index'] = {
            'size': analyzer.search_engine.index.size(),
            'index_type': analyzer.config.faiss_config.index_type,
            'metric': analyzer.config.faiss_config.metric_type
        }
    
    return report


def evaluate_model(analyzer: MLRetrospectiveAnalyzer,
                  test_tenders: List[Dict[str, Any]],
                  n_queries: int = 10) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.
    
    Args:
        analyzer: Trained analyzer
        test_tenders: Test tender data
        n_queries: Number of test queries to run
        
    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model on {len(test_tenders)} test tenders")
    
    # Select random queries
    if n_queries > len(test_tenders):
        n_queries = len(test_tenders)
    
    query_indices = np.random.choice(len(test_tenders), n_queries, replace=False)
    
    results = []
    for i, idx in enumerate(query_indices):
        query_tender = test_tenders[idx]
        
        try:
            # Find similar tenders
            search_results = analyzer.find_similar_tenders(query_tender, k=5)
            
            # Check if query tender itself is in results (should be excluded or ranked high)
            result_indices = [r['index'] for r in search_results['similar_tenders']]
            
            results.append({
                'query_index': int(idx),
                'n_results': len(search_results['similar_tenders']),
                'avg_similarity': np.mean([r['similarity_score'] for r in search_results['similar_tenders']]),
                'top_similarity': search_results['similar_tenders'][0]['similarity_score'] if search_results['similar_tenders'] else 0
            })
        except Exception as e:
            logger.warning(f"Error processing query {i}: {e}")
            results.append({
                'query_index': int(idx),
                'error': str(e)
            })
    
    # Compute evaluation metrics
    if results and all('avg_similarity' in r for r in results):
        avg_similarities = [r['avg_similarity'] for r in results if 'avg_similarity' in r]
        top_similarities = [r['top_similarity'] for r in results if 'top_similarity' in r]
        
        evaluation = {
            'n_queries': len(results),
            'avg_similarity_mean': float(np.mean(avg_similarities)),
            'avg_similarity_std': float(np.std(avg_similarities)),
            'top_similarity_mean': float(np.mean(top_similarities)),
            'top_similarity_std': float(np.std(top_similarities)),
            'query_results': results
        }
    else:
        evaluation = {
            'n_queries': len(results),
            'query_results': results
        }
    
    return evaluation


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML-based retrospective analysis model for government procurement tenders"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to JSON file with tender data"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./models",
        help="Directory to save trained model (default: ./models)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to ML configuration JSON file (optional)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use for training (for testing)"
    )
    
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on test split after training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = MLRetrospectiveConfig(**config_dict)
        else:
            logger.info("Using default configuration")
            config = get_ml_config()
        
        # Load training data
        tenders = load_training_data(args.data, args.max_samples)
        
        # Split data if evaluation is requested
        if args.evaluate and args.test_split > 0:
            from sklearn.model_selection import train_test_split
            train_tenders, test_tenders = train_test_split(
                tenders, test_size=args.test_split, random_state=42
            )
            logger.info(f"Split data: {len(train_tenders)} training, {len(test_tenders)} test")
        else:
            train_tenders = tenders
            test_tenders = []
        
        # Train model
        analyzer = train_model(train_tenders, config, args.output_dir)
        
        # Evaluate if requested
        if args.evaluate and test_tenders:
            logger.info("Evaluating model on test data")
            evaluation = evaluate_model(analyzer, test_tenders)
            
            eval_path = os.path.join(args.output_dir, "evaluation_results.json")
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {eval_path}")
            
            # Print summary
            if 'avg_similarity_mean' in evaluation:
                print(f"\nEvaluation Summary:")
                print(f"  Average similarity: {evaluation['avg_similarity_mean']:.3f} ± {evaluation['avg_similarity_std']:.3f}")
                print(f"  Top similarity: {evaluation['top_similarity_mean']:.3f} ± {evaluation['top_similarity_std']:.3f}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()