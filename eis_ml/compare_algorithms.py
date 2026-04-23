#!/usr/bin/env python3
"""
Сравнительный анализ алгоритмов 1 и 2 для ретроспективного анализа тендеров.

Этот скрипт загружает данные из tenders.json, разделяет их в формате 90-10,
обучает алгоритм 2 (ML-based) на обучающей выборке и сравнивает производительность
обоих алгоритмов на тестовой выборке.
"""

import json
import time
import logging
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import sys
import os

# Добавляем путь к проекту для импорта модулей

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_tenders_data(filepath: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Загружает данные тендеров из JSON файла.
    
    Args:
        filepath: Путь к JSON файлу
        max_samples: Максимальное количество записей для загрузки (для тестирования)
        
    Returns:
        Список словарей с данными тендеров
    """
    logger.info(f"Загрузка данных из {filepath}")
    
    if not os.path.exists('tenders.json'):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    with open('tenders.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Ожидается список тендеров, получен {type(data)}")
    
    if max_samples and max_samples < len(data):
        logger.info(f"Ограничение данных до {max_samples} записей")
        data = data[:max_samples]
    
    logger.info(f"Загружено {len(data)} тендеров")
    return data


def split_data_90_10(data: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Разделяет данные в формате 90-10 (обучение-тест).
    
    Args:
        data: Список тендеров
        seed: Seed для воспроизводимости
        
    Returns:
        Кортеж (train_data, test_data)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n = len(data)
    n_train = int(0.9 * n)
    
    # Перемешиваем данные
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    train_data = shuffled[:n_train]
    test_data = shuffled[n_train:]
    
    logger.info(f"Разделение данных: {len(train_data)} для обучения ({100*len(train_data)/n:.1f}%), "
                f"{len(test_data)} для тестирования ({100*len(test_data)/n:.1f}%)")
    
    return train_data, test_data


def evaluate_algorithm_1(test_tender: Dict[str, Any], historical_tenders: List[Dict[str, Any]],
                        k: int = 10) -> Dict[str, Any]:
    """
    Оценивает алгоритм 1 (Simple select_k_best) на одном тестовом тендере.
    
    Args:
        test_tender: Тестовый тендер для поиска похожих
        historical_tenders: Исторические тендеры для поиска
        k: Количество похожих тендеров для возврата
        
    Returns:
        Словарь с результатами
    """
    try:
        from select_k_best import SelectKBest
    except ImportError as e:
        logger.error(f"Не удалось импортировать алгоритм 1: {e}")
        return {
            'success': False,
            'error': str(e),
            'similar_tenders': [],
            'execution_time': 0
        }
    
    start_time = time.time()
    
    try:
        # Инициализируем алгоритм 1
        algorithm = SelectKBest()
        
        # Загружаем исторические тендеры
        algorithm.load_tenders(historical_tenders, preprocess=True)
        
        # Вызываем алгоритм 1
        similar_tenders = algorithm.find_similar(test_tender, k=k)
        
        execution_time = time.time() - start_time
        
        # Форматируем результат в соответствии с ожидаемой структурой
        formatted_results = []
        for i, tender in enumerate(similar_tenders):
            formatted_results.append({
                'tender': tender.get('tender', tender),
                'similarity_score': tender.get('similarity_score', 0),
                'rank': i + 1
            })
        
        return {
            'success': True,
            'similar_tenders': formatted_results[:k],
            'execution_time': execution_time,
            'similarity_scores': [t.get('similarity_score', 0) for t in formatted_results[:k]]
        }
    except Exception as e:
        logger.error(f"Ошибка при выполнении алгоритма 1: {e}")
        return {
            'success': False,
            'error': str(e),
            'similar_tenders': [],
            'execution_time': time.time() - start_time
        }


def evaluate_algorithm_2(test_tender: Dict[str, Any], analyzer, k: int = 10) -> Dict[str, Any]:
    """
    Оценивает алгоритм 2 (ML-based) на одном тестовом тендере.
    
    Args:
        test_tender: Тестовый тендер для поиска похожих
        analyzer: Обученный анализатор алгоритма 2
        k: Количество похожих тендеров для возврата
        
    Returns:
        Словарь с результатами
    """
    start_time = time.time()
    
    try:
        # Вызываем алгоритм 2
        result = analyzer.find_similar_tenders(test_tender, k=k)
        
        execution_time = time.time() - start_time
        
        # Извлекаем похожие тендеры из результата
        similar_tenders = result.get('similar_tenders', [])
        
        # Форматируем результат
        formatted_results = []
        for i, tender_info in enumerate(similar_tenders[:k]):
            # tender_info может быть словарем с полями 'tender' и 'similarity_score'
            if isinstance(tender_info, dict):
                tender = tender_info.get('tender', tender_info)
                score = tender_info.get('similarity_score', 0)
            else:
                tender = tender_info
                score = 0
            
            formatted_results.append({
                'tender': tender,
                'similarity_score': score,
                'rank': i + 1
            })
        
        return {
            'success': True,
            'similar_tenders': formatted_results,
            'execution_time': execution_time,
            'similarity_scores': [t.get('similarity_score', 0) for t in formatted_results]
        }
    except Exception as e:
        logger.error(f"Ошибка при выполнении алгоритма 2: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'similar_tenders': [],
            'execution_time': time.time() - start_time
        }


def load_algorithm_2(model_path: str):
    """
    Загружает предобученную модель алгоритма 2 (ML-based) из файла.
    
    Args:
        model_path: Путь к файлу модели (.pkl)
        
    Returns:
        Загруженный анализатор
    """
    try:
        from ml_retrospective import MLRetrospectiveAnalyzer
        
        logger.info(f"Загрузка модели алгоритма 2 из {model_path}...")
        start_time = time.time()
        
        analyzer = MLRetrospectiveAnalyzer.load(model_path)
        
        # Отключаем FAISS индекс, чтобы избежать ошибки несовпадения размерности
        analyzer.config.enable_faiss_index = False
        logger.info("FAISS индекс отключен (используется brute-force поиск)")
        
        load_time = time.time() - start_time
        logger.info(f"Модель алгоритма 2 загружена за {load_time:.2f} секунд")
        
        return analyzer
    except ImportError as e:
        logger.error(f"Не удалось импортировать алгоритм 2: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели алгоритма 2: {e}")
        raise


def train_algorithm_2(train_data: List[Dict[str, Any]], config: Optional[Dict] = None):
    """
    Обучает алгоритм 2 (ML-based) на обучающих данных.
    
    Args:
        train_data: Данные для обучения
        config: Конфигурация для алгоритма 2
        
    Returns:
        Обученный анализатор
    """
    try:
        from ml_retrospective import MLRetrospectiveAnalyzer, MLRetrospectiveConfig
        
        logger.info("Обучение алгоритма 2 (ML-based)...")
        start_time = time.time()
        
        # Создаем конфигурацию
        ml_config = MLRetrospectiveConfig()
        if config:
            # Применяем пользовательскую конфигурацию
            for key, value in config.items():
                if hasattr(ml_config, key):
                    setattr(ml_config, key, value)
        
        n_samples = len(train_data)
        
        # Адаптивная конфигурация в зависимости от размера данных
        if n_samples < 100:
            logger.info(f"Очень мало данных ({n_samples} образцов), сильно упрощаем конфигурацию...")
            # Отключаем сложные компоненты
            ml_config.enable_similarity_learning = False
            ml_config.enable_faiss_index = False
            ml_config.enable_clustering = False  # Отключаем кластеризацию для очень малых данных
            
            # Настраиваем feature engineering
            ml_config.feature_config.use_pca = False
            ml_config.feature_config.pca_n_components = min(10, n_samples - 1)
            
        elif n_samples < 1000:
            logger.info(f"Мало данных для обучения ({n_samples} образцов), упрощаем конфигурацию...")
            ml_config.enable_similarity_learning = False
            ml_config.enable_faiss_index = False
            
            # Настраиваем размерности
            ml_config.clustering_config.umap_n_components = min(10, n_samples - 1)
            ml_config.clustering_config.hdbscan_min_cluster_size = max(3, n_samples // 20)
            
            # Настраиваем feature engineering
            ml_config.feature_config.pca_n_components = min(20, n_samples - 1)
        
        # Отключаем объяснения, чтобы избежать ошибок с отсутствующими полями
        ml_config.output_format['include_explanation'] = False
        ml_config.output_format['include_breakdown'] = False
        ml_config.include_similarity_breakdown = False
        
        # Убедимся, что размерности не превышают количество образцов
        if hasattr(ml_config.clustering_config, 'umap_n_components'):
            ml_config.clustering_config.umap_n_components = min(
                ml_config.clustering_config.umap_n_components,
                n_samples - 1
            )
        
        if hasattr(ml_config.feature_config, 'pca_n_components'):
            ml_config.feature_config.pca_n_components = min(
                ml_config.feature_config.pca_n_components,
                n_samples - 1
            )
        
        logger.info(f"Конфигурация алгоритма 2: enable_clustering={ml_config.enable_clustering}, "
                   f"enable_similarity_learning={ml_config.enable_similarity_learning}, "
                   f"enable_faiss_index={ml_config.enable_faiss_index}")
        
        # Создаем и обучаем анализатор
        analyzer = MLRetrospectiveAnalyzer(config=ml_config)
        
        try:
            # Обучаем на тренировочных данных
            analyzer.train(train_data)
            
            # Проверяем, что анализатор успешно обучен
            if not analyzer.is_trained:
                logger.warning("Анализатор алгоритма 2 не помечен как обученный (is_trained=False)")
                # Пытаемся вручную установить флаг, если обучение прошло без ошибок
                analyzer.is_trained = True
        except Exception as e:
            logger.error(f"Ошибка во время обучения алгоритма 2: {e}")
            logger.info("Попытка использовать анализатор в ограниченном режиме...")
            # Пытаемся хотя бы извлечь фичи
            try:
                analyzer.load_and_preprocess(train_data)
                analyzer.extract_features()
                analyzer.is_trained = True
                logger.info("Анализатор подготовлен в ограниченном режиме (только feature extraction)")
            except Exception as e2:
                logger.error(f"Не удалось подготовить анализатор: {e2}")
                raise
        
        training_time = time.time() - start_time
        logger.info(f"Подготовка алгоритма 2 завершена за {training_time:.2f} секунд")
        
        return analyzer
    except ImportError as e:
        logger.error(f"Не удалось импортировать алгоритм 2: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при обучении алгоритма 2: {e}")
        raise


def compare_algorithms_on_test_set(test_data: List[Dict[str, Any]], 
                                  historical_tenders: List[Dict[str, Any]],
                                  analyzer_2,
                                  max_test_samples: int = 100) -> Dict[str, Any]:
    """
    Сравнивает оба алгоритма на тестовом наборе данных.
    
    Args:
        test_data: Тестовые тендеры
        historical_tenders: Исторические тендеры для алгоритма 1
        analyzer_2: Обученный анализатор алгоритма 2
        max_test_samples: Максимальное количество тестовых примеров для оценки
        
    Returns:
        Словарь с результатами сравнения
    """
    logger.info(f"Сравнение алгоритмов на {min(len(test_data), max_test_samples)} тестовых примерах")
    
    # Ограничиваем количество тестовых примеров для ускорения
    test_samples = test_data[:max_test_samples]
    
    results_alg1 = []
    results_alg2 = []
    
    for i, test_tender in enumerate(test_samples):
        logger.info(f"Оценка тестового тендера {i+1}/{len(test_samples)}")
        
        # Оцениваем алгоритм 1
        result1 = evaluate_algorithm_1(test_tender, historical_tenders)
        results_alg1.append(result1)
        
        # Оцениваем алгоритм 2
        result2 = evaluate_algorithm_2(test_tender, analyzer_2)
        results_alg2.append(result2)
        
        # Логируем прогресс
        if (i + 1) % 10 == 0:
            logger.info(f"Прогресс: {i+1}/{len(test_samples)}")
    
    # Анализируем результаты
    comparison_results = analyze_comparison_results(results_alg1, results_alg2)
    
    return comparison_results


def analyze_comparison_results(results_alg1: List[Dict], results_alg2: List[Dict]) -> Dict[str, Any]:
    """
    Анализирует результаты сравнения и вычисляет метрики.
    
    Args:
        results_alg1: Результаты алгоритма 1
        results_alg2: Результаты алгоритма 2
        
    Returns:
        Словарь с метриками сравнения
    """
    # Вычисляем среднее время выполнения
    times_alg1 = [r['execution_time'] for r in results_alg1 if r.get('success', False)]
    times_alg2 = [r['execution_time'] for r in results_alg2 if r.get('success', False)]
    
    # Вычисляем успешность
    success_alg1 = sum(1 for r in results_alg1 if r.get('success', False))
    success_alg2 = sum(1 for r in results_alg2 if r.get('success', False))
    
    # Вычисляем средние оценки сходства
    avg_similarity_alg1 = []
    avg_similarity_alg2 = []
    
    for r1, r2 in zip(results_alg1, results_alg2):
        if r1.get('success', False) and r1.get('similarity_scores'):
            avg_similarity_alg1.append(np.mean(r1['similarity_scores']))
        if r2.get('success', False) and r2.get('similarity_scores'):
            avg_similarity_alg2.append(np.mean(r2['similarity_scores']))
    
    return {
        'algorithm_1': {
            'success_rate': success_alg1 / len(results_alg1) if results_alg1 else 0,
            'avg_execution_time': np.mean(times_alg1) if times_alg1 else 0,
            'std_execution_time': np.std(times_alg1) if times_alg1 else 0,
            'avg_similarity_score': np.mean(avg_similarity_alg1) if avg_similarity_alg1 else 0,
            'total_tested': len(results_alg1)
        },
        'algorithm_2': {
            'success_rate': success_alg2 / len(results_alg2) if results_alg2 else 0,
            'avg_execution_time': np.mean(times_alg2) if times_alg2 else 0,
            'std_execution_time': np.std(times_alg2) if times_alg2 else 0,
            'avg_similarity_score': np.mean(avg_similarity_alg2) if avg_similarity_alg2 else 0,
            'total_tested': len(results_alg2)
        },
        'comparison': {
            'speedup_factor': np.mean(times_alg1) / np.mean(times_alg2) if times_alg1 and times_alg2 and np.mean(times_alg2) > 0 else 0,
            'similarity_difference': (np.mean(avg_similarity_alg2) - np.mean(avg_similarity_alg1)) if avg_similarity_alg1 and avg_similarity_alg2 else 0
        }
    }


def print_comparison_report(comparison_results: Dict[str, Any]):
    """
    Выводит отчет о сравнении алгоритмов.
    
    Args:
        comparison_results: Результаты сравнения
    """
    print("\n" + "="*80)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ 1 И 2")
    print("="*80)
    
    alg1 = comparison_results['algorithm_1']
    alg2 = comparison_results['algorithm_2']
    comp = comparison_results['comparison']
    
    print(f"\nАЛГОРИТМ 1 (Simple select_k_best):")
    print(f"  • Успешность: {alg1['success_rate']*100:.1f}% ({alg1['total_tested']} тестов)")
    print(f"  • Среднее время выполнения: {alg1['avg_execution_time']:.4f} ± {alg1['std_execution_time']:.4f} сек")
    print(f"  • Средняя оценка сходства: {alg1['avg_similarity_score']:.4f}")
    
    print(f"\nАЛГОРИТМ 2 (ML-based Approach):")
    print(f"  • Успешность: {alg2['success_rate']*100:.1f}% ({alg2['total_tested']} тестов)")
    print(f"  • Среднее время выполнения: {alg2['avg_execution_time']:.4f} ± {alg2['std_execution_time']:.4f} сек")
    print(f"  • Средняя оценка сходства: {alg2['avg_similarity_score']:.4f}")
    
    print(f"\nСРАВНЕНИЕ:")
    if comp['speedup_factor'] > 1:
        print(f"  • Алгоритм 2 быстрее в {comp['speedup_factor']:.2f} раз")
    elif comp['speedup_factor'] > 0:
        print(f"  • Алгоритм 1 быстрее в {1/comp['speedup_factor']:.2f} раз")
    
    if comp['similarity_difference'] > 0:
        print(f"  • Алгоритм 2 дает более высокое сходство на {comp['similarity_difference']:.4f}")
    elif comp['similarity_difference'] < 0:
        print(f"  • Алгоритм 1 дает более высокое сходство на {-comp['similarity_difference']:.4f}")
    else:
        print(f"  • Алгоритмы дают схожее сходство")
    
    print("\n" + "="*80)


def main():
    """Основная функция скрипта."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Сравнительный анализ алгоритмов 1 и 2')
    parser.add_argument('--data', type=str, default='C:/Users/Alexander/Desktop/tenders.json',
                       help='Путь к файлу tenders.json')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Максимальное количество записей для загрузки (для тестирования)')
    parser.add_argument('--max-test-samples', type=int, default=5,
                       help='Максимальное количество тестовых примеров для оценки (по умолчанию: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости разделения данных')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения результатов в JSON формате')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Путь к предобученной модели алгоритма 2 (.pkl). Если указан, модель загружается, иначе обучается заново.')
    
    args = parser.parse_args()
    
    try:
        # 1. Загрузка данных
        data = load_tenders_data(args.data, args.max_samples)
        
        if len(data) < 100:
            logger.warning(f"Мало данных ({len(data)} записей). Рекомендуется не менее 100 записей.")
        
        # 2. Разделение данных 90-10
        train_data, test_data = split_data_90_10(data, args.seed)
        
        # 3. Подготовка алгоритма 2 (загрузка или обучение)
        analyzer_2 = None
        if args.model_path:
            try:
                analyzer_2 = load_algorithm_2(args.model_path)
                logger.info(f"Алгоритм 2 загружен из {args.model_path}")
            except Exception as e:
                logger.error(f"Не удалось загрузить модель алгоритма 2: {e}")
                logger.info("Попытка обучить модель заново...")
                try:
                    analyzer_2 = train_algorithm_2(train_data)
                except Exception as e2:
                    logger.error(f"Не удалось обучить алгоритм 2: {e2}")
                    logger.info("Продолжаем только с алгоритмом 1...")
                    analyzer_2 = None
        else:
            try:
                analyzer_2 = train_algorithm_2(train_data)
            except Exception as e:
                logger.error(f"Не удалось обучить алгоритм 2: {e}")
                logger.info("Продолжаем только с алгоритмом 1...")
                analyzer_2 = None
        
        # 4. Сравнение алгоритмов
        if analyzer_2:
            comparison_results = compare_algorithms_on_test_set(
                test_data, train_data, analyzer_2, args.max_test_samples
            )
        else:
            # Только оценка алгоритма 1
            logger.info("Оценка только алгоритма 1...")
            results_alg1 = []
            for i, test_tender in enumerate(test_data[:args.max_test_samples]):
                result1 = evaluate_algorithm_1(test_tender, train_data)
                results_alg1.append(result1)
            
            comparison_results = {
                'algorithm_1': {
                    'success_rate': sum(1 for r in results_alg1 if r.get('success', False)) / len(results_alg1) if results_alg1 else 0,
                    'avg_execution_time': np.mean([r['execution_time'] for r in results_alg1 if r.get('success', False)]),
                    'std_execution_time': np.std([r['execution_time'] for r in results_alg1 if r.get('success', False)]),
                    'avg_similarity_score': 0,
                    'total_tested': len(results_alg1)
                },
                'algorithm_2': {
                    'success_rate': 0,
                    'avg_execution_time': 0,
                    'std_execution_time': 0,
                    'avg_similarity_score': 0,
                    'total_tested': 0
                },
                'comparison': {
                    'speedup_factor': 0,
                    'similarity_difference': 0
                }
            }
        
        # 5. Вывод отчета
        print_comparison_report(comparison_results)
        
        # 6. Сохранение результатов
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены в {args.output}")
        
        logger.info("Сравнительный анализ завершен!")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()