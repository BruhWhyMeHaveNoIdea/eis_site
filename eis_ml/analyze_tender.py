#!/usr/bin/env python3
"""
Скрипт для анализа тендеров на основе двух подходов из README.md.

Использует:
1. SimplePricePredictor с MLRetrospectiveAnalyzer (первый блок кода)
2. Прямой анализ через MLRetrospectiveAnalyzer.find_similar() (второй блок кода)

Скрипт берет на вход JSON файл с одним тендером, загружает исторические данные
из tenders.json, обучает анализатор и выполняет оба подхода.
"""

import pprint
import json
import sys
import os
import argparse
import logging
import traceback
from typing import Dict, List, Any, Optional
import time

# Настройка кодировки UTF-8 для вывода в консоль (Windows)
# Не применяем при работе через subprocess с JSON выводом
if sys.platform == 'win32' and '--json' not in sys.argv:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Импорт необходимых модулей проекта
try:
    from price_predictor_simple import SimplePricePredictor
    from ml_retrospective import MLRetrospectiveAnalyzer
    from core.preprocessing import preprocess_tender
except ImportError as e:
    pprint.pprint(f"Ошибка импорта модулей: {e}")
    pprint.pprint("Убедитесь, что вы находитесь в корневой директории проекта.")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _output_result(result: Dict[str, Any], output_json: bool = False):
    """Вывод результатов анализа в JSON или текстовом формате."""
    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        pprint.pprint(f"\n{'='*80}")
        pprint.pprint("СВОДКА АНАЛИЗА")
        pprint.pprint(f"{'='*80}")
        
        total_time = time.time() - result.get('metadata', {}).get('start_time', time.time())
        pprint.pprint(f"Общее время выполнения: {total_time:.2f} секунд")
        
        if result.get('analyzer_trained'):
            pprint.pprint(f"Модель успешно обучена на {result.get('metadata', {}).get('historical_tenders_count', 0)} тендерах")
        
        pprint.pprint(f"\nАнализ завершен. Результаты выведены выше.")


def load_tenders(filepath: str) -> List[Dict[str, Any]]:
    """Загрузить тендеры из JSON файла."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Если данные - словарь с ключом 'tenders' или подобным
        if isinstance(data, dict):
            # Попробуем найти ключ с тендерами
            for key in ['tenders', 'data', 'items']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Если не нашли, вернем весь словарь как список из одного элемента
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Неподдерживаемый формат данных в {filepath}")
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {filepath}: {e}")
        raise


def save_model_if_needed(analyzer: MLRetrospectiveAnalyzer, model_path: str = None):
    """Сохранить модель, если указан путь."""
    if model_path:
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Используем метод save() вместо save_model()
            analyzer.save(model_path)
            logger.info(f"Модель сохранена в {model_path}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить модель: {e}")


def analyze_tender(new_tender_file: str, historical_data_file: str = "tenders.json",
                   k: int = 5, save_model_path: Optional[str] = None,
                   model_path: Optional[str] = None, force_retrain: bool = False,
                   output_json: bool = False):
    """
    Основная функция анализа тендера.
    
    Args:
        new_tender_file: Путь к JSON файлу с новым тендером
        historical_data_file: Путь к JSON файлу с историческими данными
        k: Количество похожих тендеров для поиска
        save_model_path: Путь для сохранения модели (опционально)
        model_path: Путь к предобученной модели (если None, пробуем models/ml_retrospective_model.pkl)
        force_retrain: Принудительно переобучить модель, даже если есть готовая
        output_json: Если True, выводит результаты в формате JSON
    """
    
    # Структура для сбора результатов
    analysis_result = {
        'success': True,
        'tender_info': {},
        'price_prediction': None,
        'similar_tenders': [],
        'metadata': {}
    }
    
    if not output_json:
        pprint.pprint("=" * 80)
        pprint.pprint("АНАЛИЗ ТЕНДЕРА")
        pprint.pprint("=" * 80)
    
    # 1. Загрузить исторические данные (могут понадобиться для обучения)
    if not output_json:
        pprint.pprint(f"\n1. Загрузка исторических данных из {historical_data_file}...")
    start_time = time.time()
    historical_tenders = []
    try:
        historical_tenders = load_tenders(historical_data_file)
        if not output_json:
            pprint.pprint(f"   Загружено {len(historical_tenders)} исторических тендеров")
        analysis_result['metadata']['historical_tenders_count'] = len(historical_tenders)
    except Exception as e:
        if not output_json:
            pprint.pprint(f"   Внимание: Не удалось загрузить исторические данные: {e}")
        if force_retrain or not model_path:
            if not output_json:
                pprint.pprint("   ОШИБКА: Исторические данные необходимы для обучения модели")
            analysis_result['success'] = False
            analysis_result['error'] = "Нет исторических данных для обучения"
            _output_result(analysis_result, output_json)
            return
    
    # 2. Загрузить новый тендер
    if not output_json:
        pprint.pprint(f"\n2. Загрузка нового тендера из {new_tender_file}...")
    try:
        new_tenders = load_tenders(new_tender_file)
        if len(new_tenders) == 0:
            if not output_json:
                pprint.pprint("   ОШИБКА: Файл не содержит тендеров")
            analysis_result['success'] = False
            analysis_result['error'] = "Файл не содержит тендеров"
            _output_result(analysis_result, output_json)
            return
        new_tender = new_tenders[0]
        if len(new_tenders) > 1:
            if not output_json:
                pprint.pprint(f"   Внимание: файл содержит {len(new_tenders)} тендеров, анализируем первый")
        
        # Сохранить информацию о тендере в результат
        analysis_result['tender_info'] = {
            'name': new_tender.get('Наименование объекта закупки', 'Не указано'),
            'customer': new_tender.get('Организация, осуществляющая размещение', new_tender.get('Заказчик', 'Не указано')),
            'price': new_tender.get('Начальная (максимальная) цена контракта', 'Не указана'),
            'region': new_tender.get('Регион', 'Не указан'),
            'procurement_method': new_tender.get('Способ определения поставщика (подрядчика, исполнителя)', 'Не указан')
        }
        
        # Вывести информацию о новом тендере
        if not output_json:
            tender_name = new_tender.get('Наименование объекта закупки', 'Не указано')
            tender_customer = new_tender.get('Заказчик', 'Не указано')
            tender_price = new_tender.get('Начальная (максимальная) цена контракта', 'Не указана')
            pprint.pprint(f"   Новый тендер: {tender_name}")
            pprint.pprint(f"   Заказчик: {tender_customer}")
            pprint.pprint(f"   Цена: {tender_price}")
    except Exception as e:
        if not output_json:
            pprint.pprint(f"   ОШИБКА: Не удалось загрузить новый тендер: {e}")
        analysis_result['success'] = False
        analysis_result['error'] = f"Не удалось загрузить новый тендер: {e}"
        _output_result(analysis_result, output_json)
        return
    
    # 3. Инициализировать анализатор алгоритма 2
    if not output_json:
        pprint.pprint(f"\n3. Инициализация MLRetrospectiveAnalyzer...")
    analyzer = None
    
    # Определить путь к модели по умолчанию
    default_model_path = "models/ml_retrospective_model.pkl"
    if model_path is None and os.path.exists(default_model_path):
        model_path = default_model_path
    
    # Попытка загрузить существующую модель
    if not force_retrain and model_path and os.path.exists(model_path):
        try:
            if not output_json:
                pprint.pprint(f"   Загрузка предобученной модели из {model_path}...")
            # Используем классовый метод load()
            analyzer = MLRetrospectiveAnalyzer.load(model_path)
            if not output_json:
                pprint.pprint("   Модель успешно загружена")
            analysis_result['metadata']['model_loaded_from'] = model_path
        except Exception as e:
            if not output_json:
                pprint.pprint(f"   Внимание: Не удалось загрузить модель: {e}")
                pprint.pprint("   Пробуем обучить модель заново...")
            analyzer = None
    
    # Если модель не загружена, обучаем заново
    if analyzer is None:
        if not historical_tenders:
            if not output_json:
                pprint.pprint("   ОШИБКА: Нет исторических данных для обучения модели")
            analysis_result['success'] = False
            analysis_result['error'] = "Нет исторических данных для обучения модели"
            _output_result(analysis_result, output_json)
            return
        
        try:
            analyzer = MLRetrospectiveAnalyzer()
            if not output_json:
                pprint.pprint("   Обучение модели...")
            analyzer.train(historical_tenders)
            if not output_json:
                pprint.pprint("   Обучение завершено успешно")
            analysis_result['metadata']['model_trained'] = True
            
            # Сохранить модель, если указан путь
            if save_model_path:
                save_model_if_needed(analyzer, save_model_path)
            elif model_path and not os.path.exists(model_path):
                # Сохранить в путь по умолчанию, если его нет
                save_model_if_needed(analyzer, model_path)
        except Exception as e:
            if not output_json:
                pprint.pprint(f"   ОШИБКА: Не удалось обучить анализатор: {e}")
            analysis_result['success'] = False
            analysis_result['error'] = f"Не удалось обучить модель: {e}"
            import traceback
            traceback.print_exc()
            if not output_json:
                pprint.pprint("   Продолжаем с упрощенным подходом...")
            analyzer = None
    
    # 4. Сохранить модель, если требуется (дополнительно)
    if analyzer and save_model_path and save_model_path != model_path:
        save_model_if_needed(analyzer, save_model_path)
        analysis_result['metadata']['model_saved_to'] = save_model_path
    
    # 5. ПЕРВЫЙ ПОДХОД: SimplePricePredictor с MLRetrospectiveAnalyzer
    if not output_json:
        pprint.pprint(f"\n{'='*80}")
        pprint.pprint("ПЕРВЫЙ ПОДХОД: SimplePricePredictor с MLRetrospectiveAnalyzer")
        pprint.pprint(f"{'='*80}")
    
    if analyzer:
        try:
            # Создать предиктор
            predictor = SimplePricePredictor(analyzer, method='weighted_median')
            
            # Предсказать цену для нового тендера
            if not output_json:
                pprint.pprint(f"\nПрогнозирование цены методом weighted_median (k={k})...")
            result = predictor.predict(new_tender, k=k)
            
            # Сохранить результат прогнозирования
            analysis_result['price_prediction'] = {
                'predicted_price': result.get('predicted_price'),
                'method': result.get('method', 'weighted_median'),
                'confidence': result.get('confidence', 0),
                'price_range': list(result.get('price_range', [None, None])),
                'similar_tenders_used': result.get('similar_tenders_used', 0),
                'similar_tenders_total': result.get('similar_tenders_total', 0)
            }
            
            if not output_json:
                # Вывести результаты
                pprint.pprint(f"\nРезультаты прогнозирования цены:")
                pprint.pprint(f"  Прогнозируемая цена: {result.get('predicted_price', 'N/A'):,.2f}" 
                      if result.get('predicted_price') else "  Прогнозируемая цена: N/A")
                pprint.pprint(f"  Метод: {result.get('method', 'N/A')}")
                pprint.pprint(f"  Уверенность: {result.get('confidence', 0):.2%}")
                
                price_range = result.get('price_range', (None, None))
                if price_range[0] is not None and price_range[1] is not None:
                    pprint.pprint(f"  Диапазон цен похожих тендеров: {price_range[0]:,.2f} - {price_range[1]:,.2f}")
                
                similar_used = result.get('similar_tenders_used', 0)
                similar_total = result.get('similar_tenders_total', 0)
                pprint.pprint(f"  Использовано похожих тендеров: {similar_used} из {similar_total}")
                
                # Вывести детали похожих тендеров
                similar_tenders = result.get('similar_tenders', [])
                if similar_tenders:
                    pprint.pprint(f"\n  Детали похожих тендеров:")
                    for i, tender_info in enumerate(similar_tenders[:3]):  # Покажем только первые 3
                        tender = tender_info.get('tender', {})
                        similarity = tender_info.get('similarity_score', 0)
                        price = tender.get('Начальная (максимальная) цена контракта', 'N/A')
                        name = tender.get('Наименование объекта закупки', 'N/A')[:50] + "..." if len(tender.get('Наименование объекта закупки', '')) > 50 else tender.get('Наименование объекта закупки', 'N/A')
                        pprint.pprint(f"    {i+1}. Сходство: {similarity:.4f}, Цена: {price}")
                        pprint.pprint(f"       Наименование: {name}")
            
        except Exception as e:
            if not output_json:
                pprint.pprint(f"  ОШИБКА в SimplePricePredictor: {e}")
            analysis_result['price_prediction_error'] = str(e)
            import traceback
            traceback.print_exc()
    else:
        if not output_json:
            pprint.pprint("  Пропущено (анализатор не обучен)")
    
    # 6. ВТОРОЙ ПОДХОД: прямой анализ через MLRetrospectiveAnalyzer.find_similar()
    if not output_json:
        pprint.pprint(f"\n{'='*80}")
        pprint.pprint("ВТОРОЙ ПОДХОД: прямой анализ через MLRetrospectiveAnalyzer.find_similar()")
        pprint.pprint(f"{'='*80}")
    
    if analyzer:
        try:
            # Найти похожие тендеры с использованием обученной модели
            if not output_json:
                pprint.pprint(f"\nПоиск похожих тендеров (k={k})...")
            results = analyzer.find_similar_tenders(new_tender, k=k)
            
            # Вывести результаты
            similar_tenders = results.get('similar_tenders', [])
            if not output_json:
                pprint.pprint(f"\nНайдено {len(similar_tenders)} похожих тендеров")
            
            # Сохранить похожие тендеры в результат
            for i, tender_info in enumerate(similar_tenders):
                tender = tender_info.get('tender', {})
                similarity = tender_info.get('similarity_score', 0)
                cluster = tender_info.get('cluster', 'N/A')
                
                # Извлечь ключевые поля
                name = tender.get('Наименование объекта закупки', 'N/A')
                customer = tender.get('Организация, осуществляющая размещение', 'N/A')
                price = tender.get('Начальная (максимальная) цена контракта', 'N/A')
                region = tender.get('Регион', 'N/A')
                method = tender.get('Способ определения поставщика (подрядчика, исполнителя)', 'N/A')
                
                similar_tender_data = {
                    'rank': i + 1,
                    'similarity_score': similarity,
                    'cluster': cluster,
                    'tender_name': name,
                    'customer': customer,
                    'price': price,
                    'region': region,
                    'procurement_method': method
                }
                
                if 'similarity_breakdown' in tender_info:
                    similar_tender_data['similarity_breakdown'] = tender_info['similarity_breakdown']
                
                analysis_result['similar_tenders'].append(similar_tender_data)
                
                if not output_json:
                    if len(name) > 60:
                        name = name[:57] + "..."
                    if len(customer) > 40:
                        customer = customer[:37] + "..."
                    
                    pprint.pprint(f"\n{i+1}. Сходство: {similarity:.4f}, Кластер: {cluster}")
                    pprint.pprint(f"   Наименование: {name}")
                    pprint.pprint(f"   Заказчик: {customer}")
                    pprint.pprint(f"   Цена: {price}")
                    pprint.pprint(f"   Регион: {region}")
                    pprint.pprint(f"   Способ закупки: {method}")
                    
                    # Дополнительная информация, если есть
                    if 'similarity_breakdown' in tender_info:
                        breakdown = tender_info['similarity_breakdown']
                        if isinstance(breakdown, dict):
                            top_fields = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
                            if top_fields:
                                pprint.pprint(f"   Ключевые факторы сходства: {', '.join([f'{k}: {v:.3f}' for k, v in top_fields])}")
            
            # Вывести метаинформацию
            if 'metadata' in results:
                metadata = results['metadata']
                analysis_result['search_metadata'] = metadata
                if not output_json:
                    pprint.pprint(f"\nМетаинформация поиска:")
                    pprint.pprint(f"  Время выполнения: {metadata.get('search_time_ms', 0):.1f} мс")
                    pprint.pprint(f"  Всего кандидатов: {metadata.get('total_candidates', 0)}")
                    pprint.pprint(f"  Отфильтровано: {metadata.get('filtered_candidates', 0)}")
        
        except Exception as e:
            if not output_json:
                pprint.pprint(f"  ОШИБКА в find_similar: {e}")
            analysis_result['similar_tenders_error'] = str(e)
            import traceback
            traceback.print_exc()
    else:
        if not output_json:
            pprint.pprint("  Пропущено (анализатор не обучен)")
    
    # 7. Сводка
    analysis_result['metadata']['execution_time_seconds'] = time.time() - start_time
    analysis_result['metadata']['start_time'] = start_time
    
    if analyzer:
        analysis_result['analyzer_trained'] = True
        analysis_result['metadata']['historical_tenders_count'] = len(historical_tenders)
        if save_model_path:
            analysis_result['metadata']['model_saved_to'] = save_model_path
    else:
        analysis_result['analyzer_trained'] = False
    
    _output_result(analysis_result, output_json)


def main():
    """Основная функция для обработки аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Анализ тендера с использованием двух подходов из README.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s new_tender.json
  %(prog)s new_tender.json --historical custom_tenders.json
  %(prog)s new_tender.json --k 10 --save-model models/my_model.pkl
        """
    )
    
    parser.add_argument(
        'new_tender',
        help='Путь к JSON файлу с новым тендером для анализа'
    )
    
    parser.add_argument(
        '--historical', '-H',
        default='tenders.json',
        help='Путь к JSON файлу с историческими данными (по умолчанию: tenders.json)'
    )
    
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=5,
        help='Количество похожих тендеров для поиска (по умолчанию: 5)'
    )
    
    parser.add_argument(
        '--save-model',
        help='Путь для сохранения обученной модели (опционально)'
    )
    
    parser.add_argument(
        '--model',
        '-m',
        help='Путь к предобученной модели (по умолчанию: models/ml_retrospective_model.pkl)'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Принудительно переобучить модель, даже если есть готовая'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод (логирование DEBUG)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Вывод результатов в формате JSON (для API)'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Проверка существования файлов
    if not os.path.exists(args.new_tender):
        pprint.pprint(f"ОШИБКА: Файл с новым тендером не найден: {args.new_tender}")
        sys.exit(1)

    # Проверка исторических данных, только если нужны для обучения
    if (args.force_retrain or not args.model) and not os.path.exists(args.historical):
        pprint.pprint(f"ОШИБКА: Файл с историческими данными не найден: {args.historical}")
        pprint.pprint("Убедитесь, что файл tenders.json существует в текущей директории.")
        pprint.pprint("Или используйте предобученную модель с опцией --model")
        sys.exit(1)

    # Запуск анализа
    try:
        analyze_tender(
            new_tender_file=args.new_tender,
            historical_data_file=args.historical,
            k=args.k,
            save_model_path=args.save_model,
            model_path=args.model,
            force_retrain=args.force_retrain,
            output_json=args.json
        )
    except KeyboardInterrupt:
        pprint.pprint("\n\nАнализ прерван пользователем.")
        sys.exit(1)
    except Exception as e:
        print(f"\nКритическая ошибка при выполнении анализа: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()