#!/usr/bin/env python3
"""
Скрипт для анализа тендеров на основе двух подходов из README.md.

Использует:
1. SimplePricePredictor с MLRetrospectiveAnalyzer (первый блок кода)
2. Прямой анализ через MLRetrospectiveAnalyzer.find_similar() (второй блок кода)

Скрипт берет на вход JSON файл с одним тендером, загружает исторические данные
из tenders.json, обучает анализатор и выполняет оба подхода.
"""

import json
import sys
import os
import argparse
import logging
from typing import Dict, List, Any, Optional
import time

# Настройка кодировки UTF-8 для вывода в консоль (Windows)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Импорт необходимых модулей проекта
try:
    from price_predictor_simple import SimplePricePredictor
    from ml_retrospective import MLRetrospectiveAnalyzer
    from core.preprocessing import preprocess_tender
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print("Убедитесь, что вы находитесь в корневой директории проекта.")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                   model_path: Optional[str] = None, force_retrain: bool = False):
    """
    Основная функция анализа тендера.
    
    Args:
        new_tender_file: Путь к JSON файлу с новым тендером
        historical_data_file: Путь к JSON файлу с историческими данными
        k: Количество похожих тендеров для поиска
        save_model_path: Путь для сохранения модели (опционально)
        model_path: Путь к предобученной модели (если None, пробуем models/ml_retrospective_model.pkl)
        force_retrain: Принудительно переобучить модель, даже если есть готовая
    """
    print("=" * 80)
    print("АНАЛИЗ ТЕНДЕРА")
    print("=" * 80)
    
    # 1. Загрузить исторические данные (могут понадобиться для обучения)
    print(f"\n1. Загрузка исторических данных из {historical_data_file}...")
    start_time = time.time()
    historical_tenders = []
    try:
        historical_tenders = load_tenders(historical_data_file)
        print(f"   Загружено {len(historical_tenders)} исторических тендеров")
    except Exception as e:
        print(f"   Внимание: Не удалось загрузить исторические данные: {e}")
        if force_retrain or not model_path:
            print("   ОШИБКА: Исторические данные необходимы для обучения модели")
            return
    
    # 2. Загрузить новый тендер
    print(f"\n2. Загрузка нового тендера из {new_tender_file}...")
    try:
        new_tenders = load_tenders(new_tender_file)
        if len(new_tenders) == 0:
            print("   ОШИБКА: Файл не содержит тендеров")
            return
        new_tender = new_tenders[0]
        if len(new_tenders) > 1:
            print(f"   Внимание: файл содержит {len(new_tenders)} тендеров, анализируем первый")
        
        # Вывести информацию о новом тендере
        tender_name = new_tender.get('Наименование объекта закупки', 'Не указано')
        tender_customer = new_tender.get('Заказчик', 'Не указано')
        tender_price = new_tender.get('Начальная (максимальная) цена контракта', 'Не указана')
        print(f"   Новый тендер: {tender_name}")
        print(f"   Заказчик: {tender_customer}")
        print(f"   Цена: {tender_price}")
    except Exception as e:
        print(f"   ОШИБКА: Не удалось загрузить новый тендер: {e}")
        return
    
    # 3. Инициализировать анализатор алгоритма 2
    print(f"\n3. Инициализация MLRetrospectiveAnalyzer...")
    analyzer = None
    
    # Определить путь к модели по умолчанию
    default_model_path = "models/ml_retrospective_model.pkl"
    if model_path is None and os.path.exists(default_model_path):
        model_path = default_model_path
    
    # Попытка загрузить существующую модель
    if not force_retrain and model_path and os.path.exists(model_path):
        try:
            print(f"   Загрузка предобученной модели из {model_path}...")
            # Используем классовый метод load()
            analyzer = MLRetrospectiveAnalyzer.load(model_path)
            print("   Модель успешно загружена")
        except Exception as e:
            print(f"   Внимание: Не удалось загрузить модель: {e}")
            print("   Пробуем обучить модель заново...")
            analyzer = None
    
    # Если модель не загружена, обучаем заново
    if analyzer is None:
        if not historical_tenders:
            print("   ОШИБКА: Нет исторических данных для обучения модели")
            return
        
        try:
            analyzer = MLRetrospectiveAnalyzer()
            print("   Обучение модели...")
            analyzer.train(historical_tenders)
            print("   Обучение завершено успешно")
            
            # Сохранить модель, если указан путь
            if save_model_path:
                save_model_if_needed(analyzer, save_model_path)
            elif model_path and not os.path.exists(model_path):
                # Сохранить в путь по умолчанию, если его нет
                save_model_if_needed(analyzer, model_path)
        except Exception as e:
            print(f"   ОШИБКА: Не удалось обучить анализатор: {e}")
            import traceback
            traceback.print_exc()
            print("   Продолжаем с упрощенным подходом...")
            analyzer = None
    
    # 4. Сохранить модель, если требуется (дополнительно)
    if analyzer and save_model_path and save_model_path != model_path:
        save_model_if_needed(analyzer, save_model_path)
    
    # 5. ПЕРВЫЙ ПОДХОД: SimplePricePredictor с MLRetrospectiveAnalyzer
    print(f"\n{'='*80}")
    print("ПЕРВЫЙ ПОДХОД: SimplePricePredictor с MLRetrospectiveAnalyzer")
    print(f"{'='*80}")
    
    if analyzer:
        try:
            # Создать предиктор
            predictor = SimplePricePredictor(analyzer, method='weighted_median')
            
            # Предсказать цену для нового тендера
            print(f"\nПрогнозирование цены методом weighted_median (k={k})...")
            result = predictor.predict(new_tender, k=k)
            
            # Вывести результаты
            print(f"\nРезультаты прогнозирования цены:")
            print(f"  Прогнозируемая цена: {result.get('predicted_price', 'N/A'):,.2f}" 
                  if result.get('predicted_price') else "  Прогнозируемая цена: N/A")
            print(f"  Метод: {result.get('method', 'N/A')}")
            print(f"  Уверенность: {result.get('confidence', 0):.2%}")
            
            price_range = result.get('price_range', (None, None))
            if price_range[0] is not None and price_range[1] is not None:
                print(f"  Диапазон цен похожих тендеров: {price_range[0]:,.2f} - {price_range[1]:,.2f}")
            
            similar_used = result.get('similar_tenders_used', 0)
            similar_total = result.get('similar_tenders_total', 0)
            print(f"  Использовано похожих тендеров: {similar_used} из {similar_total}")
            
            # Вывести детали похожих тендеров
            similar_tenders = result.get('similar_tenders', [])
            if similar_tenders:
                print(f"\n  Детали похожих тендеров:")
                for i, tender_info in enumerate(similar_tenders[:3]):  # Покажем только первые 3
                    tender = tender_info.get('tender', {})
                    similarity = tender_info.get('similarity_score', 0)
                    price = tender.get('Начальная (максимальная) цена контракта', 'N/A')
                    name = tender.get('Наименование объекта закупки', 'N/A')[:50] + "..." if len(tender.get('Наименование объекта закупки', '')) > 50 else tender.get('Наименование объекта закупки', 'N/A')
                    print(f"    {i+1}. Сходство: {similarity:.4f}, Цена: {price}")
                    print(f"       Наименование: {name}")
            
        except Exception as e:
            print(f"  ОШИБКА в SimplePricePredictor: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Пропущено (анализатор не обучен)")
    
    # 6. ВТОРОЙ ПОДХОД: прямой анализ через MLRetrospectiveAnalyzer.find_similar()
    print(f"\n{'='*80}")
    print("ВТОРОЙ ПОДХОД: прямой анализ через MLRetrospectiveAnalyzer.find_similar()")
    print(f"{'='*80}")
    
    if analyzer:
        try:
            # Найти похожие тендеры с использованием обученной модели
            print(f"\nПоиск похожих тендеров (k={k})...")
            results = analyzer.find_similar_tenders(new_tender, k=k)
            
            # Вывести результаты
            similar_tenders = results.get('similar_tenders', [])
            print(f"\nНайдено {len(similar_tenders)} похожих тендеров")
            
            for i, tender_info in enumerate(similar_tenders):
                tender = tender_info.get('tender', {})
                similarity = tender_info.get('similarity_score', 0)
                cluster = tender_info.get('cluster', 'N/A')
                
                # Извлечь ключевые поля
                name = tender.get('Наименование объекта закупки', 'N/A')
                if len(name) > 60:
                    name = name[:57] + "..."
                
                customer = tender.get('Заказчик', 'N/A')
                if len(customer) > 40:
                    customer = customer[:37] + "..."
                
                price = tender.get('Начальная (максимальная) цена контракта', 'N/A')
                region = tender.get('Регион', 'N/A')
                method = tender.get('Способ определения поставщика (подрядчика, исполнителя)', 'N/A')
                
                print(f"\n{i+1}. Сходство: {similarity:.4f}, Кластер: {cluster}")
                print(f"   Наименование: {name}")
                print(f"   Заказчик: {customer}")
                print(f"   Цена: {price}")
                print(f"   Регион: {region}")
                print(f"   Способ закупки: {method}")
                
                # Дополнительная информация, если есть
                if 'similarity_breakdown' in tender_info:
                    breakdown = tender_info['similarity_breakdown']
                    if isinstance(breakdown, dict):
                        top_fields = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
                        if top_fields:
                            print(f"   Ключевые факторы сходства: {', '.join([f'{k}: {v:.3f}' for k, v in top_fields])}")
            
            # Вывести метаинформацию
            if 'metadata' in results:
                metadata = results['metadata']
                print(f"\nМетаинформация поиска:")
                print(f"  Время выполнения: {metadata.get('search_time_ms', 0):.1f} мс")
                print(f"  Всего кандидатов: {metadata.get('total_candidates', 0)}")
                print(f"  Отфильтровано: {metadata.get('filtered_candidates', 0)}")
        
        except Exception as e:
            print(f"  ОШИБКА в find_similar: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Пропущено (анализатор не обучен)")
    
    # 7. Сводка
    print(f"\n{'='*80}")
    print("СВОДКА АНАЛИЗА")
    print(f"{'='*80}")
    
    total_time = time.time() - start_time
    print(f"Общее время выполнения: {total_time:.2f} секунд")
    
    if analyzer:
        print(f"Модель успешно обучена на {len(historical_tenders)} тендерах")
        if save_model_path:
            print(f"Модель сохранена в: {save_model_path}")
    else:
        print(f"Модель не обучена, анализ ограничен")
    
    print(f"\nАнализ завершен. Результаты выведены выше.")


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
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Проверка существования файлов
    if not os.path.exists(args.new_tender):
        print(f"ОШИБКА: Файл с новым тендером не найден: {args.new_tender}")
        sys.exit(1)
    
    # Проверка исторических данных, только если нужны для обучения
    if (args.force_retrain or not args.model) and not os.path.exists(args.historical):
        print(f"ОШИБКА: Файл с историческими данными не найден: {args.historical}")
        print("Убедитесь, что файл tenders.json существует в текущей директории.")
        print("Или используйте предобученную модель с опцией --model")
        sys.exit(1)
    
    # Запуск анализа
    try:
        analyze_tender(
            new_tender_file=args.new_tender,
            historical_data_file=args.historical,
            k=args.k,
            save_model_path=args.save_model,
            model_path=args.model,
            force_retrain=args.force_retrain
        )
    except KeyboardInterrupt:
        print("\n\nАнализ прерван пользователем.")
        sys.exit(1)
    except Exception as e:
        print(f"\nКритическая ошибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()