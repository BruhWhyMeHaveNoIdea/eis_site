from django.shortcuts import render
from django.conf import settings
import subprocess
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
import os
import sys
from datetime import datetime
from django.db.models import Q
from django.db.models.functions import Lower, Trim
from .models import Tenders
from .utils import update_tenders_from_data





@api_view(["GET"])
def analyze_tender_api(request, tender_id):
    """
    API endpoint для ML-анализа конкретного тендера.
    Запускает analyze_tender.py через subprocess.
    
    Особенности:
    - Кеширование результатов на 24 часа
    - Максимум 2 одновременных анализа
    - Блокировка повторного анализа одного тендера
    """
    # === 1. Проверяем кеш ===
    cache_key = get_ml_cache_key(tender_id)
    cached_result = cache.get(cache_key)
    if cached_result:
        return Response({**cached_result, "from_cache": True})
    
    # === 2. Проверяем, не анализируется ли уже этот тендер ===
    lock_key = get_ml_analysis_lock_key(tender_id)
    if cache.get(lock_key):
        return Response(
            {"error": "Анализ этого тендера уже выполняется. Подождите немного."}, 
            status=409
        )
    
    # === 3. Проверяем лимит одновременных анализов ===
    concurrent_count = cache.get(ML_CONCURRENT_KEY, 0)
    if concurrent_count >= MAX_CONCURRENT_ML:
        return Response(
            {"error": f"Превышен лимит одновременных анализов ({MAX_CONCURRENT_ML}). Попробуйте позже."}, 
            status=429
        )
        
    # Увеличиваем счётчик
    cache.set(ML_CONCURRENT_KEY, concurrent_count + 1, 300)  # 5 минут таймаут
    
    try:
        # Устанавливаем блокировку на 3 минуты
        cache.set(lock_key, True, 180)
        
        # Получаем тендер из БД
        tender = Tenders.objects.get(id=tender_id)
        
        # Формируем данные тендера в формате, ожидаемом ML-модулем
        tender_data = {
            'Идентификационный код закупки (ИКЗ)': tender.icz,
            'Наименование объекта закупки': tender.object_name,
            'Организация, осуществляющая размещение': tender.customer_full_name,
            'Регион': tender.region,
            'Способ определения поставщика (подрядчика, исполнителя)': tender.procurement_method,
            'Этап закупки': tender.stage,
            'Начальная (максимальная) цена контракта': str(tender.initial_price) if tender.initial_price else '',
            'Срок исполнения контракта': tender.contract_execution_period,
            'Место поставки товара, выполнения работы или оказания услуги': tender.delivery_place,
            'Требования к участникам': tender.participant_requirements,
            'Преимущества': tender.advantages or '',
        }
        
        # Сохраняем временный файл с данными тендера
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump([tender_data], f, ensure_ascii=False)
            temp_tender_file = f.name
        
        try:
            # Путь к Python в виртуальном окружении (в корневой директории)
            python_path = os.path.join(settings.EIS_PARSER_ROOT, '.venv', 'Scripts', 'python.exe')
            script_path = os.path.join(settings.EIS_ML_ROOT, 'analyze_tender.py')
            historical_file = settings.EIS_ML_ROOT / 'tenders.json'
            model_path = settings.EIS_ML_ROOT / 'models' / 'ml_retrospective_model.pkl'
            
            # Проверяем существование файлов
            if not os.path.exists(python_path):
                return Response({"error": "Python в виртуальном окружении eis_ml не найден"}, status=500)
            
            if not os.path.exists(script_path):
                return Response({"error": "Скрипт analyze_tender.py не найден"}, status=500)
            
            if not os.path.exists(historical_file):
                return Response({"error": "Файл исторических данных tenders.json не найден"}, status=500)
            
            if not os.path.exists(model_path):
                return Response({"error": "Модель ML не найдена. Требуется обучение."}, status=500)
            
            # Запускаем анализ через subprocess
            result = subprocess.run(
                [
                    python_path, script_path,
                    temp_tender_file,
                    '--historical', str(historical_file),
                    '--model', str(model_path),
                    '--k', '5',
                    '--json'  # Вывод в формате JSON для API
                ],
                capture_output=True,
                encoding='utf-8',
                cwd=settings.EIS_ML_ROOT,
                errors='replace',
                timeout=120  # 2 минуты на анализ
            )
            if result.returncode == 0:
                ml_result = json.loads(result.stdout)
                
                # === 4. Сохраняем в кеш на 24 часа ===
                cache.set(cache_key, ml_result, 86400)
                
                return Response({**ml_result, "from_cache": False})
            else:
                print("STDERR", result.stderr)
                error_msg = result.stderr if result.stderr else 'Неизвестная ошибка ML-анализа'
                return Response({"error": error_msg}, status=500)
                
        finally:
            os.unlink(temp_tender_file)
        
    except subprocess.TimeoutExpired:
        return Response({"error": "Превышено время ожидания ML-анализа (120 сек)"}, status=500)
    except Tenders.DoesNotExist:
        return Response({"error": "Тендер не найден"}, status=404)
    except json.JSONDecodeError as e:
        return Response({"error": f"Ошибка парсинга ответа ML: {e}"}, status=500)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    finally:
        # Снимаем блокировку тендера
        cache.delete(lock_key)
        # Уменьшаем счётчик одновременных анализов
        current_count = cache.get(ML_CONCURRENT_KEY, 0)
        if current_count > 0:
            cache.set(ML_CONCURRENT_KEY, current_count - 1, 300)


def index(request):
    if not request.session.session_key:
        request.session.create()
    
    # Получаем фильтры из GET-параметров
    region = request.GET.get('region', '')
    procurement_method = request.GET.get('procurement_method', '')
    stage = request.GET.get('stage', '')
    end_date_from = request.GET.get('end_date_from', '')
    end_date_to = request.GET.get('end_date_to', '')
    price_min = request.GET.get('price_min', '')
    price_max = request.GET.get('price_max', '')
    search = request.GET.get('search', '')
    
    # Базовый queryset
    tenders = Tenders.objects.all()
    
    # Применяем фильтры
    if region:
        tenders = tenders.filter(region__icontains=region)
    if procurement_method:
        tenders = tenders.filter(procurement_method__icontains=procurement_method)
    if stage:
        tenders = tenders.filter(stage=stage)
    if end_date_from:
        try:
            date_from = datetime.strptime(end_date_from, '%Y-%m-%d')
            tenders = tenders.filter(end_date_time__date__gte=date_from.date())
        except ValueError:
            pass
    if end_date_to:
        try:
            date_to = datetime.strptime(end_date_to, '%Y-%m-%d')
            tenders = tenders.filter(end_date_time__date__lte=date_to.date())
        except ValueError:
            pass
    if price_min:
        try:
            tenders = tenders.filter(initial_price__gte=float(price_min))
        except ValueError:
            pass
    if price_max:
        try:
            tenders = tenders.filter(initial_price__lte=float(price_max))
        except ValueError:
            pass
    if search:
        tenders = tenders.filter(
            Q(object_name__icontains=search) |
            Q(customer_full_name__icontains=search) |
            Q(icz__icontains=search)
        )
    
    # Получаем уникальные значения для фильтров (нормализованные)
    regions = Tenders.objects.annotate(
        region_norm=Lower(Trim('region'))
    ).exclude(region_norm='').values_list('region_norm', flat=True).distinct().order_by('region_norm')
    
    methods = Tenders.objects.annotate(
        method_norm=Lower(Trim('procurement_method'))
    ).exclude(method_norm='').values_list('method_norm', flat=True).distinct().order_by('method_norm')
    
    stages = Tenders.objects.annotate(
        stage_norm=Lower(Trim('stage'))
    ).exclude(stage_norm='').values_list('stage_norm', flat=True).distinct().order_by('stage_norm')
    
    context = {
        'tenders': tenders,
        'regions': regions,
        'methods': methods,
        'stages': stages,
        'filters': {
            'region': region,
            'procurement_method': procurement_method,
            'stage': stage,
            'end_date_from': end_date_from,
            'end_date_to': end_date_to,
            'price_min': price_min,
            'price_max': price_max,
            'search': search,
        }
    }
    
    return render(request, 'index.html', context)




from django.core.cache import cache
import hashlib


def get_ml_cache_key(tender_id: int, model_version: str = "v1") -> str:
    """Генерирует ключ кеша для ML-анализа"""
    return f"ml_analysis_{tender_id}_{model_version}"


def get_ml_analysis_lock_key(tender_id: int) -> str:
    """Ключ блокировки для предотвращения дублирования анализа одного тендера"""
    return f"ml_analysis_lock_{tender_id}"


# Глобальный счётчик одновременных ML-анализов
MAX_CONCURRENT_ML = 2  # Максимум 2 анализа одновременно
ML_CONCURRENT_KEY = "ml_concurrent_count"


@api_view(["GET"])
def start_parser(request):
    """
    Запуск парсера с защитой от одновременных запусков.
    """
    # Проверяем, не запущен ли уже парсер
    if cache.get('parser_running'):
        return Response(
            {"error": "Парсер уже запущен. Пожалуйста, дождитесь завершения."}, 
            status=409
        )
    
    # Устанавливаем блокировку на 10 минут (максимальное время выполнения)
    cache.set('parser_running', True, 600)
    
    try:
        print("Started Parcing")


        python_path = os.path.join(settings.EIS_PARSER_ROOT, '.venv', 'Scripts', 'python.exe')
        script_path = os.path.join('parser', 'parser.py')

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [python_path, script_path],
            capture_output=True, 
            encoding='utf-8',
            cwd=settings.EIS_PARSER_ROOT,
            errors='replace',
            env=env,
            timeout=600  # 10 минут таймаут
        )
        
        # Логируем вывод для отладки
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr length: {len(result.stderr)}")
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Stdout preview: {result.stdout[:500]}")
                return Response({"error": f"Ошибка парсинга JSON: {e}", "stdout_preview": result.stdout[:500]}, status=500)
            
            try:
                stats = update_tenders_from_data(data)
                
                # Очищаем весь ML-кеш при обновлении данных
                cache.clear()
                
                return Response({
                    "status": "ok",
                    "created": stats['created'],
                    "updated": stats['updated'],
                    "errors_count": len(stats['errors']),
                    "ml_cache_cleared": True
                }, status=200)
            except Exception as e:
                print(str(e))
                return Response({"error": str(e)}, status=500)
        else:
            print(f"Parser error: {result.stderr}")
            return Response({'error': result.stderr}, status=500)
    
    except subprocess.TimeoutExpired:
        return Response({"error": "Превышено время выполнения парсера (10 мин)"}, status=500)
    finally:
        # Снимаем блокировку
        cache.delete('parser_running')


@api_view(["POST"])
def clear_ml_cache(request, tender_id=None):
    """
    Очистка кеша ML-анализа.
    
    - POST /api/ml-cache/clear/ — очистить весь кеш
    - POST /api/ml-cache/clear/{tender_id}/ — очистить кеш конкретного тендера
    """
    if tender_id:
        cache_key = get_ml_cache_key(tender_id)
        cache.delete(cache_key)
        return Response({"status": "ok", "message": f"Кеш для тендера {tender_id} очищен"})
    else:
        # Очищаем весь ML-кеш
        deleted_count = 0
        # Django cache не поддерживает массовое удаление по паттерну,
        # поэтому просто сбрасываем счётчик
        cache.delete(ML_CONCURRENT_KEY)
        return Response({"status": "ok", "message": "ML-кеш очищен"})