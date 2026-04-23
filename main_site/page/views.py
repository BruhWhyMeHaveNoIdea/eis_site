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
    """
    try:
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
                    '--k', '5'
                ],
                capture_output=True,
                encoding='utf-8',
                cwd=settings.EIS_ML_ROOT,
                errors='replace',
                timeout=120  # 2 минуты на анализ
            )
            print("RES", result)
            if result.returncode == 0:
                ml_result = json.loads(result.stdout)
                return Response(ml_result)
            else:
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




@api_view(["GET"])
def start_parser(request):
    print("Started Parcing")

    python_path = os.path.join(settings.EIS_PARSER_ROOT, '.venv', 'Scripts', 'python.exe')
    script_path = os.path.join('parser', 'parser.py')

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(
        [python_path, script_path],
        capture_output=True, 
        encoding='utf-8',
        cwd=settings.EIS_PARSER_ROOT, #TODO: Fix cwd
        errors='replace',
        env=env
    )
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        
        try:
            update_tenders_from_data(data)
            return Response(status=200)
        except Exception as e:
            print(str(e))
            return Response({"error": str(e)}, status=500)
    return Response({'error': result.stderr}, status=500)