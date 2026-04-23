from datetime import datetime
from decimal import Decimal
from django.db import transaction
from .models import Tenders


def parse_datetime(date_str: str, time_str: str = None) -> datetime | None:
    """
    Парсинг даты и времени из строки формата 'DD.MM.YYYY HH:MM' или 'DD.MM.YYYY'.
    """
    if not date_str:
        return None
    
    try:
        if time_str and time_str.strip():
            dt_str = f"{date_str} {time_str}"
            return datetime.strptime(dt_str, "%d.%m.%Y %H:%M")
        elif ' ' in date_str:
            return datetime.strptime(date_str, "%d.%m.%Y %H:%M")
        else:
            dt = datetime.strptime(date_str, "%d.%m.%Y")
            return datetime(dt.year, dt.month, dt.day)
    except ValueError:
        return None


def parse_date(date_str: str) -> datetime | None:
    """
    Парсинг даты из строки формата 'DD.MM.YYYY'.
    """
    if not date_str:
        return None
    
    try:
        if ' ' in date_str:
            date_str = date_str.split()[0]
        return datetime.strptime(date_str, "%d.%m.%Y")
    except ValueError:
        return None


def parse_price(price_str: str) -> Decimal | None:
    """
    Парсинг цены из строки (удаляет пробелы и заменяет запятую на точку).
    """
    if not price_str:
        return None
    
    try:
        cleaned = price_str.replace(' ', '').replace(',', '.')
        return Decimal(cleaned)
    except Exception:
        return None


def clean_phone(phone_str: str) -> str:
    """
    Очистка номера телефона от лишних символов.
    """
    if not phone_str:
        return ''
    
    # Убираем лишние символы, оставляем цифры, +, -, скобки
    cleaned = ''.join(c for c in phone_str if c.isdigit() or c in '+-()')
    return cleaned[:50]


def map_tender_data(raw_data: dict) -> dict:
    """
    Маппинг сырых данных в формат модели Tenders.
    """
    # Парсинг дат
    start_dt = parse_datetime(raw_data.get('Дата и время начала срока подачи заявок', ''))
    end_dt = parse_datetime(raw_data.get('Дата и время окончания срока подачи заявок', ''))
    results_dt = parse_date(raw_data.get('Дата подведения итогов определения поставщика (подрядчика, исполнителя)', ''))
    
    # Парсинг цены
    price_field = raw_data.get('Начальная (максимальная) цена контракта') or raw_data.get('Максимальное значение цены контракта')
    initial_price = parse_price(price_field) if price_field else None
    
    # Обеспечение заявки
    bid_security_required = raw_data.get('Требуется обеспечение заявки', 'Нет') == 'Да'
    bid_security_amount = raw_data.get('Размер обеспечения заявки', '')
    if bid_security_amount and 'РОССИЙСКИЙ РУБЛЬ' in bid_security_amount:
        bid_security_amount = bid_security_amount.replace('РОССИЙСКИЙ РУБЛЬ', '').strip()
    
    # Обеспечение исполнения контракта
    performance_required = raw_data.get('Требуется обеспечение исполнения контракта', 'Нет') == 'Да'
    performance_security_size = raw_data.get('Размер обеспечения исполнения контракта', '')
    
    # Контакты
    email = raw_data.get('Адрес электронной почты', '')
    phone = clean_phone(raw_data.get('Номер контактного телефона', ''))
    
    # Площадка
    platform_name = raw_data.get('Наименование электронной площадки в информационно-телекоммуникационной сети «Интернет»', '')
    platform_url = raw_data.get('Адрес электронной площадки в информационно-телекоммуникационной сети «Интернет»', '')
    
    return {
        'icz': raw_data.get('Идентификационный код закупки (ИКЗ)', ''),
        'object_name': raw_data.get('Наименование объекта закупки', ''),
        'customer_full_name': raw_data.get('Организация, осуществляющая размещение', ''),
        'region': raw_data.get('Регион', ''),
        'procurement_method': raw_data.get('Способ определения поставщика (подрядчика, исполнителя)', ''),
        'stage': raw_data.get('Этап закупки', ''),
        'start_date_time': start_dt,
        'end_date_time': end_dt,
        'results_date': results_dt.date() if results_dt else None,
        'initial_price': initial_price,
        'contract_execution_period': raw_data.get('Срок исполнения контракта', ''),
        'participant_requirements': raw_data.get('Требования к участникам', ''),
        'advantages': raw_data.get('Преимущества', ''),
        'delivery_place': raw_data.get('Место поставки товара, выполнения работы или оказания услуги', '') or raw_data.get('Место поставки товара', ''),
        'bid_security_required': bid_security_required,
        'bid_security_amount': bid_security_amount if bid_security_required else None,
        'performance_required': performance_required,
        'performance_security_size': performance_security_size if performance_required else None,
        'email': email,
        'phone': phone,
        'electronic_platform_name': platform_name,
        'electronic_platform_url': platform_url,
    }


@transaction.atomic
def update_tenders_from_data(tenders_data: list[dict]) -> dict:
    """
    Обновление или создание записей о тендерах в БД.
    
    Args:
        tenders_data: Список словарей с данными о тендерах.
    
    Returns:
        Словарь со статистикой: {'created': int, 'updated': int, 'errors': list}
    """
    stats = {'created': 0, 'updated': 0, 'errors': []}
    
    for idx, raw_tender in enumerate(tenders_data):
        try:
            mapped_data = map_tender_data(raw_tender)
            
            # Проверка обязательных полей
            if not mapped_data['icz']:
                stats['errors'].append(f"Запись {idx + 1}: отсутствует ИКЗ")
                continue
            
            if not mapped_data['object_name']:
                stats['errors'].append(f"Запись {idx + 1} (ИКЗ: {mapped_data['icz']}): отсутствует наименование объекта закупки")
                continue
            
            # Обновление или создание
            tender, created = Tenders.objects.update_or_create(
                icz=mapped_data['icz'],
                defaults=mapped_data
            )
            
            if created:
                stats['created'] += 1
            else:
                stats['updated'] += 1
                
        except Exception as e:
            stats['errors'].append(f"Запись {idx + 1}: {str(e)}")
    
    return stats