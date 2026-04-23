from datetime import datetime


def validate_params(start_page: int, end_page: int, per_page: int, pub_date: str, close_date: str = "") -> dict:
    """Валидирует параметры командной строки"""

    # Проверка pagination
    if not isinstance(start_page, int) or start_page <= 0:
        raise ValueError("Некорректный аргумент для start_page")
    if not isinstance(end_page, int) or end_page <= 0:
        raise ValueError("Некорректный аргумент для end_page")
    if not isinstance(per_page, int) or per_page <= 0:
        raise ValueError("Некорректный аргумент для per_page")

    if start_page > end_page:
        raise ValueError("Обратите внимание: start_page > end_page")

    # Проверка dates
    if not isinstance(pub_date, str):
        raise ValueError("Некорректный аргумент для pub_date")
    try:
        datetime.strptime(pub_date, "%d.%m.%Y")
    except ValueError:
        raise ValueError("Нужен формат: DD.MM.YYYY")

    if close_date:
        if not isinstance(close_date, str):
            raise ValueError("Некорректный аргумент для close_date")
        try:
            datetime.strptime(close_date, "%d.%m.%Y")
        except ValueError:
            raise ValueError("Нужен формат: DD.MM.YYYY")

    return {
        "pagination": {
            "per_page": per_page,
            "start_page": start_page,
            "end_page": end_page
        },
        "filters": {
            "pub_date": pub_date,
            "close_date": close_date
        }
    }

