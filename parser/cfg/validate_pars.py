from datetime import datetime
from typing import Optional

def validate_parser_params(
    start_page: int,
    end_page: int,
    per_page: int,
    pub_date: str,
    close_date: Optional[str] = None
) -> None:
    """
    Валидирует параметры парсера и выбрасывает исключение при ошибках.
    
    Args:
        start_page: Стартовый номер страницы (int > 0)
        end_page: Конечный номер страницы (int > start_page)
        per_page: Количество объектов на странице (1 <= int <= 1000)
        pub_date: Дата открытия заявки (формат дд.мм.гггг)
        close_date: Дата закрытия заявки (формат дд.мм.гггг, опционально)
    
    Raises:
        ValueError: При некорректных параметрах
    """
    
    # Проверка типов
    if not isinstance(start_page, int) or not isinstance(end_page, int) or not isinstance(per_page, int):
        raise ValueError("start_page, end_page и per_page должны быть целыми числами")
    
    if not isinstance(pub_date, str):
        raise ValueError("pub_date должен быть строкой")
    
    if close_date is not None and not isinstance(close_date, str):
        raise ValueError("close_date должен быть строкой или None")
    
    # Проверка диапазонов
    if start_page < 1:
        raise ValueError("start_page должен быть >= 1")
    
    if end_page < start_page:
        raise ValueError("end_page должен быть >= start_page")
    
    if per_page < 1 or per_page > 1000:
        raise ValueError("per_page должен быть от 1 до 1000")
    
    # Валидация формата дат
    try:
        datetime.strptime(pub_date, "%d.%m.%Y")
    except ValueError:
        raise ValueError("pub_date должен быть в формате дд.мм.гггг")
    
    if close_date:
        try:
            datetime.strptime(close_date, "%d.%m.%Y")
        except ValueError:
            raise ValueError("close_date должен быть в формате дд.мм.гггг")
        
        # Проверка логики дат
        pub_dt = datetime.strptime(pub_date, "%d.%m.%Y")
        close_dt = datetime.strptime(close_date, "%d.%m.%Y")
        
        if close_dt < pub_dt:
            raise ValueError("close_date не может быть раньше pub_date")