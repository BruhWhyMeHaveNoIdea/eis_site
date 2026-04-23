"""
Configuration package for retrospective analysis algorithm.

This package provides a unified configuration system with backward compatibility
for the old config.py and ml_config.py modules.
"""

import warnings
from typing import Dict, Any, List

from .settings import (
    get_config as _get_config,
    reload_config as _reload_config,
    get_field_mapping as _get_field_mapping,
    get_weight_preset as _get_weight_preset,
    DEFAULT_CONFIG,
    PRESETS,
    ConfigLoader
)

# Re-export core functions
get_config = _get_config
reload_config = _reload_config
get_field_mapping = _get_field_mapping
get_weight_preset = _get_weight_preset

# ============================================================================
# Backward Compatibility Constants (extracted from DEFAULT_CONFIG)
# ============================================================================

def _get_config_value(key: str, default: Any = None) -> Any:
    """Get value from DEFAULT_CONFIG with dot notation."""
    keys = key.split('.')
    value = DEFAULT_CONFIG
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value

# Similarity weights
DEFAULT_WEIGHTS = _get_config_value('weights', {
    'text': 0.3,
    'description': 0.2,
    'customer': 0.15,
    'region': 0.1,
    'method': 0.05,
    'price': 0.1,
    'date': 0.05,
    'other': 0.05
})

WEIGHT_PRESETS = PRESETS if PRESETS else {
    'balanced': DEFAULT_WEIGHTS,
    'text_heavy': {
        'text': 0.4,
        'description': 0.3,
        'customer': 0.15,
        'region': 0.05,
        'method': 0.02,
        'price': 0.05,
        'date': 0.02,
        'other': 0.01
    },
    'region_heavy': {
        'text': 0.2,
        'description': 0.15,
        'customer': 0.1,
        'region': 0.3,
        'method': 0.05,
        'price': 0.1,
        'date': 0.05,
        'other': 0.05
    },
    'price_sensitive': {
        'text': 0.2,
        'description': 0.15,
        'customer': 0.1,
        'region': 0.05,
        'method': 0.05,
        'price': 0.3,
        'date': 0.1,
        'other': 0.05
    },
    'temporal': {
        'text': 0.2,
        'description': 0.15,
        'customer': 0.1,
        'region': 0.05,
        'method': 0.05,
        'price': 0.1,
        'date': 0.3,
        'other': 0.05
    }
}

# Field mappings
TEXT_FIELDS = _get_config_value('field_mappings.text_fields', [
    'Наименование объекта закупки',
    'Наименование закупки',
    'Заказчик',
    'Организатор',
    'Требования к участникам',
    'Критерии оценки заявок',
    'Гарантийные обязательства',
    'Условия оплаты'
])

NUMERIC_FIELDS = _get_config_value('field_mappings.numeric_fields', [
    'Начальная (максимальная) цена контракта',
    'Цена контракта',
    'Размер обеспечения заявки',
    'Размер обеспечения исполнения контракта'
])

DATE_FIELDS = _get_config_value('field_mappings.date_fields', [
    'Дата публикации',
    'Дата окончания подачи заявок',
    'Дата подведения итогов'
])

CATEGORICAL_FIELDS = _get_config_value('field_mappings.categorical_fields', [
    'Способ определения поставщика',
    'Регион',
    'Валюта',
    'Источник финансирования',
    'Статус закупки',
    'Электронная площадка'
])

# Embedding configuration
EMBEDDING_MODEL = _get_config_value('embedding.model', "paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIMENSION = _get_config_value('embedding.dimension', 384)
EMBEDDING_DEVICE = _get_config_value('embedding.device', "auto")
EMBEDDING_CACHE_DIR = _get_config_value('embedding.cache_dir', None)

TEXT_FIELD_WEIGHTS = _get_config_value('embedding.text_field_weights', {
    'Наименование объекта закупки_cleaned': 0.4,
    'Наименование закупки_cleaned': 0.3,
    'Заказчик_cleaned': 0.2,
    'Требования к участникам_cleaned': 0.05,
    'Критерии оценки заявок_cleaned': 0.05
})

# Similarity calculation parameters
NUMERICAL_SIMILARITY_METHOD = _get_config_value('similarity.numerical.method', 'gaussian')
NUMERICAL_SCALE_AUTO = _get_config_value('similarity.numerical.scale_auto', True)
NUMERICAL_SCALE_FACTOR = _get_config_value('similarity.numerical.scale_factor', 0.5)

TEMPORAL_SIMILARITY_METHOD = _get_config_value('similarity.temporal.method', 'exponential')
TEMPORAL_SCALE_DAYS = _get_config_value('similarity.temporal.scale_days', 90)

# Region hierarchy
REGION_HIERARCHY = _get_config_value('similarity.categorical.hierarchical_mappings.region', {
    'Центральный федеральный округ': [
        'Москва', 'Московская область', 'Белгородская область', 'Брянская область',
        'Владимирская область', 'Воронежская область', 'Ивановская область',
        'Калужская область', 'Костромская область', 'Курская область',
        'Липецкая область', 'Орловская область', 'Рязанская область',
        'Смоленская область', 'Тамбовская область', 'Тверская область',
        'Тульская область', 'Ярославская область'
    ],
    'Северо-Западный федеральный округ': [
        'Санкт-Петербург', 'Ленинградская область', 'Архангельская область',
        'Вологодская область', 'Калининградская область', 'Республика Карелия',
        'Республика Коми', 'Мурманская область', 'Новгородская область',
        'Псковская область', 'Ненецкий автономный округ'
    ],
    'Южный федеральный округ': [
        'Республика Адыгея', 'Республика Калмыкия', 'Краснодарский край',
        'Астраханская область', 'Волгоградская область', 'Ростовская область',
        'Республика Крым', 'Севастополь'
    ],
    'Северо-Кавказский федеральный округ': [
        'Республика Дагестан', 'Республика Ингушетия', 'Кабардино-Балкарская Республика',
        'Карачаево-Черкесская Республика', 'Республика Северная Осетия-Алания',
        'Чеченская Республика', 'Ставропольский край'
    ],
    'Приволжский федеральный округ': [
        'Республика Башкортостан', 'Республика Марий Эл', 'Республика Мордовия',
        'Республика Татарстан', 'Удмуртская Республика', 'Чувашская Республика',
        'Кировская область', 'Нижегородская область', 'Оренбургская область',
        'Пензенская область', 'Пермский край', 'Самарская область',
        'Саратовская область', 'Ульяновская область'
    ],
    'Уральский федеральный округ': [
        'Курганская область', 'Свердловская область', 'Тюменская область',
        'Челябинская область', 'Ханты-Мансийский автономный округ',
        'Ямало-Ненецкий автономный округ'
    ],
    'Сибирский федеральный округ': [
        'Республика Алтай', 'Республика Тыва', 'Республика Хакасия',
        'Алтайский край', 'Красноярский край', 'Иркутская область',
        'Кемеровская область', 'Новосибирская область', 'Омская область',
        'Томская область'
    ],
    'Дальневосточный федеральный округ': [
        'Республика Бурятия', 'Республика Саха (Якутия)', 'Забайкальский край',
        'Камчатский край', 'Приморский край', 'Хабаровский край',
        'Амурская область', 'Магаданская область', 'Сахалинская область',
        'Еврейская автономная область', 'Чукотский автономный округ'
    ]
})

# Procurement method groups
PROCUREMENT_METHOD_GROUPS = _get_config_value('similarity.categorical.hierarchical_mappings.procurement_method', {
    'electronic_auction': [
        'Электронный аукцион',
        'Аукцион в электронной форме',
        'Электронные торги',
        'Аукцион'
    ],
    'competition': [
        'Конкурс',
        'Открытый конкурс',
        'Конкурс с ограниченным участием',
        'Двухэтапный конкурс',
        'Закрытый конкурс'
    ],
    'quotation': [
        'Запрос котировок',
        'Запрос предложений',
        'Запрос цен'
    ],
    'single_source': [
        'Закупка у единственного поставщика',
        'Единственный поставщик',
        'Без проведения торгов'
    ]
})

# Algorithm parameters
DEFAULT_K = _get_config_value('algorithm.default_k', 10)
MIN_SIMILARITY_THRESHOLD = _get_config_value('algorithm.min_similarity_threshold', 0.0)
EXCLUDE_SELF = _get_config_value('algorithm.exclude_self', True)
BATCH_SIZE = _get_config_value('algorithm.batch_size', 32)
N_JOBS = _get_config_value('algorithm.n_jobs', 1)

# Output configuration
OUTPUT_FORMAT = _get_config_value('output', {
    'include_full_tenders': True,
    'include_breakdown': True,
    'include_explanation': True,
    'include_metadata': True,
    'pretty_print': True,
    'indent': 2
})

ESSENTIAL_FIELDS = _get_config_value('output.essential_fields', [
    'Идентификационный код закупки (ИКЗ)',
    'Номер извещения',
    'Наименование объекта закупки',
    'Наименование закупки',
    'Начальная (максимальная) цена контракта',
    'Цена контракта',
    'Регион',
    'Заказчик',
    'Дата публикации',
    'Способ определения поставщика',
    'Статус закупки'
])

# Performance configuration
CACHE_EMBEDDINGS = _get_config_value('performance.cache_embeddings', True)
PREPROCESS_ON_LOAD = _get_config_value('performance.preprocess_on_load', True)
COMPUTE_STATISTICS = _get_config_value('performance.compute_statistics', True)
USE_FAST_SIMILARITY = _get_config_value('performance.use_fast_similarity', False)

# Logging configuration
LOGGING_CONFIG = _get_config_value('logging', {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
})

# ============================================================================
# ML Configuration (from ml_config.py)
# ============================================================================

# Note: ML-specific configuration is now part of config.settings.
# For backward compatibility, we import the MLConfig class from ml_config
# if it exists, otherwise we define a minimal stub.

try:
    from .ml_compat import MLConfig, get_ml_config, validate_ml_config, get_default_text_fields
except ImportError:
    # Define minimal stubs to avoid import errors
    class MLConfig:
        """Stub for MLConfig."""
        pass
    
    def get_ml_config():
        return MLConfig()
    
    def validate_ml_config(config):
        return True
    
    def get_default_text_fields():
        return TEXT_FIELDS

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'get_config',
    'reload_config',
    'get_field_mapping',
    'get_weight_preset',
    'DEFAULT_CONFIG',
    'PRESETS',
    'ConfigLoader',
    'DEFAULT_WEIGHTS',
    'WEIGHT_PRESETS',
    'TEXT_FIELDS',
    'NUMERIC_FIELDS',
    'DATE_FIELDS',
    'CATEGORICAL_FIELDS',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIMENSION',
    'EMBEDDING_DEVICE',
    'EMBEDDING_CACHE_DIR',
    'TEXT_FIELD_WEIGHTS',
    'NUMERICAL_SIMILARITY_METHOD',
    'NUMERICAL_SCALE_AUTO',
    'NUMERICAL_SCALE_FACTOR',
    'TEMPORAL_SIMILARITY_METHOD',
    'TEMPORAL_SCALE_DAYS',
    'REGION_HIERARCHY',
    'PROCUREMENT_METHOD_GROUPS',
    'DEFAULT_K',
    'MIN_SIMILARITY_THRESHOLD',
    'EXCLUDE_SELF',
    'BATCH_SIZE',
    'N_JOBS',
    'OUTPUT_FORMAT',
    'ESSENTIAL_FIELDS',
    'CACHE_EMBEDDINGS',
    'PREPROCESS_ON_LOAD',
    'COMPUTE_STATISTICS',
    'USE_FAST_SIMILARITY',
    'LOGGING_CONFIG',
    'MLConfig',
    'get_ml_config',
    'validate_ml_config',
    'get_default_text_fields',
]