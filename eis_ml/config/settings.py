"""
Configuration system for retrospective analysis algorithm.

This module provides a unified configuration system that loads settings from:
1. Environment variables (with prefixes)
2. YAML configuration files
3. JSON configuration files
4. Python dictionaries with defaults

All hardcoded values from the original config.py and other modules are moved here
and made configurable through this system.

Principles:
- No hardcoded field names, dictionaries with keywords, or filters by individual words
- All field names are configurable
- All weights and parameters are configurable
- Configuration can be overridden by environment variables
- Support for multiple configuration profiles
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Similarity weights
    "weights": {
        "text": 0.3,
        "description": 0.2,
        "customer": 0.15,
        "region": 0.1,
        "method": 0.05,
        "price": 0.1,
        "date": 0.05,
        "other": 0.05
    },
    
    # Field mappings (configurable field names)
    "field_mappings": {
        "text_fields": [
            "Наименование объекта закупки",
            "Наименование закупки",
            "Заказчик",
            "Организатор",
            "Требования к участникам",
            "Критерии оценки заявок",
            "Гарантийные обязательства",
            "Условия оплаты"
        ],
        "numeric_fields": [
            "Начальная (максимальная) цена контракта",
            "Цена контракта",
            "Размер обеспечения заявки",
            "Размер обеспечения исполнения контракта"
        ],
        "date_fields": [
            "Дата публикации",
            "Дата окончания подачи заявок",
            "Дата подведения итогов"
        ],
        "categorical_fields": [
            "Способ определения поставщика",
            "Регион",
            "Валюта",
            "Источник финансирования",
            "Статус закупки",
            "Электронная площадка"
        ],
        "id_field": "Идентификационный код закупки (ИКЗ)",
        "title_field": "Наименование объекта закупки",
        "price_field": "Начальная (максимальная) цена контракта",
        "customer_field": "Заказчик",
        "region_field": "Регион",
        "method_field": "Способ определения поставщика"
    },
    
    # Embedding configuration
    "embedding": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "device": "auto",
        "cache_dir": None,
        "text_field_weights": {
            "Наименование объекта закупки_cleaned": 0.4,
            "Наименование закупки_cleaned": 0.3,
            "Заказчик_cleaned": 0.2,
            "Требования к участникам_cleaned": 0.05,
            "Критерии оценки заявок_cleaned": 0.05
        }
    },
    
    # Similarity calculation parameters
    "similarity": {
        "numerical": {
            "method": "gaussian",  # 'gaussian', 'exponential', 'inverse'
            "scale_auto": True,
            "scale_factor": 0.5
        },
        "temporal": {
            "method": "exponential",  # 'exponential', 'gaussian', 'linear'
            "scale_days": 90
        },
        "categorical": {
            "hierarchical_mappings": {
                "region": {
                    "Центральный федеральный округ": [
                        "Москва", "Московская область", "Белгородская область", "Брянская область",
                        "Владимирская область", "Воронежская область", "Ивановская область",
                        "Калужская область", "Костромская область", "Курская область",
                        "Липецкая область", "Орловская область", "Рязанская область",
                        "Смоленская область", "Тамбовская область", "Тверская область",
                        "Тульская область", "Ярославская область"
                    ],
                    "Северо-Западный федеральный округ": [
                        "Санкт-Петербург", "Ленинградская область", "Архангельская область",
                        "Вологодская область", "Калининградская область", "Республика Карелия",
                        "Республика Коми", "Мурманская область", "Новгородская область",
                        "Псковская область", "Ненецкий автономный округ"
                    ],
                    "Южный федеральный округ": [
                        "Республика Адыгея", "Республика Калмыкия", "Краснодарский край",
                        "Астраханская область", "Волгоградская область", "Ростовская область",
                        "Республика Крым", "Севастополь"
                    ],
                    "Северо-Кавказский федеральный округ": [
                        "Республика Дагестан", "Республика Ингушетия", "Кабардино-Балкарская Республика",
                        "Карачаево-Черкесская Республика", "Республика Северная Осетия-Алания",
                        "Чеченская Республика", "Ставропольский край"
                    ],
                    "Приволжский федеральный округ": [
                        "Республика Башкортостан", "Республика Марий Эл", "Республика Мордовия",
                        "Республика Татарстан", "Удмуртская Республика", "Чувашская Республика",
                        "Кировская область", "Нижегородская область", "Оренбургская область",
                        "Пензенская область", "Пермский край", "Самарская область",
                        "Саратовская область", "Ульяновская область"
                    ],
                    "Уральский федеральный округ": [
                        "Курганская область", "Свердловская область", "Тюменская область",
                        "Челябинская область", "Ханты-Мансийский автономный округ",
                        "Ямало-Ненецкий автономный округ"
                    ],
                    "Сибирский федеральный округ": [
                        "Республика Алтай", "Республика Тыва", "Республика Хакасия",
                        "Алтайский край", "Красноярский край", "Иркутская область",
                        "Кемеровская область", "Новосибирская область", "Омская область",
                        "Томская область"
                    ],
                    "Дальневосточный федеральный округ": [
                        "Республика Бурятия", "Республика Саха (Якутия)", "Забайкальский край",
                        "Камчатский край", "Приморский край", "Хабаровский край",
                        "Амурская область", "Магаданская область", "Сахалинская область",
                        "Еврейская автономная область", "Чукотский автономный округ"
                    ]
                },
                "procurement_method": {
                    "electronic_auction": [
                        "Электронный аукцион",
                        "Аукцион в электронной форме",
                        "Электронные торги",
                        "Аукцион"
                    ],
                    "competition": [
                        "Конкурс",
                        "Открытый конкурс",
                        "Конкурс с ограниченным участием",
                        "Двухэтапный конкурс",
                        "Закрытый конкурс"
                    ],
                    "quotation": [
                        "Запрос котировок",
                        "Запрос предложений",
                        "Запрос цен"
                    ],
                    "single_source": [
                        "Закупка у единственного поставщика",
                        "Единственный поставщик",
                        "Без проведения торгов"
                    ]
                }
            }
        }
    },
    
    # Algorithm parameters
    "algorithm": {
        "default_k": 10,
        "min_similarity_threshold": 0.0,
        "exclude_self": True,
        "batch_size": 32,
        "n_jobs": 1
    },
    
    # Output configuration
    "output": {
        "include_full_tenders": True,
        "include_breakdown": True,
        "include_explanation": True,
        "include_metadata": True,
        "pretty_print": True,
        "indent": 2,
        "essential_fields": [
            "Идентификационный код закупки (ИКЗ)",
            "Номер извещения",
            "Наименование объекта закупки",
            "Наименование закупки",
            "Начальная (максимальная) цена контракта",
            "Цена контракта",
            "Регион",
            "Заказчик",
            "Дата публикации",
            "Способ определения поставщика",
            "Статус закупки"
        ]
    },
    
    # Performance configuration
    "performance": {
        "cache_embeddings": True,
        "preprocess_on_load": True,
        "compute_statistics": True,
        "use_fast_similarity": False
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    },
    
    # ML configuration (simplified, detailed config in ml_config.py)
    "ml": {
        "enable": False,
        "config_path": "./ml_config.json"
    }
}

# ============================================================================
# Configuration Presets
# ============================================================================

PRESETS = {
    "balanced": DEFAULT_CONFIG,
    
    "text_heavy": {
        "weights": {
            "text": 0.4,
            "description": 0.3,
            "customer": 0.15,
            "region": 0.05,
            "method": 0.02,
            "price": 0.05,
            "date": 0.02,
            "other": 0.01
        }
    },
    
    "region_heavy": {
        "weights": {
            "text": 0.2,
            "description": 0.15,
            "customer": 0.1,
            "region": 0.3,
            "method": 0.05,
            "price": 0.1,
            "date": 0.05,
            "other": 0.05
        }
    },
    
    "price_sensitive": {
        "weights": {
            "text": 0.2,
            "description": 0.15,
            "customer": 0.1,
            "region": 0.05,
            "method": 0.05,
            "price": 0.3,
            "date": 0.1,
            "other": 0.05
        }
    },
    
    "temporal": {
        "weights": {
            "text": 0.2,
            "description": 0.15,
            "customer": 0.1,
            "region": 0.05,
            "method": 0.05,
            "price": 0.1,
            "date": 0.3,
            "other": 0.05
        }
    }
}

# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Load and manage configuration from multiple sources."""
    
    def __init__(self, config_path: Optional[str] = None, preset: str = "balanced"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            preset: Configuration preset name
        """
        self.config = DEFAULT_CONFIG.copy()
        self.preset = preset
        self.config_path = config_path
        
        # Load configuration in order of precedence
        self._load_preset(preset)
        if config_path:
            self._load_file(config_path)
        self._load_environment()
        
        # Validate configuration
        self._validate()
    
    def _load_preset(self, preset: str):
        """Load configuration preset."""
        if preset in PRESETS:
            preset_config = PRESETS[preset]
            self._deep_update(self.config, preset_config)
            logger.info(f"Loaded configuration preset: {preset}")
        else:
            logger.warning(f"Unknown preset: {preset}, using 'balanced'")
    
    def _load_file(self, config_path: str):
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return
            
            self._deep_update(self.config, file_config)
            logger.info(f"Loaded configuration from file: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {e}")
    
    def _load_environment(self):
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to configuration paths
        env_mappings = {
            # Weights
            "RETRO_WEIGHT_TEXT": ("weights", "text"),
            "RETRO_WEIGHT_DESCRIPTION": ("weights", "description"),
            "RETRO_WEIGHT_CUSTOMER": ("weights", "customer"),
            "RETRO_WEIGHT_REGION": ("weights", "region"),
            "RETRO_WEIGHT_METHOD": ("weights", "method"),
            "RETRO_WEIGHT_PRICE": ("weights", "price"),
            "RETRO_WEIGHT_DATE": ("weights", "date"),
            "RETRO_WEIGHT_OTHER": ("weights", "other"),
            
            # Embedding
            "RETRO_EMBEDDING_MODEL": ("embedding", "model"),
            "RETRO_EMBEDDING_DEVICE": ("embedding", "device"),
            
            # Algorithm
            "RETRO_DEFAULT_K": ("algorithm", "default_k"),
            "RETRO_MIN_SIMILARITY": ("algorithm", "min_similarity_threshold"),
            "RETRO_BATCH_SIZE": ("algorithm", "batch_size"),
            
            # Performance
            "RETRO_CACHE_EMBEDDINGS": ("performance", "cache_embeddings"),
            "RETRO_PREPROCESS_ON_LOAD": ("performance", "preprocess_on_load"),
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                
                # Set value in configuration
                self._set_nested_value(env_config, config_path, value)
        
        if env_config:
            self._deep_update(self.config, env_config)
            logger.info("Loaded configuration from environment variables")
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Recursively update target dictionary with source dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _set_nested_value(self, config: Dict[str, Any], path: tuple, value: Any):
        """Set a value in a nested dictionary using a tuple path."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate(self):
        """Validate configuration."""
        errors = []
        
        # Check weights sum
        weights = self.config.get("weights", {})
        if weights:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                errors.append(f"Weights sum to {total}, should be approximately 1.0")
        
        # Check required fields
        required_sections = ["field_mappings", "embedding", "algorithm"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

# ============================================================================
# Global Configuration Instance
# ============================================================================

# Default configuration instance
_config_instance: Optional[ConfigLoader] = None

def get_config(config_path: Optional[str] = None, preset: str = "balanced") -> ConfigLoader:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to configuration file
        preset: Configuration preset name
    
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path, preset)
    
    return _config_instance

def reload_config(config_path: Optional[str] = None, preset: str = "balanced"):
    """
    Reload configuration with new settings.
    
    Args:
        config_path: Path to configuration file
        preset: Configuration preset name
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path, preset)

# ============================================================================
# Helper Functions
# ============================================================================

def get_field_mapping(field_type: str) -> List[str]:
    """
    Get field names for a specific field type.
    
    Args:
        field_type: One of 'text', 'numeric', 'date', 'categorical'
    
    Returns:
        List of field names
    """
    config = get_config()
    field_mappings = config.get("field_mappings", {})
    
    if field_type == "text":
        return field_mappings.get("text_fields", [])
    elif field_type == "numeric":
        return field_mappings.get("numeric_fields", [])
    elif field_type == "date":
        return field_mappings.get("date_fields", [])
    elif field_type == "categorical":
        return field_mappings.get("categorical_fields", [])
    else:
        raise ValueError(f"Unknown field type: {field_type}")

def get_weight_preset(preset: str) -> Dict[str, float]:
    """
    Get weight configuration for a preset.
    
    Args:
        preset: Preset name
    
    Returns:
        Dictionary of weights
    """
    if preset in PRESETS:
        preset_config = PRESETS[preset]
        return preset_config.get("weights", DEFAULT_CONFIG["weights"])
    else:
        return DEFAULT_CONFIG["weights"]

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Load configuration
    config = get_config()
    print("Configuration loaded successfully")
    print(f"Weights: {config.get('weights')}")
    print(f"Text fields: {get_field_mapping('text')}")
    
    # Example: Create configuration file
    sample_config = {
        "weights": {
            "text": 0.35,
            "description": 0.25,
            "customer": 0.15,
            "region": 0.1,
            "method": 0.05,
            "price": 0.05,
            "date": 0.03,
            "other": 0.02
        },
        "algorithm": {
            "default_k": 5,
            "batch_size": 64
        }
    }
    
    # Save sample configuration
    config_dir = Path(__file__).parent
    sample_path = config_dir / "sample_config.yaml"
    with open(sample_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Sample configuration saved to: {sample_path}")