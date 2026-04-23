"""
Synthetic test data for comprehensive testing.

This module provides programmatically generated test data covering edge cases,
realistic Russian tender data, and stress test scenarios.
"""

import numpy as np
from datetime import datetime, timedelta
import random
import string


def generate_tender(tender_id, **overrides):
    """
    Generate a synthetic tender with realistic Russian procurement data.
    
    Args:
        tender_id: Unique identifier for the tender
        **overrides: Custom values to override defaults
        
    Returns:
        Dictionary representing a tender
    """
    # Base tender template
    tender = {
        "Идентификационный код закупки (ИКЗ)": f"TEST{tender_id:06d}",
        "Наименование объекта закупки": f"Поставка оборудования типа {tender_id}",
        "Наименование закупки": f"Закупка оборудования {tender_id}",
        "Начальная (максимальная) цена контракта": f"{random.randint(100000, 10000000):,}".replace(',', ' ') + ",00",
        "Дата публикации": (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%d.%m.%Y %H:%M"),
        "Дата окончания подачи заявок": (datetime(2024, 1, 1) + timedelta(days=random.randint(10, 375))).strftime("%d.%m.%Y %H:%M"),
        "Способ определения поставщика": random.choice([
            "Электронный аукцион", "Конкурс", "Запрос котировок", 
            "Запрос предложений", "Единственный поставщик"
        ]),
        "Регион": random.choice([
            "Москва", "Санкт-Петербург", "Московская область", 
            "Новосибирская область", "Свердловская область",
            "Республика Татарстан", "Краснодарский край"
        ]),
        "Заказчик": random.choice([
            "Министерство образования", "Министерство здравоохранения",
            "Городская администрация", "Федеральное агентство",
            "Государственная компания", "Муниципальное учреждение"
        ]),
        "Требования к участникам": random.choice([
            "Наличие опыта выполнения аналогичных контрактов",
            "Наличие необходимых лицензий",
            "Отсутствие задолженностей по налогам",
            "Соответствие требованиям 44-ФЗ",
            "Наличие квалифицированных специалистов"
        ]),
        "Критерии оценки заявок": random.choice([
            "Цена контракта", "Срок выполнения", "Квалификация участника",
            "Качество предлагаемых товаров", "Опыт выполнения аналогичных работ"
        ]),
        "Валюта": "Рубль",
        "Источник финансирования": random.choice([
            "Федеральный бюджет", "Региональный бюджет", 
            "Местный бюджет", "Внебюджетные источники"
        ]),
        "Категория": random.choice([
            "Техника и оборудование", "Строительные работы", 
            "Услуги", "Товары", "Научные исследования"
        ])
    }
    
    # Apply overrides
    tender.update(overrides)
    return tender


def generate_tender_batch(n, start_id=0, **common_overrides):
    """
    Generate a batch of synthetic tenders.
    
    Args:
        n: Number of tenders to generate
        start_id: Starting ID for tender numbering
        **common_overrides: Common overrides for all tenders
        
    Returns:
        List of tender dictionaries
    """
    tenders = []
    for i in range(n):
        tender_id = start_id + i
        tender = generate_tender(tender_id, **common_overrides)
        tenders.append(tender)
    return tenders


def generate_edge_case_tenders():
    """
    Generate tenders specifically designed to test edge cases.
    
    Returns:
        List of edge case tender dictionaries
    """
    edge_cases = []
    
    # 1. Empty tender
    edge_cases.append({})
    
    # 2. Minimal tender
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "MINIMAL",
        "Наименование объекта закупки": "Минимальный"
    })
    
    # 3. Very long fields
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "LONG",
        "Наименование объекта закупки": "A" * 10000,
        "Наименование закупки": "B" * 5000,
        "Требования к участникам": "C" * 20000,
    })
    
    # 4. Special characters
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "SPECIAL",
        "Наименование объекта закупки": "Тест с \"кавычками\", & амперсандами, <тегами>, 'апострофами'",
        "Заказчик": "ООО \"Рога & копыта\" © 2024",
        "Регион": "Москва & область"
    })
    
    # 5. Unicode
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "UNICODE",
        "Наименование объекта закупки": "Поставка αβγδε equipment με ελληνικά",
        "Заказчик": "Компания © 2024 • Все права защищены",
        "Регион": "Санкт-Петербург"
    })
    
    # 6. Numeric edge cases
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "NUMERIC",
        "Наименование объекта закупки": "1234567890",
        "Начальная (максимальная) цена контракта": "0,00",
        "Регион": "77 регион"
    })
    
    # 7. Very large number
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "BIGNUM",
        "Наименование объекта закупки": "Крупный проект",
        "Начальная (максимальная) цена контракта": "9 999 999 999 999,99"
    })
    
    # 8. Negative number (if supported)
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "NEGATIVE",
        "Наименование объекта закупки": "Тест с отрицательной ценой",
        "Начальная (максимальная) цена контракта": "-500 000,00"
    })
    
    # 9. Date edge cases
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "DATES",
        "Наименование объекта закупки": "Тест дат",
        "Дата публикации": "01.01.1970 00:00",  # Epoch
        "Дата окончания подачи заявок": "31.12.2099 23:59"  # Far future
    })
    
    # 10. Missing fields
    edge_cases.append({
        "Идентификационный код закупки (ИКЗ)": "MISSING",
        # Intentionally missing most fields
    })
    
    return edge_cases


def generate_clustered_tenders(n_per_cluster=20, n_clusters=3, noise=5):
    """
    Generate tenders with clear cluster structure for testing clustering.
    
    Args:
        n_per_cluster: Number of tenders per cluster
        n_clusters: Number of clusters
        noise: Number of noise tenders (not in any cluster)
        
    Returns:
        List of tender dictionaries with cluster labels
    """
    all_tenders = []
    cluster_labels = []
    
    # Cluster themes
    cluster_themes = [
        {
            "category": "Техника и оборудование",
            "region": "Москва",
            "method": "Электронный аукцион",
            "price_range": (100000, 500000)
        },
        {
            "category": "Строительные работы",
            "region": "Санкт-Петербург",
            "method": "Конкурс",
            "price_range": (500000, 2000000)
        },
        {
            "category": "Услуги",
            "region": "Московская область",
            "method": "Запрос котировок",
            "price_range": (50000, 300000)
        }
    ]
    
    tender_id = 0
    
    # Generate cluster tenders
    for cluster_id in range(n_clusters):
        theme = cluster_themes[cluster_id % len(cluster_themes)]
        
        for i in range(n_per_cluster):
            overrides = {
                "Категория": theme["category"],
                "Регион": theme["region"],
                "Способ определения поставщика": theme["method"],
                "Начальная (максимальная) цена контракта": f"{random.randint(*theme['price_range']):,}".replace(',', ' ') + ",00",
                "Наименование объекта закупки": f"{theme['category']} {i+1}"
            }
            tender = generate_tender(tender_id, **overrides)
            all_tenders.append(tender)
            cluster_labels.append(cluster_id)
            tender_id += 1
    
    # Generate noise tenders
    for i in range(noise):
        # Mix attributes from different clusters
        overrides = {
            "Категория": random.choice(["Техника и оборудование", "Строительные работы", "Услуги"]),
            "Регион": random.choice(["Москва", "Санкт-Петербург", "Московская область", "Новосибирск"]),
            "Способ определения поставщика": random.choice(["Электронный аукцион", "Конкурс", "Запрос котировок", "Единственный поставщик"]),
            "Начальная (максимальная) цена контракта": f"{random.randint(10000, 10000000):,}".replace(',', ' ') + ",00"
        }
        tender = generate_tender(tender_id, **overrides)
        all_tenders.append(tender)
        cluster_labels.append(-1)  # Noise label
        tender_id += 1
    
    return all_tenders, cluster_labels


def generate_similarity_test_pairs():
    """
    Generate test pairs for similarity testing.
    
    Returns:
        List of tuples (tender1, tender2, expected_similarity_description)
    """
    base_tender = generate_tender(0)
    
    pairs = []
    
    # 1. Identical tenders
    pairs.append((base_tender, base_tender.copy(), "identical"))
    
    # 2. Same category, different region
    tender2 = base_tender.copy()
    tender2["Регион"] = "Санкт-Петербург"
    pairs.append((base_tender, tender2, "same_category_different_region"))
    
    # 3. Different category, same region
    tender3 = base_tender.copy()
    tender3["Категория"] = "Строительные работы"
    pairs.append((base_tender, tender3, "different_category_same_region"))
    
    # 4. Same price range (±10%)
    tender4 = base_tender.copy()
    price = float(base_tender["Начальная (максимальная) цена контракта"].replace(' ', '').replace(',', '.'))
    new_price = price * 1.1
    tender4["Начальная (максимальная) цена контракта"] = f"{new_price:,.2f}".replace(',', ' ').replace('.', ',')
    pairs.append((base_tender, tender4, "similar_price"))
    
    # 5. Very different price (10x)
    tender5 = base_tender.copy()
    new_price = price * 10
    tender5["Начальная (максимальная) цена контракта"] = f"{new_price:,.2f}".replace(',', ' ').replace('.', ',')
    pairs.append((base_tender, tender5, "different_price"))
    
    # 6. Same date (within week)
    tender6 = base_tender.copy()
    date_obj = datetime.strptime(base_tender["Дата публикации"], "%d.%m.%Y %H:%M")
    new_date = date_obj + timedelta(days=3)
    tender6["Дата публикации"] = new_date.strftime("%d.%m.%Y %H:%M")
    pairs.append((base_tender, tender6, "similar_date"))
    
    # 7. Very different date (1 year)
    tender7 = base_tender.copy()
    new_date = date_obj + timedelta(days=365)
    tender7["Дата публикации"] = new_date.strftime("%d.%m.%Y %H:%M")
    pairs.append((base_tender, tender7, "different_date"))
    
    # 8. Completely different
    tender8 = generate_tender(999, 
                             Категория="Научные исследования",
                             Регион="Новосибирская область",
                             Способ определения поставщика="Единственный поставщик",
                             Начальная (максимальная) цена контракта="50 000,00")
    pairs.append((base_tender, tender8, "completely_different"))
    
    return pairs


def generate_stress_test_tenders(n=10000):
    """
    Generate a large number of tenders for stress/performance testing.
    
    Args:
        n: Number of tenders to generate
        
    Returns:
        List of tender dictionaries
    """
    print(f"Generating {n} tenders for stress testing...")
    
    tenders = []
    for i in range(n):
        # Use simple generation for speed
        tender = {
            "Идентификационный код закупки (ИКЗ)": f"STRESS{i:08d}",
            "Наименование объекта закупки": f"Тендер {i}",
            "Начальная (максимальная) цена контракта": f"{random.randint(10000, 10000000):,}".replace(',', ' ') + ",00",
            "Регион": random.choice(["Москва", "Санкт-Петербург", "Московская область"]),
            "Способ определения поставщика": random.choice(["Электронный аукцион", "Конкурс"]),
            "Дата публикации": (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))).strftime("%d.%m.%Y")
        }
        tenders.append(tender)
    
    print(f"Generated {len(tenders)} tenders")
    return tenders


def generate_malicious_inputs():
    """
    Generate potentially malicious inputs for security testing.
    
    Returns:
        Dictionary of malicious inputs by category
    """
    return {
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "/absolute/../etc/passwd",
            "C:\\Windows\\..\\..\\Autoexec.bat"
        ],
        "sql_injection": [
            "'; DROP TABLE tenders; --",
            "' OR '1'='1",
            "1; INSERT INTO users VALUES ('hacker', 'password')",
            "admin'--"
        ],
        "command_injection": [
            "$(rm -rf /)",
            "| cat /etc/passwd",
            "; ls -la",
            "`id`",
            "&& shutdown -h now"
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert('xss')",
            "<svg/onload=alert(1)>"
        ],
        "json_injection": [
            '{"__class__": "os.system", "__args__": ["ls"]}',
            '{"__reduce__": "evil"}',
            '{"key": "value", "__import__": "os"}'
        ],
        "large_inputs": [
            "A" * 1000000,  # 1MB
            "A" * 10000000,  # 10MB
            "A" * 100000000,  # 100MB (be careful!)
        ],
        "special_characters": [
            "\x00\x01\x02\x03",  # Binary
            "\n\r\t\b\f",  # Control characters
            "\\\"\'",  # Escaped quotes
            "©®™€",  # Special symbols
            "αβγδε",  # Unicode
        ]
    }


def save_tenders_to_json(tenders, filepath):
    """
    Save tenders to a JSON file.
    
    Args:
        tenders: List of tender dictionaries
        filepath: Path to save JSON file
    """
    import json
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({"tenders": tenders}, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(tenders)} tenders to {filepath}")


def load_tenders_from_json(filepath):
    """
    Load tenders from a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of tender dictionaries
    """
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("tenders", [])


# Example usage
if __name__ == "__main__":
    # Generate sample data
    print("Generating sample test data...")
    
    # Small sample
    sample_tenders = generate_tender_batch(10)
    print(f"Generated {len(sample_tenders)} sample tenders")
    
    # Edge cases
    edge_tenders = generate_edge_case_tenders()
    print(f"Generated {len(edge_tenders)} edge case tenders")
    
    # Clustered data
    clustered_tenders, labels = generate_clustered_tenders(n_per_cluster=5, n_clusters=2, noise=3)
    print(f"Generated {len(clustered_tenders)} clustered tenders with {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
    
    # Similarity test pairs
    similarity_pairs = generate_similarity_test_pairs()
    print(f"Generated {len(similarity_pairs)} similarity test pairs")
    
    # Save to files for later use
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_file = os.path.join(tmpdir, "sample_tenders.json")
        save_tenders_to_json(sample_tenders, sample_file)
        
        edge_file = os.path.join(tmpdir, "edge_tenders.json")
        save_tenders_to_json(edge_tenders, edge_file)
        
        print(f"Sample data saved to temporary directory: {tmpdir}")
    
    print("Test data generation complete!")