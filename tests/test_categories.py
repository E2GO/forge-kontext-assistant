"""Test script for analysis categories"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ka_modules.analysis_categories import AnalysisCategories, find_category_for_keyword

def test_categories():
    """Тестирование категорий анализа"""
    
    print("=" * 60)
    print("TESTING ANALYSIS CATEGORIES")
    print("=" * 60)
    
    # 1. Тест структуры
    print("\n1. Testing category structure:")
    categories = [
        'FANTASY_ELEMENTS',
        'MATERIALS', 
        'FINE_DETAILS',
        'LIGHTING_ATMOSPHERE',
        'ARTISTIC_STYLES'
    ]
    
    for cat_name in categories:
        if hasattr(AnalysisCategories, cat_name):
            cat = getattr(AnalysisCategories, cat_name)
            subcats = list(cat.keys())
            print(f"✓ {cat_name}: {len(subcats)} subcategories")
            print(f"  Subcategories: {', '.join(subcats[:3])}...")
    
    # 2. Тест поиска ключевых слов
    print("\n2. Testing keyword search:")
    test_cases = [
        # Fantasy elements
        ('dragon horn', 'Should find in fantasy body modifications'),
        ('glowing eyes', 'Should find in fantasy body modifications'),
        ('plate armor', 'Should find in medieval fantasy'),
        
        # Materials
        ('golden', 'Should find in precious metals'),
        ('leather', 'Should find in natural fabrics'),
        ('crystalline', 'Should find in synthetic energy'),
        
        # Fine details
        ('tattoo', 'Should find in body art'),
        ('ruby', 'Should find in gems'),
        
        # Lighting
        ('sunset', 'Should find in evening time'),
        ('fog', 'Should find in weather'),
        
        # Styles
        ('oil painting', 'Should find in traditional painting'),
        ('anime', 'Should find in eastern cultural styles')
    ]
    
    for keyword, description in test_cases:
        result = find_category_for_keyword(keyword)
        if result:
            print(f"✓ '{keyword}': {' > '.join(result)}")
        else:
            print(f"✗ '{keyword}': NOT FOUND - {description}")
    
    # 3. Статистика
    print("\n3. Category statistics:")
    total_keywords = len(AnalysisCategories.get_all_keywords())
    print(f"Total unique keywords: {total_keywords}")
    
    for cat_name in categories:
        keywords = AnalysisCategories.get_all_keywords(cat_name)
        print(f"{cat_name}: {len(keywords)} keywords")
    
    # 4. Проверка на дубликаты
    print("\n4. Checking for duplicates:")
    all_keywords_list = []
    for cat_name in categories:
        cat = getattr(AnalysisCategories, cat_name)
        _extract_all_keywords_list(cat, all_keywords_list)
    
    duplicates = [kw for kw in set(all_keywords_list) if all_keywords_list.count(kw) > 1]
    if duplicates:
        print(f"Found {len(duplicates)} duplicate keywords:")
        for dup in duplicates[:5]:
            print(f"  - '{dup}' appears {all_keywords_list.count(dup)} times")
    else:
        print("✓ No duplicate keywords found")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)


def _extract_all_keywords_list(data, keywords_list):
    """Извлечь все ключевые слова в список (с дубликатами)"""
    if isinstance(data, dict):
        for value in data.values():
            _extract_all_keywords_list(value, keywords_list)
    elif isinstance(data, list):
        keywords_list.extend(data)


if __name__ == "__main__":
    test_categories()