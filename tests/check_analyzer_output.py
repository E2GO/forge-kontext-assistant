"""Check the actual output structure of SmartImageAnalyzer"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ka_modules.image_analyzer import SmartImageAnalyzer
from PIL import Image
import numpy as np
import json

def check_analyzer_output():
    """Проверяем реальную структуру вывода анализатора"""
    print("=" * 60)
    print("CHECKING ANALYZER OUTPUT STRUCTURE")
    print("=" * 60)
    
    # Создаем тестовое изображение
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # 1. Проверяем базовый анализатор
    print("\n1. SmartImageAnalyzer output:")
    analyzer = SmartImageAnalyzer()
    
    # Включаем mock режим для быстрого теста
    analyzer.set_mock_mode(True)
    result = analyzer.analyze(test_image)
    
    print("\nFull result structure:")
    print(json.dumps(result, indent=2, default=str))
    
    # 2. Проверяем типы данных
    print("\n2. Data types in result:")
    for key, value in result.items():
        print(f"  {key}: {type(value).__name__}")
        if key == 'objects' and isinstance(value, list) and value:
            print(f"    First object type: {type(value[0]).__name__}")
            if isinstance(value[0], dict):
                print(f"    Object keys: {list(value[0].keys())}")
        
    # 3. Проверяем objects_detailed
    if 'objects_detailed' in result:
        print("\n3. Objects detailed structure:")
        objs = result['objects_detailed']
        if objs:
            print(f"  Type: {type(objs).__name__}")
            if isinstance(objs, list) and objs:
                print(f"  First item: {objs[0]}")
    
    # 4. Ищем composition
    print("\n4. Looking for 'composition' field:")
    if 'composition' in result:
        print(f"  ✓ Found 'composition': {result['composition']}")
    else:
        print("  ✗ 'composition' not found in base analyzer")
    
    return result


def check_enhanced_analyzer_output():
    """Проверяем, откуда берется composition в EnhancedAnalyzer"""
    print("\n" + "=" * 60)
    print("CHECKING ENHANCED ANALYZER")
    print("=" * 60)
    
    from ka_modules.analyzer_enhanced import EnhancedAnalyzer
    
    # Создаем тестовое изображение
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    analyzer = EnhancedAnalyzer()
    analyzer.set_mock_mode(True)
    
    # Анализ без enhanced режима
    result = analyzer.analyze(test_image, enhanced=False)
    
    print("\nEnhancedAnalyzer result keys:")
    print(list(result.keys()))
    
    # Проверяем, добавляет ли EnhancedAnalyzer поле composition
    parent_result = super(EnhancedAnalyzer, analyzer).analyze(test_image)
    
    print("\nParent class result keys:")
    print(list(parent_result.keys()))
    
    # Сравниваем
    extra_keys = set(result.keys()) - set(parent_result.keys())
    if extra_keys:
        print(f"\nExtra keys added by EnhancedAnalyzer: {extra_keys}")


if __name__ == "__main__":
    base_result = check_analyzer_output()
    check_enhanced_analyzer_output()