"""Test enhanced analyzer with real data structure"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ka_modules.analyzer_enhanced import EnhancedAnalyzer
from PIL import Image
import numpy as np
import json

def test_with_real_structure():
    """Тест с реальной структурой данных"""
    print("=" * 60)
    print("TESTING ENHANCED ANALYZER WITH REAL DATA")
    print("=" * 60)
    
    # Создаем тестовое изображение
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # 1. Создаем анализатор
    analyzer = EnhancedAnalyzer()
    analyzer.set_mock_mode(True)
    
    # 2. Тест базового режима
    print("\n1. Testing basic mode (enhanced=False):")
    result_basic = analyzer.analyze(test_image, enhanced=False)
    print(f"✓ Basic analysis completed")
    print(f"  Keys: {list(result_basic.keys())}")
    
    # 3. Тест расширенного режима
    print("\n2. Testing enhanced mode (enhanced=True):")
    analyzer.set_enhanced_mode(True)
    result_enhanced = analyzer.analyze(test_image)
    
    if 'enhanced_analysis' in result_enhanced:
        print("✓ Enhanced analysis completed successfully!")
        
        enhanced = result_enhanced['enhanced_analysis']
        print("\nEnhanced analysis results:")
        
        # Выводим найденные элементы
        if enhanced['fantasy_elements']:
            print(f"  Fantasy elements: {len(enhanced['fantasy_elements'])} groups found")
            for group in enhanced['fantasy_elements'][:2]:  # Первые 2 группы
                print(f"    - {group['category']}: {group['count']} elements")
        
        if enhanced['materials']:
            print(f"  Materials: {len(enhanced['materials'])} found")
            for mat in enhanced['materials'][:3]:  # Первые 3 материала
                print(f"    - {mat['material']} ({mat['type']})")
        
        if enhanced['fine_details']:
            print(f"  Fine details: {len(enhanced['fine_details'])} found")
        
        if enhanced['lighting']:
            lighting = enhanced['lighting']
            if lighting.get('time_of_day'):
                print(f"  Lighting: {lighting['time_of_day']}")
            if lighting.get('mood'):
                print(f"  Mood: {lighting['mood']}")
        
        if enhanced['style']:
            style = enhanced['style']
            if style.get('primary_style'):
                print(f"  Primary style: {style['primary_style']}")
        
        print(f"\n  Total elements found: {enhanced['total_elements_found']}")
        print(f"  Categories detected: {enhanced['detected_categories']}")
        
    else:
        print("✗ Enhanced analysis failed")
        if 'enhanced_analysis_error' in result_enhanced:
            print(f"  Error: {result_enhanced['enhanced_analysis_error']}")
    
    # 4. Тест с кастомными данными для проверки детекции
    print("\n3. Testing with custom fantasy description:")
    
    # Создаем кастомный мок анализатор с фэнтези описанием
    class CustomMockAnalyzer(EnhancedAnalyzer):
        def analyze(self, image, task_type=None, enhanced=None):
            # Переопределяем базовый результат для теста
            if hasattr(self, '_use_custom_result'):
                # Подменяем базовый результат
                original_analyze = super().analyze
                result = self._custom_result.copy()
                result['analysis_time'] = 0.1
                
                # Если enhanced, добавляем расширенный анализ
                if enhanced or self.enhanced_mode:
                    try:
                        enhanced_result = self._perform_enhanced_analysis(image, result)
                        result['enhanced_analysis'] = enhanced_result
                    except Exception as e:
                        result['enhanced_analysis_error'] = str(e)
                
                return result
            else:
                return super().analyze(image, task_type, enhanced)
    
    # Создаем кастомный результат с фэнтези элементами
    custom_result = {
        "size": "512x512",
        "mode": "RGB",
        "analysis_mode": "custom_test",
        "description": "A dragon knight in plate armor with glowing eyes and demon horns, wielding a magical sword with fire magic effects. The scene is painted in oil painting style with dramatic rim lighting at sunset. The knight wears golden armor with ruby gems and leather straps.",
        "objects": {
            "main": ["knight", "sword", "armor", "dragon"],
            "secondary": ["fire", "gems", "horns"],
            "all": ["knight", "sword", "armor", "dragon", "fire", "gems", "horns"]
        },
        "style": {
            "type": "oil painting",
            "mood": "dramatic",
            "lighting": "rim lighting",
            "color_palette": "warm"
        },
        "environment": {
            "setting": "fantasy battlefield",
            "time_of_day": "sunset",
            "weather": "clear"
        },
        "composition": {
            "aspect_ratio": 1.0,
            "orientation": "portrait",
            "complexity": "high"
        }
    }
    
    custom_analyzer = CustomMockAnalyzer()
    custom_analyzer._use_custom_result = True
    custom_analyzer._custom_result = custom_result
    
    result_custom = custom_analyzer.analyze(test_image, enhanced=True)
    
    if 'enhanced_analysis' in result_custom:
        print("✓ Custom fantasy analysis completed!")
        
        enhanced = result_custom['enhanced_analysis']
        
        # Детальный вывод найденных фэнтези элементов
        if enhanced['fantasy_elements']:
            print(f"\n  Found {len(enhanced['fantasy_elements'])} fantasy element groups:")
            for group in enhanced['fantasy_elements']:
                print(f"    {group['category']}: {group['count']} items")
                for elem in group['elements'][:3]:  # Первые 3 элемента
                    print(f"      - {elem['detected']} (confidence: {elem['confidence']:.2f})")
        
        # Материалы
        if enhanced['materials']:
            print(f"\n  Found {len(enhanced['materials'])} materials:")
            for mat in enhanced['materials'][:5]:
                print(f"    - {mat['material']} ({mat['type']}/{mat['subcategory']})")
        
        # Стиль
        if enhanced['style']['primary_style']:
            print(f"\n  Detected style: {enhanced['style']['primary_style']}")
            if enhanced['style']['techniques']:
                print(f"  Techniques: {', '.join(enhanced['style']['techniques'])}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_with_real_structure()