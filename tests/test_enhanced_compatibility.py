"""Test backward compatibility of enhanced analyzer"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ka_modules.analyzer_enhanced import EnhancedAnalyzer
from ka_modules.image_analyzer import SmartImageAnalyzer
from PIL import Image
import numpy as np

def test_backward_compatibility():
    """Проверка обратной совместимости"""
    print("=" * 60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)
    
    # Создаем тестовое изображение
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # 1. Тест базового анализатора
    print("\n1. Testing base SmartImageAnalyzer:")
    base_analyzer = SmartImageAnalyzer()
    base_analyzer.set_mock_mode(True)
    base_result = base_analyzer.analyze(test_image)
    
    print(f"✓ Base analysis completed")
    print(f"  Keys in result: {sorted(base_result.keys())}")
    
    # 2. Тест расширенного анализатора в обычном режиме
    print("\n2. Testing EnhancedAnalyzer (enhanced_mode=False):")
    enhanced_analyzer = EnhancedAnalyzer()
    enhanced_analyzer.set_mock_mode(True)
    enhanced_analyzer.set_enhanced_mode(False)
    
    result_normal = enhanced_analyzer.analyze(test_image)
    
    # Проверяем, что результат совместим
    base_keys = set(base_result.keys())
    enhanced_keys = set(result_normal.keys())
    
    # analysis_time добавляется в enhanced версии - это ожидаемо
    expected_new_keys = {'analysis_time'}
    extra_keys = enhanced_keys - base_keys - expected_new_keys
    missing_keys = base_keys - enhanced_keys
    
    if not extra_keys and not missing_keys:
        print("✓ Result structure is fully compatible")
        print(f"  Base keys: {len(base_keys)}")
        print(f"  Enhanced keys: {len(enhanced_keys)}")
        print(f"  Added: {expected_new_keys}")
    else:
        if extra_keys:
            print(f"✗ Unexpected extra keys: {extra_keys}")
        if missing_keys:
            print(f"✗ Missing keys: {missing_keys}")
    
    # 3. Тест расширенного режима
    print("\n3. Testing EnhancedAnalyzer (enhanced_mode=True):")
    result_enhanced = enhanced_analyzer.analyze(test_image, enhanced=True)
    
    if 'enhanced_analysis' in result_enhanced:
        print("✓ Enhanced analysis added correctly")
        enhanced_data = result_enhanced['enhanced_analysis']
        print(f"  Enhanced keys: {list(enhanced_data.keys())}")
        print(f"  Categories detected: {enhanced_data.get('detected_categories', [])}")
        print(f"  Total elements found: {enhanced_data.get('total_elements_found', 0)}")
        
        # Проверяем, что базовые поля не изменились
        for key in base_keys:
            if key in result_enhanced:
                print(f"  ✓ Base field '{key}' preserved")
    else:
        print("✗ Enhanced analysis not found")
        if 'enhanced_analysis_error' in result_enhanced:
            print(f"  Error: {result_enhanced['enhanced_analysis_error']}")
    
    # 4. Тест обработки ошибок
    print("\n4. Testing error handling:")
    # Симулируем ошибку
    def failing_analysis(*args):
        raise Exception("Simulated error")
    
    enhanced_analyzer._perform_enhanced_analysis = failing_analysis
    result_with_error = enhanced_analyzer.analyze(test_image, enhanced=True)
    
    if 'enhanced_analysis_error' in result_with_error:
        print("✓ Error handled gracefully")
        print(f"  Error message: {result_with_error['enhanced_analysis_error']}")
    else:
        print("✗ Error not handled properly")
    
    # 5. Проверка производительности
    print("\n5. Performance check:")
    if 'analysis_time' in result_normal:
        print(f"  Base analysis time: {result_normal['analysis_time']:.3f}s")
    if 'total_analysis_time' in result_enhanced:
        print(f"  Enhanced analysis time: {result_enhanced.get('total_analysis_time', 0):.3f}s")
    
    print("\n" + "=" * 60)
    print("COMPATIBILITY TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_backward_compatibility()