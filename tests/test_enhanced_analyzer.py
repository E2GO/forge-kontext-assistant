# tests/test_enhanced_analyzer.py

import pytest
from ka_modules.analyzer_enhanced import EnhancedAnalyzer
from PIL import Image
import numpy as np

class TestEnhancedAnalyzer:
    """Тесты для расширенного анализатора"""
    
    @pytest.fixture
    def analyzer(self):
        return EnhancedAnalyzer()
    
    @pytest.fixture
    def test_image(self):
        # Создаем тестовое изображение
        return Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    def test_backward_compatibility(self, analyzer, test_image):
        """Проверка обратной совместимости"""
        # По умолчанию должен работать как базовый
        result = analyzer.analyze_image(test_image)
        
        assert 'objects' in result
        assert 'description' in result
        assert 'enhanced_analysis' not in result  # Расширенный анализ выключен
    
    def test_enhanced_mode(self, analyzer, test_image):
        """Проверка расширенного режима"""
        # Включаем расширенный режим явно
        result = analyzer.analyze_image(test_image, enhanced=True)
        
        assert 'objects' in result  # Базовые поля на месте
        assert 'description' in result
        assert 'enhanced_analysis' in result  # Добавлен расширенный анализ
        
        enhanced = result['enhanced_analysis']
        assert 'fantasy_elements' in enhanced
        assert 'materials' in enhanced
        assert 'fine_details' in enhanced
        assert 'lighting' in enhanced
        assert 'style' in enhanced
    
    def test_graceful_degradation(self, analyzer, test_image):
        """Проверка устойчивости к ошибкам"""
        # Симулируем ошибку в расширенном анализе
        def failing_analysis(*args):
            raise Exception("Test error")
        
        analyzer._perform_enhanced_analysis = failing_analysis
        
        # Должен вернуть базовый результат без падения
        result = analyzer.analyze_image(test_image, enhanced=True)
        assert 'enhanced_analysis' not in result
        assert 'description' in result  # Базовый функционал работает