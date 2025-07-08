# test_current_state.py - проверка текущей функциональности

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ka_modules.image_analyzer import ImageAnalyzer
from PIL import Image

def test_current_analyzer():
    """Тестируем текущие возможности"""
    analyzer = ImageAnalyzer()
    
    # Тестовое изображение
    test_image_path = "test_images/sample.jpg"  # Нужно подготовить
    
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path)
        result = analyzer.analyze_image(image)
        
        print("=== CURRENT ANALYSIS OUTPUT ===")
        print(f"Objects found: {result.get('objects', [])}")
        print(f"Description: {result.get('description', '')}")
        print(f"Style: {result.get('style', 'not detected')}")
        print("==============================")
        
        # Сохраняем baseline для сравнения
        with open("baseline_analysis.json", "w") as f:
            json.dump(result, f, indent=2)
    else:
        print("Please add test image to test_images/sample.jpg")

if __name__ == "__main__":
    test_current_analyzer()