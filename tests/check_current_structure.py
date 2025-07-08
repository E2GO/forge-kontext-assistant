"""Check current project structure"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ka_modules.image_analyzer import ImageAnalyzer
    print("✓ Found ImageAnalyzer")
except ImportError:
    print("✗ ImageAnalyzer not found")
    
try:
    from ka_modules.image_analyzer import SmartImageAnalyzer
    print("✓ Found SmartImageAnalyzer")
    
    # Проверяем методы
    print("\nMethods in SmartImageAnalyzer:")
    for attr in dir(SmartImageAnalyzer):
        if not attr.startswith('_') and callable(getattr(SmartImageAnalyzer, attr)):
            print(f"  - {attr}")
            
except ImportError as e:
    print(f"✗ SmartImageAnalyzer import error: {e}")

# Проверяем структуру ka_modules
print("\n\nChecking ka_modules structure:")
ka_modules_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ka_modules')

if os.path.exists(ka_modules_path):
    print(f"ka_modules path: {ka_modules_path}")
    print("Files in ka_modules:")
    for file in os.listdir(ka_modules_path):
        if file.endswith('.py'):
            print(f"  - {file}")
            
# Попробуем импортировать и посмотреть содержимое
try:
    import ka_modules.image_analyzer as img_analyzer
    print("\n\nClasses/functions in image_analyzer.py:")
    for item in dir(img_analyzer):
        if not item.startswith('_'):
            obj = getattr(img_analyzer, item)
            if isinstance(obj, type):
                print(f"  Class: {item}")
                # Методы класса
                methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))]
                if methods:
                    print(f"    Methods: {', '.join(methods[:5])}...")
except Exception as e:
    print(f"Error inspecting image_analyzer: {e}")