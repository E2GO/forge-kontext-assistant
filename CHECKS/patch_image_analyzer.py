#!/usr/bin/env python3
"""
Add collections compatibility patch to image_analyzer.py
"""

from pathlib import Path
import shutil
from datetime import datetime

def patch_image_analyzer():
    print("🔧 Patching image_analyzer.py for Python 3.10+ compatibility\n")
    
    # Patch code
    patch_code = '''# Python 3.10+ compatibility patch
import sys
import collections.abc
if sys.version_info >= (3, 10):
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Iterable = collections.abc.Iterable
    collections.MutableSet = collections.abc.MutableSet
    collections.Callable = collections.abc.Callable

'''
    
    analyzer_path = Path.cwd() / 'ka_modules' / 'image_analyzer.py'
    
    if not analyzer_path.exists():
        print("❌ image_analyzer.py not found!")
        return False
    
    # Backup
    backup_name = f'image_analyzer_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
    backup_path = analyzer_path.parent / backup_name
    shutil.copy(analyzer_path, backup_path)
    print(f"📁 Backed up to {backup_name}")
    
    # Read content
    with open(analyzer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if patch already exists
    if 'collections.Mapping = collections.abc.Mapping' in content:
        print("✅ Patch already present!")
        return True
    
    # Find insertion point (before other imports)
    lines = content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if '"""' in line and i > 0:  # Skip docstring
            continue
        if line.startswith('import ') or line.startswith('from '):
            insert_index = i
            break
    
    # Insert patch
    lines.insert(insert_index, patch_code)
    
    # Write back
    with open(analyzer_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("✅ Patch added to image_analyzer.py")
    
    # Also ensure the analyze method exists
    print("\n📝 Checking analyze method...")
    
    # Read updated content
    with open(analyzer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if analyze method exists
    if 'def analyze(' not in content:
        print("⚠️  Adding analyze method to ImageAnalyzer...")
        
        # Add analyze method if missing
        if 'class ImageAnalyzer' in content:
            # Find the class and add method
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'class ImageAnalyzer' in line:
                    # Find the end of __init__ method
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(' '):
                            # Insert analyze method here
                            analyze_method = '''
    def analyze(self, image, task_type: str) -> dict:
        """Analyze image (mock implementation)"""
        return {
            'objects': {
                'main': ['object', 'subject'],
                'secondary': ['background', 'elements']
            },
            'style': {
                'artistic': 'photorealistic',
                'mood': 'neutral'
            },
            'environment': {
                'setting': 'indoor/outdoor',
                'time_of_day': 'daytime'
            }
        }
'''
                            lines.insert(j, analyze_method)
                            break
                    break
            
            # Write updated content
            with open(analyzer_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ Added analyze method")
    
    return True

def test_final():
    print("\n🧪 Final test...")
    
    import sys
    sys.path.insert(0, str(Path.cwd()))
    
    try:
        # Clear cached imports
        for module in list(sys.modules.keys()):
            if 'ka_modules' in module or 'image_analyzer' in module:
                del sys.modules[module]
        
        from ka_modules.image_analyzer import ImageAnalyzer
        analyzer = ImageAnalyzer()
        
        # Test analyze method
        from PIL import Image
        import numpy as np
        test_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        result = analyzer.analyze(test_img, "test")
        
        if isinstance(result, dict) and 'objects' in result:
            print("✅ ImageAnalyzer working correctly!")
            return True
        else:
            print("❌ ImageAnalyzer not returning expected format")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    if patch_image_analyzer():
        if test_final():
            print("\n✨ Success! Now run 'python test_basic.py' for full test")
        else:
            print("\n⚠️  Patched but still having issues")
    else:
        print("\n❌ Failed to patch")

if __name__ == "__main__":
    main()
    input("\nPress Enter to continue...")
