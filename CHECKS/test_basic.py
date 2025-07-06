#!/usr/bin/env python3
"""
Basic functionality test for forge-kontext-assistant
Tests core components after fixes
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add project to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

def create_test_image():
    """Create a simple test image"""
    # Create RGB image 512x512
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add some color - red square in center
    img_array[156:356, 156:356, 0] = 255  # Red channel
    return Image.fromarray(img_array)

def test_functionality():
    """Test basic functionality"""
    print("🧪 TESTING FORGE-KONTEXT-ASSISTANT\n")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Import modules
    print("1️⃣  Testing module imports...")
    tests_total += 1
    try:
        from ka_modules.templates import PromptTemplates
        from ka_modules.prompt_generator import PromptGenerator
        print("   ✅ Module imports successful")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
    
    # Test 2: Create instances
    print("\n2️⃣  Testing class instantiation...")
    tests_total += 1
    try:
        templates = PromptTemplates()
        generator = PromptGenerator()
        print("   ✅ Classes instantiated successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
        return
    
    # Test 3: Test templates
    print("\n3️⃣  Testing template generation...")
    tests_total += 1
    try:
        # Test object color template
        template = templates.get_object_color_template(
            target="car",
            current_color="red",
            new_color="blue"
        )
        if "Change" in template and "car" in template:
            print(f"   ✅ Template generated: '{template[:50]}...'")
            tests_passed += 1
        else:
            print(f"   ❌ Template seems wrong: {template}")
    except Exception as e:
        print(f"   ❌ Template generation failed: {e}")
    
    # Test 4: Test prompt generation
    print("\n4️⃣  Testing prompt generation...")
    tests_total += 1
    try:
        prompt = generator.generate(
            task_type="object_color",
            user_intent="make the car blue",
            context_strength=0.7
        )
        if prompt and len(prompt) > 20:
            print(f"   ✅ Prompt generated ({len(prompt)} chars)")
            print(f"      Preview: '{prompt[:60]}...'")
            tests_passed += 1
        else:
            print(f"   ❌ Prompt too short or empty")
    except Exception as e:
        print(f"   ❌ Prompt generation failed: {e}")
    
    # Test 5: Test validators
    print("\n5️⃣  Testing validators...")
    tests_total += 1
    try:
        from utils.validators import validate_image, validate_prompt
        
        # Test image validation
        test_img = create_test_image()
        valid, msg = validate_image(test_img)
        
        if valid:
            print(f"   ✅ Image validation works: {msg}")
            tests_passed += 1
        else:
            print(f"   ❌ Image validation failed: {msg}")
    except ImportError:
        print("   ⚠️  Validators not found (run setup_project.py)")
    except Exception as e:
        print(f"   ❌ Validation test failed: {e}")
    
    # Test 6: Test cache
    print("\n6️⃣  Testing cache...")
    tests_total += 1
    try:
        from utils.cache import ResultCache
        
        cache = ResultCache()
        cache.set("test_image", "test_task", {"result": "test"})
        cached = cache.get("test_image", "test_task")
        
        if cached and cached.get("result") == "test":
            print("   ✅ Cache working correctly")
            tests_passed += 1
        else:
            print("   ❌ Cache not working properly")
    except ImportError:
        print("   ⚠️  Cache module not found (run setup_project.py)")
    except Exception as e:
        print(f"   ❌ Cache test failed: {e}")
    
    # Test 7: Test image analyzer (mock mode)
    print("\n7️⃣  Testing image analyzer...")
    tests_total += 1
    try:
        from ka_modules.image_analyzer import ImageAnalyzer
        
        analyzer = ImageAnalyzer()
        test_img = create_test_image()
        result = analyzer.analyze(test_img, "object_color")
        
        if isinstance(result, dict) and 'objects' in result:
            print("   ✅ Image analyzer returns valid structure")
            print(f"      Objects found: {result['objects']['main']}")
            tests_passed += 1
        else:
            print("   ❌ Invalid analyzer output")
    except Exception as e:
        print(f"   ❌ Image analyzer test failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print(f"📊 RESULTS: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("✅ All tests passed! The system is ready to use.")
    elif tests_passed >= tests_total * 0.7:
        print("⚠️  Most tests passed. Some features may not work.")
    else:
        print("❌ Many tests failed. Run setup_project.py to fix issues.")
    
    print("\n💡 Note: This tests basic functionality only.")
    print("   Full integration with Forge WebUI requires restart.")

if __name__ == "__main__":
    test_functionality()
