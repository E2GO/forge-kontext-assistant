"""
Test integration without Forge dependencies
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add the extension root to sys.path
extension_root = Path(__file__).parent
sys.path.insert(0, str(extension_root))

def create_test_image(color_name, size=(512, 512)):
    """Create a simple colored test image"""
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255)
    }
    color = colors.get(color_name, (100, 100, 100))
    
    img_array = np.full((size[0], size[1], 3), color, dtype=np.uint8)
    return Image.fromarray(img_array)

def test_standalone_integration():
    print("=== Testing Standalone Integration ===\n")
    
    # Test 1: Shared State
    print("1. Testing Shared State Module...")
    try:
        from ka_modules.shared_state import shared_state
        print("   ✓ Imported shared_state successfully")
        
        # Create and store test images
        images = [
            create_test_image('red'),
            create_test_image('green'),
            create_test_image('blue')
        ]
        
        shared_state.set_images(images)
        print(f"   ✓ Stored {shared_state.image_count} images")
        
        # Retrieve and verify
        retrieved = shared_state.get_images()
        print(f"   ✓ Retrieved {len(retrieved)} images")
        
        # Test individual access
        for i in range(3):
            img = shared_state.get_image(i)
            if img is not None:
                print(f"   ✓ Image {i+1} accessible individually")
                
    except Exception as e:
        print(f"   ✗ Shared state error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Ka Modules
    print("\n2. Testing KA Modules...")
    
    # Test templates
    try:
        from ka_modules.templates import PromptTemplates
        templates = PromptTemplates()
        print("   ✓ PromptTemplates loaded")
        
        # Test getting a template
        template = templates.get_template('object_manipulation', 'color_change')
        if template:
            print("   ✓ Template retrieval working")
        
    except Exception as e:
        print(f"   ✗ Templates error: {e}")
    
    # Test prompt generator
    try:
        from ka_modules.prompt_generator import PromptGenerator
        from ka_modules.templates import PromptTemplates
        
        templates = PromptTemplates()
        generator = PromptGenerator(templates)
        print("   ✓ PromptGenerator loaded")
        
        # Test generation
        prompt = generator.generate(
            task_type="object_manipulation",
            user_intent="make the car blue",
            image_analysis=None
        )
        
        if prompt and not prompt.startswith("Error"):
            print("   ✓ Prompt generation working")
            print(f"     Generated: {prompt[:60]}...")
        
    except Exception as e:
        print(f"   ✗ Generator error: {e}")
    
    # Test image analyzer
    try:
        from ka_modules.image_analyzer import ImageAnalyzer
        analyzer = ImageAnalyzer()
        print("   ✓ ImageAnalyzer loaded")
        
        # Test with a simple image
        test_img = create_test_image('red', (256, 256))
        result = analyzer.analyze(test_img)
        
        if result and 'size' in result:
            print("   ✓ Image analysis working")
            print(f"     Analysis includes: {list(result.keys())}")
        
    except Exception as e:
        print(f"   ✗ Analyzer error: {e}")
    
    # Test 3: Integration Flow
    print("\n3. Testing Integration Flow...")
    try:
        # Simulate the flow
        from ka_modules.shared_state import shared_state
        from ka_modules.image_analyzer import ImageAnalyzer
        from ka_modules.prompt_generator import PromptGenerator
        from ka_modules.templates import PromptTemplates
        
        # Create components
        analyzer = ImageAnalyzer()
        templates = PromptTemplates()
        generator = PromptGenerator(templates)
        
        # Create test image
        test_img = create_test_image('blue', (512, 512))
        
        # Store in shared state
        shared_state.set_images([test_img, None, None])
        print("   ✓ Image stored in shared state")
        
        # Retrieve from shared state
        retrieved_img = shared_state.get_image(0)
        if retrieved_img:
            print("   ✓ Image retrieved from shared state")
            
            # Analyze
            analysis = analyzer.analyze(retrieved_img)
            print("   ✓ Image analyzed")
            
            # Generate prompt
            prompt = generator.generate(
                task_type="style_transfer",
                user_intent="watercolor painting style",
                image_analysis=analysis
            )
            
            if prompt:
                print("   ✓ Prompt generated with analysis")
                print(f"     Final prompt: {prompt[:80]}...")
        
    except Exception as e:
        print(f"   ✗ Integration flow error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: File Structure
    print("\n4. Checking File Structure...")
    required_files = [
        "ka_modules/__init__.py",
        "ka_modules/shared_state.py",
        "ka_modules/templates.py",
        "ka_modules/prompt_generator.py",
        "ka_modules/image_analyzer.py",
        "scripts/kontext.py",
        "scripts/kontext_assistant.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"   ✓ {file_path} ({size} bytes)")
        else:
            print(f"   ✗ {file_path} - MISSING")
    
    print("\n=== Standalone Integration Test Complete ===")
    print("\nNote: This test runs without Forge WebUI dependencies.")
    print("The actual integration in WebUI should work if all tests above passed.")

if __name__ == "__main__":
    test_standalone_integration()