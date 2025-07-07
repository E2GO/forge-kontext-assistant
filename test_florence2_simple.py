"""
Simple test for Florence-2 functionality
"""

# Fix collections compatibility
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import os
os.environ["KONTEXT_USE_FLORENCE2"] = "true"  # Force Florence-2

from PIL import Image
from ka_modules.image_analyzer import ImageAnalyzer

def test_florence2():
    print("=== Testing Florence-2 ===\n")
    
    # Create analyzer
    analyzer = ImageAnalyzer()
    print(f"Using Florence-2: {analyzer.use_florence2}")
    
    if not analyzer.use_florence2:
        print("Florence-2 not available or not enabled")
        return
    
    # Create a simple test image
    print("\nCreating test image...")
    test_image = Image.new('RGB', (512, 512))
    # Draw something simple
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    # Red rectangle
    draw.rectangle([100, 100, 200, 200], fill='red')
    # Blue circle
    draw.ellipse([300, 300, 400, 400], fill='blue')
    # Green triangle
    draw.polygon([(256, 50), (306, 150), (206, 150)], fill='green')
    
    print("Loading Florence-2 model...")
    analyzer.load_model()
    
    print("\nAnalyzing image...")
    try:
        result = analyzer.analyze(test_image)
        
        print("\n=== Analysis Results ===")
        print(f"Description: {result.get('description', 'No description')}")
        print(f"Main objects: {result.get('objects', {}).get('main', [])}")
        print(f"Style: {result.get('style', {}).get('artistic', 'Unknown')}")
        print(f"Setting: {result.get('environment', {}).get('setting', 'Unknown')}")
        
        # Test with a real photo if available
        if os.path.exists("test_photo.jpg"):
            print("\n=== Testing with real photo ===")
            real_image = Image.open("test_photo.jpg")
            real_result = analyzer.analyze(real_image)
            print(f"Description: {real_result.get('description', 'No description')}")
            print(f"Objects detected: {real_result.get('objects', {}).get('main', [])}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nUnloading model...")
    analyzer.unload_model()
    print("Done!")

if __name__ == "__main__":
    test_florence2()