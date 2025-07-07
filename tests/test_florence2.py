"""
Test Florence-2 integration
"""

# Fix collections compatibility BEFORE any other imports
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import sys
import os
from pathlib import Path
from PIL import Image
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test different modes
def test_image_analyzer():
    """Test ImageAnalyzer with different configurations"""
    
    print("=== Testing Image Analyzer ===\n")
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='blue')
    
    # Test 1: Default mode (auto-detect)
    print("Test 1: Auto mode")
    os.environ.pop("KONTEXT_USE_FLORENCE2", None)
    
    from ka_modules.image_analyzer import ImageAnalyzer
    analyzer = ImageAnalyzer()
    
    result = analyzer.analyze(test_image)
    print(f"Using Florence-2: {analyzer.use_florence2}")
    print(f"Description: {result['description'][:100]}...")
    print(f"Objects: {result['objects']['main'][:3]}")
    
    # Test 2: Force mock mode
    print("\nTest 2: Force mock mode")
    os.environ["KONTEXT_USE_FLORENCE2"] = "false"
    
    # Need to reimport to pick up env change
    import importlib
    import ka_modules.image_analyzer
    importlib.reload(ka_modules.image_analyzer)
    
    from ka_modules.image_analyzer import ImageAnalyzer as AnalyzerMock
    analyzer_mock = AnalyzerMock()
    
    result_mock = analyzer_mock.analyze(test_image)
    print(f"Using Florence-2: {analyzer_mock.use_florence2}")
    print(f"Description: {result_mock['description'][:100]}...")
    
    # Test 3: Force Florence-2 mode
    print("\nTest 3: Force Florence-2 mode")
    os.environ["KONTEXT_USE_FLORENCE2"] = "true"
    
    importlib.reload(ka_modules.image_analyzer)
    
    from ka_modules.image_analyzer import ImageAnalyzer as AnalyzerF2
    analyzer_f2 = AnalyzerF2()
    
    if analyzer_f2.use_florence2:
        print("Florence-2 is being used!")
        print("Loading model... (this may take a moment)")
        
        start_time = time.time()
        analyzer_f2.load_model()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Analyze with Florence-2
        start_time = time.time()
        result_f2 = analyzer_f2.analyze(test_image)
        analysis_time = time.time() - start_time
        
        print(f"Analysis took {analysis_time:.2f} seconds")
        print(f"Description: {result_f2['description'][:100]}...")
        print(f"Objects: {result_f2['objects']['main'][:3]}")
        
        # Test with a real image if available
        real_image_path = Path("test_image.jpg")
        if real_image_path.exists():
            print("\nTesting with real image...")
            real_image = Image.open(real_image_path)
            real_result = analyzer_f2.analyze(real_image)
            print(f"Real image description: {real_result['description']}")
            print(f"Detected objects: {real_result['objects']['main']}")
        
        # Unload model
        print("\nUnloading model...")
        analyzer_f2.unload_model()
        print("Model unloaded")
    else:
        print("Florence-2 not available or not enough resources")
    
    # Clean up environment
    os.environ.pop("KONTEXT_USE_FLORENCE2", None)
    
    print("\n=== Test Complete ===")

def test_with_kontext_images():
    """Test analyzer with typical kontext use cases"""
    
    print("\n=== Testing Kontext Use Cases ===\n")
    
    from ka_modules.image_analyzer import ImageAnalyzer
    from ka_modules.prompt_generator import PromptGenerator
    from ka_modules.templates import PromptTemplates
    
    # Initialize components
    analyzer = ImageAnalyzer()
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    
    # Test scenarios
    test_cases = [
        {
            "image": Image.new('RGB', (800, 600), color='red'),
            "task": "object_manipulation",
            "intent": "make the car blue"
        },
        {
            "image": Image.new('RGB', (1024, 768), color='green'),
            "task": "style_transfer", 
            "intent": "oil painting style"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test['task']}")
        
        # Analyze image
        analysis = analyzer.analyze(test['image'])
        print(f"Analysis complete: {len(analysis)} fields")
        
        # Generate prompt
        prompt = generator.generate(
            task_type=test['task'],
            user_intent=test['intent'],
            image_analysis=analysis
        )
        
        print(f"Generated prompt: {prompt[:150]}...")
        
        # Verify analysis influenced the prompt
        if analysis['objects']['main']:
            print(f"Main objects detected: {analysis['objects']['main']}")

if __name__ == "__main__":
    # Check requirements first
    print("Checking requirements...\n")
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        print("Install with: pip install transformers>=4.36.0")
        sys.exit(1)
    
    # Run tests
    test_image_analyzer()
    test_with_kontext_images()