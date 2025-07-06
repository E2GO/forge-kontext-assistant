#!/usr/bin/env python3
"""
Simple test script for Florence-2 integration without external dependencies.
"""

import sys
import os

# Fix for Python 3.10 compatibility
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from pathlib import Path
from PIL import Image
import time
import torch

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from ka_modules.image_analyzer import ImageAnalyzer, Florence2ModelLoader
from ka_modules.prompt_generator import PromptGenerator
from ka_modules.templates import PromptTemplates


def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def test_basic_functionality():
    """Test basic Florence-2 functionality with a simple image."""
    print_header("Florence-2 Integration Test (Simple)")
    
    # 1. Check environment
    print("1. Environment Check:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 2. Create test image
    print("\n2. Creating test image...")
    test_image = Image.new('RGB', (512, 512), color='red')
    # Add some variation
    pixels = test_image.load()
    for i in range(100, 200):
        for j in range(100, 200):
            pixels[i, j] = (0, 0, 255)  # Blue square
    for i in range(300, 400):
        for j in range(300, 400):
            pixels[i, j] = (0, 255, 0)  # Green square
    print(f"   ✅ Created test image with colored squares")
    
    # 3. Test model loading
    print("\n3. Loading Florence-2 model...")
    loader = Florence2ModelLoader()
    
    try:
        start_time = time.time()
        loader.initialize()
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"   Device: {loader.device}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        print("\n   This might be due to:")
        print("   - First time download (requires internet)")
        print("   - Try running: pip install --upgrade transformers")
        print("   - Florence-2 compatibility issue")
        print("\n   Attempting alternative loading method...")
        
        # Try without device specification
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            print("   Loading model directly...")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            print("   ✅ Alternative loading successful!")
            print("   Note: You may need to update the image_analyzer.py")
            return False
        except Exception as e2:
            print(f"   ❌ Alternative loading also failed: {e2}")
            return False
    
    # 4. Test image analysis
    print("\n4. Analyzing test image...")
    analyzer = ImageAnalyzer()
    
    try:
        start_time = time.time()
        result = analyzer.analyze(test_image, use_cache=False)
        analysis_time = time.time() - start_time
        
        print(f"   ✅ Analysis completed in {analysis_time:.2f} seconds")
        print(f"\n   Results:")
        print(f"   - Caption: {result.raw_caption[:100]}...")
        print(f"   - Main objects: {result.objects['main']}")
        print(f"   - Style: {result.style['artistic']}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test prompt generation
    print("\n5. Testing prompt generation...")
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    
    try:
        prompt = generator.generate(
            task_type="object_manipulation",
            user_intent="make the red area blue",
            image_analysis={'objects': result.objects['main']},
            subtype="color_change"
        )
        print(f"   ✅ Generated prompt: {prompt}")
    except Exception as e:
        print(f"   ❌ Prompt generation failed: {e}")
        return False
    
    # 6. Test memory cleanup
    print("\n6. Testing memory cleanup...")
    if cuda_available:
        before_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory before cleanup: {before_mem:.2f} GB")
    
    analyzer.unload_model()
    torch.cuda.empty_cache() if cuda_available else None
    
    if cuda_available:
        after_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"   Memory after cleanup: {after_mem:.2f} GB")
        print(f"   ✅ Freed {before_mem - after_mem:.2f} GB")
    
    return True


def test_error_handling():
    """Test error handling and fallbacks."""
    print_header("Testing Error Handling")
    
    analyzer = ImageAnalyzer()
    
    # Test with invalid image
    print("1. Testing with None image...")
    try:
        result = analyzer.analyze(None)
        print("   ❌ Should have raised an error!")
    except Exception as e:
        print(f"   ✅ Correctly caught error: {type(e).__name__}")
    
    # Test with very small image
    print("\n2. Testing with tiny image...")
    tiny_image = Image.new('RGB', (10, 10), color='red')
    try:
        result = analyzer.analyze(tiny_image)
        print(f"   ✅ Handled tiny image gracefully")
        print(f"   Result type: {type(result)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n   ✅ Error handling test completed")


def main():
    """Run simple tests."""
    print("\n" + "="*60)
    print("  FLORENCE-2 INTEGRATION TEST (SIMPLE)")
    print("  Kontext Assistant V2")
    print("="*60)
    
    # Run basic test
    success = test_basic_functionality()
    
    if success:
        # Run error handling test
        test_error_handling()
        
        print_header("Test Summary")
        print("✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Try the assistant in Forge WebUI")
        print("2. Load some real images for analysis")
        print("3. Test different prompt generation tasks")
    else:
        print_header("Test Summary")
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure transformers library is installed:")
        print("   pip install transformers")
        print("2. Check internet connection for model download")
        print("3. Ensure sufficient disk space (~1.5GB) and VRAM")


if __name__ == "__main__":
    main()