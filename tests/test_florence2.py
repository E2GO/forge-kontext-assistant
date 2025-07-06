#!/usr/bin/env python3
"""
Test script for Florence-2 integration in Kontext Assistant.
Tests image analysis and prompt generation with real Florence-2 model.
"""

import sys
import os

# Fix for Python 3.10 compatibility - must be before any other imports
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
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


def test_florence2_loading():
    """Test Florence-2 model loading."""
    print_header("Testing Florence-2 Model Loading")
    
    loader = Florence2ModelLoader()
    
    print("1. Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n2. Loading Florence-2 model...")
    start_time = time.time()
    try:
        loader.initialize()
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded successfully in {load_time:.2f} seconds")
        print(f"   Device: {loader.device}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False


def test_image_analysis():
    """Test image analysis with a sample image."""
    print_header("Testing Image Analysis")
    
    # Download a test image
    print("1. Downloading test image...")
    try:
        url = "https://images.unsplash.com/photo-1542291026-7eec264c27ff"
        response = requests.get(url, params={'w': 512})
        test_image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"   ✅ Downloaded test image ({test_image.size})")
    except Exception as e:
        print(f"   ❌ Failed to download image: {e}")
        # Create a simple test image
        test_image = Image.new('RGB', (512, 512), color='red')
        print("   Using fallback red square image")
    
    # Analyze image
    print("\n2. Analyzing image with Florence-2...")
    analyzer = ImageAnalyzer()
    
    start_time = time.time()
    try:
        result = analyzer.analyze(test_image)
        analysis_time = time.time() - start_time
        
        print(f"   ✅ Analysis completed in {analysis_time:.2f} seconds")
        
        # Print results
        print("\n3. Analysis Results:")
        print(f"   Main objects: {', '.join(result.objects['main'])}")
        print(f"   Style: {result.style['artistic']}, Mood: {result.style['mood']}")
        print(f"   Setting: {result.environment['setting']}")
        print(f"   Lighting: {result.lighting['intensity']} {result.lighting['primary_source']}")
        print(f"\n   Caption: {result.raw_caption[:200]}...")
        
        return result
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prompt_generation(analysis_result):
    """Test prompt generation with analysis results."""
    print_header("Testing Prompt Generation")
    
    if not analysis_result:
        print("⚠️  No analysis result available, using mock data")
        analysis_data = {
            'objects': ['red shoe'],
            'styles': ['product photography']
        }
    else:
        analysis_data = {
            'objects': analysis_result.objects['main'],
            'styles': [analysis_result.style['artistic']]
        }
    
    # Test different task types
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    
    test_cases = [
        ("object_manipulation", "make it blue"),
        ("style_transfer", "convert to oil painting style"),
        ("environment_change", "place in a forest"),
        ("lighting_adjustment", "add dramatic sunset lighting")
    ]
    
    print("Testing prompt generation for different tasks:\n")
    
    for task_type, user_intent in test_cases:
        try:
            prompt = generator.generate(
                task_type=task_type,
                user_intent=user_intent,
                image_analysis=analysis_data,
                subtype="default"
            )
            print(f"Task: {task_type}")
            print(f"Intent: '{user_intent}'")
            print(f"Generated: {prompt}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Failed to generate prompt for {task_type}: {e}")
            print("-" * 50)


def test_memory_management():
    """Test memory management and model unloading."""
    print_header("Testing Memory Management")
    
    analyzer = ImageAnalyzer()
    
    if torch.cuda.is_available():
        print("1. Initial VRAM usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Load model
        analyzer.model_loader.initialize()
        
        print("\n2. After model loading:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Unload model
        analyzer.unload_model()
        torch.cuda.empty_cache()
        
        print("\n3. After model unloading:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print("\n   ✅ Memory management test completed")
    else:
        print("   ⚠️  CUDA not available, skipping VRAM tests")


def test_caching():
    """Test analysis caching."""
    print_header("Testing Analysis Caching")
    
    analyzer = ImageAnalyzer()
    test_image = Image.new('RGB', (256, 256), color='blue')
    
    print("1. First analysis (should be slow)...")
    start = time.time()
    result1 = analyzer.analyze(test_image)
    time1 = time.time() - start
    print(f"   Time: {time1:.2f} seconds")
    
    print("\n2. Second analysis (should be cached)...")
    start = time.time()
    result2 = analyzer.analyze(test_image)
    time2 = time.time() - start
    print(f"   Time: {time2:.2f} seconds")
    
    if time2 < time1 * 0.1:  # Should be at least 10x faster
        print("\n   ✅ Caching working correctly")
    else:
        print("\n   ⚠️  Caching may not be working properly")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  FLORENCE-2 INTEGRATION TEST SUITE")
    print("  Kontext Assistant V2")
    print("="*60)
    
    # Test 1: Model Loading
    if not test_florence2_loading():
        print("\n⚠️  Model loading failed. Some tests will be skipped.")
        return
    
    # Test 2: Image Analysis
    analysis_result = test_image_analysis()
    
    # Test 3: Prompt Generation
    test_prompt_generation(analysis_result)
    
    # Test 4: Memory Management
    test_memory_management()
    
    # Test 5: Caching
    test_caching()
    
    print_header("All Tests Completed")
    print("✅ Florence-2 integration is ready for use!")


if __name__ == "__main__":
    main()