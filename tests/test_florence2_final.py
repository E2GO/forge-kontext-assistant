#!/usr/bin/env python3
"""
Final test for Florence-2 after fixing device_map issue.
"""

import sys
import os

# Fix for Python 3.10
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from pathlib import Path
import torch
from PIL import Image
import time

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_direct_loading():
    """Test Florence-2 loading directly."""
    print("="*60)
    print("Testing Direct Florence-2 Loading")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        print("\n1. Loading model...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        if device == "cuda":
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        print("\n2. Loading processor...")
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True
        )
        print("✅ Processor loaded")
        
        print("\n3. Testing inference...")
        test_image = Image.new('RGB', (256, 256), color='red')
        
        inputs = processor(text="<CAPTION>", images=test_image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"✅ Generated caption: {generated_text}")
        
        # Cleanup
        del model
        del processor
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ka_module():
    """Test loading through ka_modules."""
    print("\n" + "="*60)
    print("Testing KA Module Loading")
    print("="*60)
    
    try:
        # Try importing from ka_modules
        try:
            from ka_modules.image_analyzer import ImageAnalyzer, Florence2ModelLoader
            print("✅ Imported from ka_modules successfully")
        except ImportError:
            # Try direct import if file is in root
            from image_analyzer import ImageAnalyzer, Florence2ModelLoader
            print("⚠️  Imported from root directory (file should be in ka_modules/)")
        
        print("\n1. Creating analyzer...")
        analyzer = ImageAnalyzer()
        print("✅ Analyzer created")
        
        print("\n2. Testing model loader...")
        loader = Florence2ModelLoader()
        
        start_time = time.time()
        loader.initialize()
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded via analyzer in {load_time:.2f} seconds")
        
        print("\n3. Testing analysis...")
        test_image = Image.new('RGB', (512, 512))
        # Draw some colored rectangles
        pixels = test_image.load()
        for i in range(100, 200):
            for j in range(100, 200):
                pixels[i, j] = (255, 0, 0)  # Red square
        for i in range(300, 400):
            for j in range(300, 400):
                pixels[i, j] = (0, 255, 0)  # Green square
        
        start_time = time.time()
        result = analyzer.analyze(test_image, use_cache=False)
        analysis_time = time.time() - start_time
        
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        print(f"\nAnalysis Results:")
        print(f"  Caption: {result.raw_caption[:100]}...")
        print(f"  Objects: {result.objects}")
        print(f"  Style: {result.style['artistic']}")
        
        # Test cleanup
        print("\n4. Testing cleanup...")
        analyzer.unload_model()
        print("✅ Model unloaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Florence-2 Integration Final Test")
    print("="*60)
    
    # System info
    print("System Information:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run tests
    print("\n" + "="*60)
    
    direct_success = test_direct_loading()
    ka_success = test_ka_module()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Direct Loading: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"KA Module: {'✅ PASS' if ka_success else '❌ FAIL'}")
    
    if direct_success and ka_success:
        print("\n🎉 All tests passed! Florence-2 is working correctly.")
        print("\nYou can now:")
        print("1. Start Forge WebUI")
        print("2. Use the Kontext Smart Assistant")
        print("3. Analyze images and generate prompts!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()