#!/usr/bin/env python3
"""
Test script for JoyCaption analyzer to verify pad_token_id handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import torch

def test_joycaption_initialization():
    """Test that JoyCaption analyzer initializes correctly with proper tokenizer setup"""
    try:
        from ka_modules.joycaption_analyzer import JoyCaptionAnalyzer
        
        # Initialize with CPU to avoid GPU memory issues during testing
        analyzer = JoyCaptionAnalyzer(model_version='beta-one', force_cpu=True)
        
        # Force initialization
        analyzer._ensure_initialized()
        
        # Check tokenizer configuration
        assert analyzer.processor is not None, "Processor not loaded"
        assert analyzer.processor.tokenizer.pad_token is not None, "pad_token not set"
        assert analyzer.processor.tokenizer.pad_token_id is not None, "pad_token_id not set"
        
        print("✅ JoyCaption initialization successful")
        print(f"   - pad_token: {analyzer.processor.tokenizer.pad_token}")
        print(f"   - pad_token_id: {analyzer.processor.tokenizer.pad_token_id}")
        print(f"   - eos_token_id: {analyzer.processor.tokenizer.eos_token_id}")
        print(f"   - padding_side: {analyzer.processor.tokenizer.padding_side}")
        
        return True
        
    except Exception as e:
        print(f"❌ JoyCaption initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_joycaption_inference():
    """Test that JoyCaption can perform inference without pad_token_id errors"""
    try:
        from ka_modules.joycaption_analyzer import JoyCaptionAnalyzer
        
        # Create a simple test image
        test_image = Image.new('RGB', (256, 256), color='red')
        
        # Initialize analyzer
        analyzer = JoyCaptionAnalyzer(model_version='beta-one', force_cpu=True)
        
        # Test basic generation
        result = analyzer._generate(
            image=test_image,
            prompt="Write a simple description of this image."
        )
        
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
        print("✅ JoyCaption inference successful")
        print(f"   Generated caption: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ JoyCaption inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_joycaption_full_analysis():
    """Test full analysis pipeline"""
    try:
        from ka_modules.joycaption_analyzer import JoyCaptionAnalyzer
        
        # Create a test image with more detail
        test_image = Image.new('RGB', (512, 512), color='blue')
        
        # Initialize analyzer
        analyzer = JoyCaptionAnalyzer(model_version='beta-one', force_cpu=True)
        
        # Run full analysis
        results = analyzer.analyze(
            image=test_image,
            modes=['descriptive']  # Just test one mode to save time
        )
        
        assert results['success'] == True, "Analysis should succeed"
        assert 'descriptive' in results, "Should have descriptive result"
        
        print("✅ JoyCaption full analysis successful")
        print(f"   Analysis time: {results.get('analysis_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ JoyCaption full analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing JoyCaption analyzer pad_token_id handling...")
    print("=" * 60)
    
    # Check if required libraries are available
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ transformers library not installed")
        sys.exit(1)
    
    tests = [
        test_joycaption_initialization,
        # Uncomment these to run full tests (requires model download)
        # test_joycaption_inference,
        # test_joycaption_full_analysis,
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed < total:
        sys.exit(1)