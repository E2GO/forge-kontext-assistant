#!/usr/bin/env python3
"""
Test GGUF integration for JoyCaption
Tests automatic download and model inference
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from PIL import Image
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gguf_direct():
    """Test GGUF implementation directly"""
    logger.info("Testing JoyCaption GGUF direct implementation...")
    
    try:
        from ka_modules.joycaption_gguf import JoyCaptionGGUF
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
        
        # Initialize GGUF analyzer
        analyzer = JoyCaptionGGUF(quantization='Q6_K')
        
        # Test analysis
        result = analyzer.analyze(test_image, mode='booru_tags_medium')
        
        logger.info(f"GGUF Direct Test Result: {result}")
        
        # Check results
        assert result['success'], "Analysis failed"
        assert 'booru_tags_medium' in result or 'danbooru_tags' in result, "No tags generated"
        
        # Unload model
        analyzer.unload_model()
        
        logger.info("✅ GGUF direct test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ GGUF direct test failed: {e}")
        return False

def test_joycaption_analyzer_gguf():
    """Test JoyCaptionAnalyzer with GGUF enabled"""
    logger.info("Testing JoyCaptionAnalyzer with GGUF...")
    
    try:
        from ka_modules.joycaption_analyzer import JoyCaptionAnalyzer
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color=(200, 100, 50))
        
        # Initialize with GGUF
        analyzer = JoyCaptionAnalyzer(use_gguf=True)
        
        # Test analysis
        result = analyzer.analyze(test_image)
        
        logger.info(f"JoyCaptionAnalyzer GGUF Result: {result}")
        
        # Check results
        assert 'success' in result, "Result missing success field"
        assert result.get('success', False), "Analysis failed"
        
        # Unload model
        analyzer.unload_model()
        
        logger.info("✅ JoyCaptionAnalyzer GGUF test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ JoyCaptionAnalyzer GGUF test failed: {e}")
        return False

def test_smart_analyzer_gguf():
    """Test SmartAnalyzer with GGUF JoyCaption"""
    logger.info("Testing SmartAnalyzer with GGUF JoyCaption...")
    
    try:
        from ka_modules.smart_analyzer import SmartAnalyzer
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color=(50, 200, 100))
        
        # Initialize smart analyzer
        analyzer = SmartAnalyzer()
        
        # Test with JoyCaption only
        result = analyzer.analyze(
            test_image, 
            use_florence=False, 
            use_joycaption=True
        )
        
        logger.info(f"SmartAnalyzer GGUF Result: {result}")
        
        # Check results
        assert result['success'], "Analysis failed"
        assert 'description' in result, "No description generated"
        
        # Unload models
        analyzer.unload_models()
        
        logger.info("✅ SmartAnalyzer GGUF test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SmartAnalyzer GGUF test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting GGUF integration tests...")
    
    tests = [
        ("Direct GGUF", test_gguf_direct),
        ("JoyCaptionAnalyzer GGUF", test_joycaption_analyzer_gguf),
        ("SmartAnalyzer GGUF", test_smart_analyzer_gguf)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name}...")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary:")
    logger.info(f"{'='*60}")
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.info("\n⚠️ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)