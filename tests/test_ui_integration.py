"""
UI Integration tests for PromptGen v2.0
Tests the complete flow from UI to model output
"""

import pytest
import logging
import time
from pathlib import Path
from PIL import Image
import torch
from typing import Dict, Any

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ka_modules.smart_analyzer import SmartAnalyzer
from ka_modules.image_analyzer import ImageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestUIIntegration:
    """Test UI integration with PromptGen v2.0"""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image similar to what UI would provide"""
        img = Image.new('RGB', (768, 512), color='white')
        # Add some visual elements
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a simple scene
        draw.rectangle([0, 300, 768, 512], fill=(34, 139, 34))  # Ground
        draw.ellipse([300, 200, 468, 368], fill=(255, 255, 0))  # Sun
        draw.rectangle([100, 250, 200, 400], fill=(139, 69, 19))  # Tree trunk
        draw.ellipse([50, 150, 250, 300], fill=(0, 128, 0))  # Tree top
        
        return img
    
    def test_ui_florence_model_switching(self):
        """Test 1: UI Florence model switching functionality"""
        logger.info("=== Test 1: UI Model Switching ===")
        
        # Simulate UI model switching
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='base'
        )
        
        # Create test image
        test_img = Image.new('RGB', (512, 512), color='blue')
        
        # Analyze with base model
        logger.info("Analyzing with base model...")
        result1 = smart_analyzer.analyze(test_img, use_florence=True, use_joycaption=False)
        
        # Switch to PromptGen
        logger.info("Switching to PromptGen v2.0...")
        smart_analyzer.florence_model_type = 'promptgen_v2'
        smart_analyzer.florence = None  # Force reload
        
        # Analyze with PromptGen
        logger.info("Analyzing with PromptGen v2.0...")
        result2 = smart_analyzer.analyze(test_img, use_florence=True, use_joycaption=False)
        
        # Validate results
        assert result1 is not None, "Base model analysis failed"
        assert result2 is not None, "PromptGen analysis failed"
        
        # Check that results are different
        assert result1.get('florence_analysis') != result2.get('florence_analysis'), \
            "Results should be different between models"
        
        logger.info("✓ Model switching works correctly")
        
        # Clean up
        smart_analyzer.unload_models()
    
    def test_ui_output_format(self, test_image):
        """Test 2: UI output format for display"""
        logger.info("=== Test 2: UI Output Format ===")
        
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='promptgen_v2'
        )
        
        # Run analysis as UI would
        result = smart_analyzer.analyze(test_image, use_florence=True, use_joycaption=False)
        
        # Check UI-required fields
        assert 'description' in result, "Missing description for UI"
        assert 'success' in result, "Missing success flag for UI"
        
        # Check PromptGen-specific UI fields
        if 'tags' in result:
            assert 'danbooru' in result['tags'], "Missing Danbooru tags for UI display"
            
            # Validate tag format for UI
            tags_text = result['tags']['danbooru']
            assert isinstance(tags_text, str), "Tags should be string for UI"
            assert len(tags_text) > 0, "Tags should not be empty"
            
            logger.info(f"✓ Tags format valid for UI: {tags_text[:100]}...")
        
        # Check mixed caption for UI
        if 'florence_analysis' in result and 'mixed_caption' in result['florence_analysis']:
            mixed = result['florence_analysis']['mixed_caption']
            assert isinstance(mixed, str), "Mixed caption should be string for UI"
            logger.info(f"✓ Mixed caption format valid: {mixed[:100]}...")
        
        logger.info("✓ UI output format validated")
        
        # Clean up
        smart_analyzer.unload_models()
    
    def test_ui_performance_requirements(self, test_image):
        """Test 3: Performance requirements for UI responsiveness"""
        logger.info("=== Test 3: UI Performance Requirements ===")
        
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='promptgen_v2'
        )
        
        # First run (includes model loading)
        start = time.time()
        result1 = smart_analyzer.analyze(test_image, use_florence=True, use_joycaption=False)
        first_run_time = time.time() - start
        
        logger.info(f"First run (with loading): {first_run_time:.2f}s")
        
        # Second run (model already loaded)
        start = time.time()
        result2 = smart_analyzer.analyze(test_image, use_florence=True, use_joycaption=False)
        second_run_time = time.time() - start
        
        logger.info(f"Second run (cached): {second_run_time:.2f}s")
        
        # UI expects subsequent runs to be faster
        assert second_run_time < first_run_time, "Caching not working properly"
        
        # Check if analysis time is reasonable for UI
        if 'total_analysis_time' in result2:
            assert result2['total_analysis_time'] < 30, "Analysis too slow for UI"
        
        logger.info("✓ Performance meets UI requirements")
        
        # Clean up
        smart_analyzer.unload_models()
    
    def test_ui_error_handling(self):
        """Test 4: UI error handling scenarios"""
        logger.info("=== Test 4: UI Error Handling ===")
        
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='promptgen_v2'
        )
        
        # Test with invalid image
        result = smart_analyzer.analyze(None, use_florence=True, use_joycaption=False)
        assert not result.get('success', True), "Should fail gracefully with None image"
        assert 'error' in result, "Should provide error message for UI"
        logger.info(f"✓ Handled None image: {result.get('error', 'No error message')}")
        
        # Test with no models selected
        test_img = Image.new('RGB', (256, 256), color='red')
        result = smart_analyzer.analyze(test_img, use_florence=False, use_joycaption=False)
        assert 'error' in result, "Should error when no models selected"
        logger.info(f"✓ Handled no models: {result.get('error', 'No error message')}")
        
        # Clean up
        smart_analyzer.unload_models()
    
    def test_ui_prompt_template_integration(self, test_image):
        """Test 5: Integration with prompt templates"""
        logger.info("=== Test 5: Prompt Template Integration ===")
        
        from ka_modules.prompt_generator import PromptGenerator
        from ka_modules.templates import PromptTemplates
        
        # Initialize components
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='promptgen_v2'
        )
        
        templates = PromptTemplates()
        generator = PromptGenerator(templates)
        
        # Analyze image
        analysis = smart_analyzer.analyze(test_image, use_florence=True, use_joycaption=False)
        
        # Generate prompts using analysis
        if analysis.get('success'):
            # Test with different styles
            styles = ['photographic', 'artistic', 'anime']
            
            for style in styles:
                prompt = generator.generate_prompt(
                    analysis=analysis,
                    style=style,
                    detail_level='high'
                )
                
                assert prompt, f"Failed to generate {style} prompt"
                assert len(prompt) > 50, f"{style} prompt too short"
                
                # Check if PromptGen tags are incorporated
                if 'tags' in analysis and 'danbooru' in analysis['tags']:
                    # Some tags should appear in the prompt
                    tags = analysis['tags']['danbooru'].split(',')
                    tags_in_prompt = sum(1 for tag in tags[:5] if tag.strip() in prompt)
                    logger.info(f"✓ {style} prompt uses {tags_in_prompt} tags")
        
        logger.info("✓ Prompt template integration works")
        
        # Clean up
        smart_analyzer.unload_models()
    
    def test_ui_batch_processing(self):
        """Test 6: Batch processing simulation"""
        logger.info("=== Test 6: Batch Processing ===")
        
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='promptgen_v2'
        )
        
        # Create multiple test images
        test_images = [
            Image.new('RGB', (512, 512), color='red'),
            Image.new('RGB', (512, 512), color='green'),
            Image.new('RGB', (512, 512), color='blue')
        ]
        
        results = []
        start_time = time.time()
        
        # Process batch
        for i, img in enumerate(test_images):
            logger.info(f"Processing image {i+1}/{len(test_images)}")
            result = smart_analyzer.analyze(img, use_florence=True, use_joycaption=False)
            results.append(result)
            
            # Check each result
            assert result.get('success'), f"Image {i+1} failed"
            assert 'description' in result, f"Image {i+1} missing description"
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_images)
        
        logger.info(f"✓ Batch processing: {avg_time:.2f}s per image")
        
        # Clean up
        smart_analyzer.unload_models()
    
    def test_ui_model_unloading(self):
        """Test 7: Model unloading for memory management"""
        logger.info("=== Test 7: Model Unloading ===")
        
        smart_analyzer = SmartAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            florence_model_type='promptgen_v2'
        )
        
        # Load and analyze
        test_img = Image.new('RGB', (512, 512), color='yellow')
        result = smart_analyzer.analyze(test_img, use_florence=True, use_joycaption=False)
        assert result.get('success'), "Analysis failed"
        
        # Check models are loaded
        assert smart_analyzer.florence is not None, "Florence not loaded"
        
        # Unload models
        logger.info("Unloading models...")
        smart_analyzer.unload_models()
        
        # Check models are unloaded
        assert smart_analyzer.florence is None, "Florence not unloaded"
        
        # Try to analyze again (should reload)
        logger.info("Analyzing after unload...")
        result2 = smart_analyzer.analyze(test_img, use_florence=True, use_joycaption=False)
        assert result2.get('success'), "Re-analysis failed"
        
        logger.info("✓ Model unloading and reloading works")
        
        # Final cleanup
        smart_analyzer.unload_models()


def run_ui_integration_tests():
    """Run all UI integration tests"""
    logger.info("=" * 60)
    logger.info("UI INTEGRATION TEST SUITE")
    logger.info("=" * 60)
    
    test_suite = TestUIIntegration()
    test_results = {}
    
    # Create test image
    test_image = test_suite.test_image()
    
    tests = [
        ("Model Switching", test_suite.test_ui_florence_model_switching),
        ("Output Format", lambda: test_suite.test_ui_output_format(test_image)),
        ("Performance", lambda: test_suite.test_ui_performance_requirements(test_image)),
        ("Error Handling", test_suite.test_ui_error_handling),
        ("Template Integration", lambda: test_suite.test_ui_prompt_template_integration(test_image)),
        ("Batch Processing", test_suite.test_ui_batch_processing),
        ("Model Unloading", test_suite.test_ui_model_unloading)
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            test_func()
            test_results[test_name] = "PASSED"
        except Exception as e:
            logger.error(f"Test '{test_name}' FAILED: {e}")
            test_results[test_name] = f"FAILED: {str(e)}"
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("UI INTEGRATION TEST RESULTS")
    logger.info("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result == "PASSED")
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    run_ui_integration_tests()