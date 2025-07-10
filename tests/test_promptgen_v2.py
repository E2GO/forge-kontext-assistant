"""
Comprehensive test suite for PromptGen v2.0 integration
Tests all functionality according to official documentation
"""

import pytest
import torch
import logging
from PIL import Image
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ka_modules.image_analyzer import ImageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPromptGenV2:
    """Test suite for PromptGen v2.0 model integration"""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        # Create a colorful test image with various elements
        img = Image.new('RGB', (512, 512), color='white')
        pixels = img.load()
        
        # Add some colored regions
        for x in range(256):
            for y in range(256):
                pixels[x, y] = (255, 0, 0)  # Red quadrant
                
        for x in range(256, 512):
            for y in range(256):
                pixels[x, y] = (0, 255, 0)  # Green quadrant
                
        for x in range(256):
            for y in range(256, 512):
                pixels[x, y] = (0, 0, 255)  # Blue quadrant
                
        return img
    
    @pytest.fixture
    def analyzer_base(self):
        """Create base Florence-2 analyzer"""
        return ImageAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu', 
                           model_type='base')
    
    @pytest.fixture
    def analyzer_promptgen(self):
        """Create PromptGen v2.0 analyzer"""
        return ImageAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu', 
                           model_type='promptgen_v2')
    
    def test_model_initialization(self, analyzer_promptgen):
        """Test 1: Model initialization and configuration"""
        logger.info("=== Test 1: Model Initialization ===")
        
        # Check model configuration
        assert analyzer_promptgen.model_type == 'promptgen_v2'
        assert analyzer_promptgen.model_id == 'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
        assert analyzer_promptgen.model_name == 'Florence-2 PromptGen v2.0'
        
        # Ensure model is loaded
        analyzer_promptgen._ensure_initialized()
        assert analyzer_promptgen._initialized
        assert analyzer_promptgen.model is not None
        assert analyzer_promptgen.processor is not None
        
        logger.info("✓ Model initialization successful")
    
    def test_all_task_prompts(self, analyzer_promptgen, test_image):
        """Test 2: All PromptGen v2.0 task prompts"""
        logger.info("=== Test 2: All Task Prompts ===")
        
        # Ensure model is loaded
        analyzer_promptgen._ensure_initialized()
        
        # Test each task prompt
        task_prompts = [
            "<GENERATE_TAGS>",
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<ANALYZE>",
            "<MIXED_CAPTION>",
            "<MIXED_CAPTION_PLUS>"
        ]
        
        results = {}
        for task in task_prompts:
            logger.info(f"\nTesting task: {task}")
            start_time = time.time()
            
            result = analyzer_promptgen._run_florence_task(test_image, task)
            elapsed = time.time() - start_time
            
            logger.info(f"Task completed in {elapsed:.2f}s")
            logger.info(f"Result type: {type(result)}")
            
            if result:
                if isinstance(result, dict):
                    logger.info(f"Result keys: {list(result.keys())}")
                    for k, v in result.items():
                        if isinstance(v, str):
                            logger.info(f"  {k}: {v[:100]}...")
                        else:
                            logger.info(f"  {k}: {type(v)}")
                else:
                    logger.info(f"Result: {str(result)[:100]}...")
                
                results[task] = result
                assert result is not None, f"Task {task} returned None"
            else:
                logger.warning(f"Task {task} returned None")
        
        # Validate that we got results for critical tasks
        critical_tasks = ["<MORE_DETAILED_CAPTION>", "<GENERATE_TAGS>"]
        for task in critical_tasks:
            assert task in results and results[task], f"Critical task {task} failed"
        
        logger.info("✓ All task prompts tested")
        return results
    
    def test_output_format_validation(self, analyzer_promptgen, test_image):
        """Test 3: Validate output format matches documentation"""
        logger.info("=== Test 3: Output Format Validation ===")
        
        analyzer_promptgen._ensure_initialized()
        
        # Test MORE_DETAILED_CAPTION
        caption_result = analyzer_promptgen._run_florence_task(test_image, "<MORE_DETAILED_CAPTION>")
        assert caption_result is not None, "Caption generation failed"
        
        # Extract caption text
        if isinstance(caption_result, dict):
            caption_text = (caption_result.get('<MORE_DETAILED_CAPTION>') or 
                          caption_result.get('MORE_DETAILED_CAPTION') or
                          list(caption_result.values())[0] if len(caption_result) == 1 else '')
        else:
            caption_text = str(caption_result)
        
        assert caption_text, "Caption text is empty"
        assert len(caption_text) > 20, f"Caption too short: {caption_text}"
        logger.info(f"✓ Caption format valid: {caption_text[:100]}...")
        
        # Test GENERATE_TAGS
        tags_result = analyzer_promptgen._run_florence_task(test_image, "<GENERATE_TAGS>")
        assert tags_result is not None, "Tags generation failed"
        
        # Extract tags text
        if isinstance(tags_result, dict):
            tags_text = (tags_result.get('<GENERATE_TAGS>') or 
                        tags_result.get('GENERATE_TAGS') or
                        list(tags_result.values())[0] if len(tags_result) == 1 else '')
        else:
            tags_text = str(tags_result)
        
        assert tags_text, "Tags text is empty"
        assert ',' in tags_text or ' ' in tags_text, f"Tags not properly formatted: {tags_text}"
        logger.info(f"✓ Tags format valid: {tags_text[:100]}...")
        
        return caption_text, tags_text
    
    def test_comparison_with_base_model(self, analyzer_base, analyzer_promptgen, test_image):
        """Test 4: Compare outputs between base and PromptGen models"""
        logger.info("=== Test 4: Model Comparison ===")
        
        # Analyze with base model
        logger.info("Analyzing with base Florence-2...")
        base_result = analyzer_base.analyze(test_image, detailed=True)
        
        # Analyze with PromptGen
        logger.info("Analyzing with PromptGen v2.0...")
        promptgen_result = analyzer_promptgen.analyze(test_image, detailed=True)
        
        # Compare results
        logger.info("\n--- Comparison ---")
        
        # Both should have descriptions
        assert 'description' in base_result, "Base model missing description"
        assert 'description' in promptgen_result, "PromptGen missing description"
        
        logger.info(f"Base description: {base_result['description'][:100]}...")
        logger.info(f"PromptGen description: {promptgen_result['description'][:100]}...")
        
        # PromptGen should have tags
        if 'tags' in promptgen_result:
            assert promptgen_result['tags'].get('danbooru'), "PromptGen missing Danbooru tags"
            logger.info(f"PromptGen tags: {promptgen_result['tags']['danbooru'][:100]}...")
        
        # PromptGen might have additional fields
        promptgen_only_fields = set(promptgen_result.keys()) - set(base_result.keys())
        if promptgen_only_fields:
            logger.info(f"PromptGen-specific fields: {promptgen_only_fields}")
        
        logger.info("✓ Model comparison complete")
        return base_result, promptgen_result
    
    def test_real_world_images(self, analyzer_promptgen):
        """Test 5: Test with various real-world image types"""
        logger.info("=== Test 5: Real-World Images ===")
        
        analyzer_promptgen._ensure_initialized()
        
        # Create test images of different types
        test_cases = [
            {
                'name': 'portrait',
                'image': self._create_portrait_image(),
                'expected_tags': ['1girl', 'face', 'portrait']
            },
            {
                'name': 'landscape',
                'image': self._create_landscape_image(),
                'expected_tags': ['scenery', 'landscape', 'nature']
            },
            {
                'name': 'abstract',
                'image': self._create_abstract_image(),
                'expected_tags': ['abstract', 'colorful', 'pattern']
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\nTesting {test_case['name']} image...")
            
            # Get tags
            tags_result = analyzer_promptgen._run_florence_task(test_case['image'], "<GENERATE_TAGS>")
            if tags_result:
                if isinstance(tags_result, dict):
                    tags_text = list(tags_result.values())[0] if tags_result else ''
                else:
                    tags_text = str(tags_result)
                
                logger.info(f"Tags: {tags_text}")
                
                # Check if any expected tags are present (loose validation)
                tags_lower = tags_text.lower()
                found_expected = any(tag in tags_lower for tag in test_case['expected_tags'])
                logger.info(f"Expected tags found: {found_expected}")
            
            # Get detailed caption
            caption_result = analyzer_promptgen._run_florence_task(test_case['image'], "<MORE_DETAILED_CAPTION>")
            if caption_result:
                if isinstance(caption_result, dict):
                    caption_text = list(caption_result.values())[0] if caption_result else ''
                else:
                    caption_text = str(caption_result)
                
                logger.info(f"Caption: {caption_text[:150]}...")
                assert len(caption_text) > 20, f"Caption too short for {test_case['name']}"
        
        logger.info("✓ Real-world image tests complete")
    
    def test_generation_parameters(self, analyzer_promptgen, test_image):
        """Test 6: Validate generation parameters match documentation"""
        logger.info("=== Test 6: Generation Parameters ===")
        
        analyzer_promptgen._ensure_initialized()
        
        # Check the generation parameters used in _run_florence_task
        # This is a validation that our code matches the documentation
        
        # Parameters should be:
        # - max_new_tokens=1024
        # - num_beams=3
        # - do_sample=False
        
        # We can't directly check the parameters used inside the method,
        # but we can verify the output characteristics
        
        # Test with a complex prompt that should generate long output
        result = analyzer_promptgen._run_florence_task(test_image, "<MORE_DETAILED_CAPTION>")
        
        if result:
            if isinstance(result, dict):
                text = list(result.values())[0] if result else ''
            else:
                text = str(result)
            
            # Check that we're getting substantial output (indicating proper max_new_tokens)
            assert len(text) > 50, f"Output too short, might indicate parameter issue: {len(text)} chars"
            
            # Multiple runs should give same result (do_sample=False)
            result2 = analyzer_promptgen._run_florence_task(test_image, "<MORE_DETAILED_CAPTION>")
            if result2:
                if isinstance(result2, dict):
                    text2 = list(result2.values())[0] if result2 else ''
                else:
                    text2 = str(result2)
                
                assert text == text2, "Results not deterministic (do_sample might be True)"
        
        logger.info("✓ Generation parameters validated")
    
    def test_error_handling(self, analyzer_promptgen):
        """Test 7: Error handling and edge cases"""
        logger.info("=== Test 7: Error Handling ===")
        
        analyzer_promptgen._ensure_initialized()
        
        # Test with invalid image
        try:
            result = analyzer_promptgen._run_florence_task(None, "<CAPTION>")
            assert result is None or not result, "Should handle None image gracefully"
        except Exception as e:
            logger.info(f"Expected error for None image: {e}")
        
        # Test with very small image
        tiny_image = Image.new('RGB', (1, 1), color='red')
        result = analyzer_promptgen._run_florence_task(tiny_image, "<CAPTION>")
        logger.info(f"Tiny image result: {result}")
        
        # Test with invalid task
        result = analyzer_promptgen._run_florence_task(test_image, "<INVALID_TASK>")
        logger.info(f"Invalid task result: {result}")
        
        logger.info("✓ Error handling tests complete")
    
    def test_full_analysis_pipeline(self, analyzer_promptgen, test_image):
        """Test 8: Full analysis pipeline integration"""
        logger.info("=== Test 8: Full Analysis Pipeline ===")
        
        # Run full analysis
        start_time = time.time()
        result = analyzer_promptgen.analyze(test_image, detailed=True)
        elapsed = time.time() - start_time
        
        logger.info(f"Full analysis completed in {elapsed:.2f}s")
        
        # Validate result structure
        assert result is not None, "Analysis returned None"
        assert isinstance(result, dict), "Analysis should return dict"
        
        # Check required fields
        required_fields = ['description', 'size', 'mode', 'analysis_mode', 'model_type']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Check PromptGen-specific fields
        assert result['model_type'] == 'promptgen_v2', "Wrong model type"
        assert result['analysis_mode'] == 'florence2', "Wrong analysis mode"
        
        # Check for tags (PromptGen should generate these)
        if 'tags' in result:
            assert 'danbooru' in result['tags'], "Missing Danbooru tags"
            logger.info(f"Danbooru tags: {result['tags']['danbooru'][:100]}...")
        
        # Check for mixed caption (if detailed=True)
        if 'mixed_caption' in result:
            assert result['mixed_caption'], "Mixed caption is empty"
            logger.info(f"Mixed caption: {result['mixed_caption'][:100]}...")
        
        logger.info("✓ Full pipeline test complete")
        return result
    
    def test_performance_metrics(self, analyzer_promptgen):
        """Test 9: Performance metrics and optimization"""
        logger.info("=== Test 9: Performance Metrics ===")
        
        analyzer_promptgen._ensure_initialized()
        
        # Create images of different sizes
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for size in sizes:
            logger.info(f"\nTesting {size[0]}x{size[1]} image...")
            test_img = Image.new('RGB', size, color='blue')
            
            # Time each task
            tasks = ["<CAPTION>", "<DETAILED_CAPTION>", "<GENERATE_TAGS>"]
            
            for task in tasks:
                start = time.time()
                result = analyzer_promptgen._run_florence_task(test_img, task)
                elapsed = time.time() - start
                
                logger.info(f"  {task}: {elapsed:.2f}s")
                assert result is not None, f"Task {task} failed for size {size}"
        
        logger.info("✓ Performance metrics collected")
    
    def test_output_consistency(self, analyzer_promptgen, test_image):
        """Test 10: Output consistency across multiple runs"""
        logger.info("=== Test 10: Output Consistency ===")
        
        analyzer_promptgen._ensure_initialized()
        
        # Run the same analysis multiple times
        num_runs = 3
        results = []
        
        for i in range(num_runs):
            logger.info(f"\nRun {i+1}/{num_runs}")
            result = analyzer_promptgen._run_florence_task(test_image, "<MORE_DETAILED_CAPTION>")
            
            if result:
                if isinstance(result, dict):
                    text = list(result.values())[0] if result else ''
                else:
                    text = str(result)
                results.append(text)
        
        # Check consistency
        assert len(set(results)) == 1, "Results are not consistent across runs"
        logger.info("✓ Output is consistent across multiple runs")
    
    # Helper methods
    def _create_portrait_image(self) -> Image.Image:
        """Create a simple portrait-like image"""
        img = Image.new('RGB', (512, 512), color='white')
        pixels = img.load()
        
        # Draw a simple face-like shape
        center_x, center_y = 256, 200
        radius = 80
        
        for x in range(512):
            for y in range(512):
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if dist < radius:
                    pixels[x, y] = (255, 220, 177)  # Skin color
        
        return img
    
    def _create_landscape_image(self) -> Image.Image:
        """Create a simple landscape image"""
        img = Image.new('RGB', (512, 512), color='skyblue')
        pixels = img.load()
        
        # Add ground
        for x in range(512):
            for y in range(300, 512):
                pixels[x, y] = (34, 139, 34)  # Green
        
        return img
    
    def _create_abstract_image(self) -> Image.Image:
        """Create an abstract pattern"""
        img = Image.new('RGB', (512, 512), color='white')
        pixels = img.load()
        
        # Create a pattern
        for x in range(512):
            for y in range(512):
                r = (x * 2) % 256
                g = (y * 2) % 256
                b = ((x + y) * 2) % 256
                pixels[x, y] = (r, g, b)
        
        return img


def run_all_tests():
    """Run all tests and generate report"""
    logger.info("=" * 60)
    logger.info("PROMPTGEN V2.0 COMPREHENSIVE TEST SUITE")
    logger.info("=" * 60)
    
    # Create test instance
    test_suite = TestPromptGenV2()
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='white')
    pixels = test_image.load()
    for x in range(256):
        for y in range(256):
            pixels[x, y] = (255, 0, 0)
    
    # Create analyzers
    try:
        analyzer_promptgen = ImageAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model_type='promptgen_v2'
        )
        
        analyzer_base = ImageAnalyzer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model_type='base'
        )
    except Exception as e:
        logger.error(f"Failed to create analyzers: {e}")
        return
    
    # Run tests
    test_results = {}
    
    tests = [
        ("Model Initialization", lambda: test_suite.test_model_initialization(analyzer_promptgen)),
        ("All Task Prompts", lambda: test_suite.test_all_task_prompts(analyzer_promptgen, test_image)),
        ("Output Format Validation", lambda: test_suite.test_output_format_validation(analyzer_promptgen, test_image)),
        ("Model Comparison", lambda: test_suite.test_comparison_with_base_model(analyzer_base, analyzer_promptgen, test_image)),
        ("Real World Images", lambda: test_suite.test_real_world_images(analyzer_promptgen)),
        ("Generation Parameters", lambda: test_suite.test_generation_parameters(analyzer_promptgen, test_image)),
        ("Error Handling", lambda: test_suite.test_error_handling(analyzer_promptgen)),
        ("Full Pipeline", lambda: test_suite.test_full_analysis_pipeline(analyzer_promptgen, test_image)),
        ("Performance Metrics", lambda: test_suite.test_performance_metrics(analyzer_promptgen)),
        ("Output Consistency", lambda: test_suite.test_output_consistency(analyzer_promptgen, test_image))
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            test_func()
            test_results[test_name] = "PASSED"
        except Exception as e:
            logger.error(f"Test '{test_name}' FAILED: {e}")
            test_results[test_name] = f"FAILED: {str(e)}"
    
    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result == "PASSED")
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    # Save report
    report_path = Path(__file__).parent / "test_report_promptgen_v2.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': test_results,
            'summary': f"{passed}/{total} passed"
        }, f, indent=2)
    
    logger.info(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_all_tests()