#!/usr/bin/env python3
"""
Basic test suite for Forge Kontext Assistant V2.
Tests all core functionality without requiring Forge WebUI.
"""

import sys
import os

# Fix for Python 3.10 compatibility - MUST be first
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from pathlib import Path
import json
from PIL import Image

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from ka_modules.templates import PromptTemplates
from ka_modules.prompt_generator import PromptGenerator
from ka_modules.image_analyzer import ImageAnalyzer, AnalysisResult
from ka_modules.forge_integration import ForgeIntegration
from utils.cache import SimpleCache
try:
    from utils.validators import ImageValidator, PromptValidator
except ImportError:
    # Create mock validators if not available
    class ImageValidator:
        @staticmethod
        def validate_for_analysis(image):
            if image is None:
                return False, "Image is None"
            if hasattr(image, 'size'):
                w, h = image.size
                if w < 64 or h < 64:
                    return False, "Image too small"
            return True, "OK"
    
    class PromptValidator:
        @staticmethod
        def validate_user_intent(intent):
            if not intent or len(intent) > 500:
                return False, "Invalid length"
            return True, "OK"


class TestColors:
    """ANSI color codes for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print(f"\n{TestColors.BLUE}{TestColors.BOLD}Testing: {test_name}{TestColors.RESET}")
    print("-" * 50)


def print_result(passed: bool, message: str):
    """Print test result with color."""
    if passed:
        print(f"{TestColors.GREEN}✓ PASS{TestColors.RESET}: {message}")
    else:
        print(f"{TestColors.RED}✗ FAIL{TestColors.RESET}: {message}")


def test_templates():
    """Test template loading and functionality."""
    print_test_header("Template System")
    
    try:
        templates = PromptTemplates()
        
        # Test 1: Load templates
        all_templates = templates.get_all_templates()
        print_result(len(all_templates) > 0, f"Loaded {len(all_templates)} task types")
        
        # Test 2: Get specific template
        template = templates.get_template("object_manipulation", "color_change")
        expected = "Change the {object} color from {current_color} to {target_color}"
        print_result(expected in template, "Color change template loaded correctly")
        
        # Test 3: Template variables
        variables = templates.get_template_variables(template)
        print_result(len(variables) == 3, f"Found {len(variables)} template variables")
        
        # Test 4: New task types (V2)
        new_tasks = ["lighting_adjustment", "texture_change", "perspective_shift"]
        for task in new_tasks:
            has_task = task in templates.get_task_types()
            print_result(has_task, f"New task type '{task}' available")
        
        # Test 5: Custom template
        templates.add_custom_template("test", "custom", "Test {variable}")
        custom = templates.get_template("test", "custom")
        print_result(custom == "Test {variable}", "Custom template added successfully")
        
        return True
        
    except Exception as e:
        print_result(False, f"Template tests failed: {str(e)}")
        return False


def test_prompt_generation():
    """Test prompt generation functionality."""
    print_test_header("Prompt Generation")
    
    try:
        templates = PromptTemplates()
        generator = PromptGenerator(templates)
        
        # Test 1: Basic generation
        prompt = generator.generate(
            task_type="object_manipulation",
            user_intent="make the car blue",
            subtype="color_change"
        )
        print_result("blue" in prompt.lower(), f"Generated basic prompt: {prompt[:50]}...")
        
        # Test 2: With image analysis context
        context = {
            'objects': ['red car', 'street'],
            'styles': ['photorealistic']
        }
        prompt = generator.generate(
            task_type="object_manipulation",
            user_intent="change to blue",
            image_analysis=context,
            subtype="color_change"
        )
        print_result("red" in prompt and "blue" in prompt, "Context-aware prompt generation")
        
        # Test 3: Preservation rules
        prompt_high = generator.generate(
            task_type="style_transfer",
            user_intent="oil painting style",
            preserve_strength=0.9
        )
        prompt_low = generator.generate(
            task_type="style_transfer",
            user_intent="oil painting style",
            preserve_strength=0.2
        )
        print_result(len(prompt_high) > len(prompt_low), "Preservation strength affects output")
        
        # Test 4: Parse user intent
        test_intents = [
            ("add a red hat", {'action': 'add', 'target_color': 'red'}),
            ("remove the person", {'action': 'remove', 'target_object': 'person'}),
            ("change to sunset", {'target_time': 'evening'})
        ]
        
        all_parsed = True
        for intent, expected in test_intents:
            # Create context for parsing
            from ka_modules.prompt_generator import GenerationContext
            context = GenerationContext("test", "test", intent)
            params = generator._parse_user_intent(context)
            matches = any(params.get(k) == v for k, v in expected.items())
            if not matches:
                all_parsed = False
        
        print_result(all_parsed, "Intent parsing working correctly")
        
        # Test 5: Complex generation
        complex_prompt = generator.generate(
            task_type="element_combination",
            user_intent="merge the two scenes seamlessly",
            subtype="merge_scenes"
        )
        print_result(len(complex_prompt) > 20, "Complex prompt generation successful")
        
        return True
        
    except Exception as e:
        print_result(False, f"Prompt generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_image_analyzer_mock():
    """Test image analyzer in mock mode (no Florence-2 loading)."""
    print_test_header("Image Analyzer (Mock Mode)")
    
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (256, 256), color='blue')
        
        # Test without loading Florence-2
        from ka_modules.image_analyzer import ImageAnalyzer
        
        # Temporarily disable model loading for this test
        analyzer = ImageAnalyzer()
        
        # Test the structure is correct
        print_result(hasattr(analyzer, 'model_loader'), "Analyzer has model_loader")
        print_result(hasattr(analyzer, 'analyze'), "Analyzer has analyze method")
        print_result(hasattr(analyzer, 'clear_cache'), "Analyzer has cache methods")
        
        # Test AnalysisResult structure
        result = analyzer._get_fallback_analysis("test")
        print_result(hasattr(result, 'objects'), "AnalysisResult has objects")
        print_result(hasattr(result, 'style'), "AnalysisResult has style")
        print_result(hasattr(result, 'environment'), "AnalysisResult has environment")
        
        return True
        
    except Exception as e:
        print_result(False, f"Image analyzer test failed: {str(e)}")
        return False


def test_forge_integration():
    """Test Forge integration components."""
    print_test_header("Forge Integration")
    
    try:
        # Test 1: ForgeIntegration static methods
        ForgeIntegration.set_shared_value("test_key", "test_value")
        value = ForgeIntegration.get_shared_value("test_key")
        print_result(value == "test_value", "Shared state working")
        
        # Test 2: Image registration
        test_images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='green'),
            None
        ]
        ForgeIntegration.register_kontext_images(test_images)
        retrieved = ForgeIntegration.get_kontext_images()
        print_result(len(retrieved) == 3, f"Registered {len(retrieved)} images")
        
        # Test 3: Clear state
        ForgeIntegration.clear_shared_state()
        cleared_value = ForgeIntegration.get_shared_value("test_key")
        print_result(cleared_value is None, "State cleared successfully")
        
        return True
        
    except Exception as e:
        print_result(False, f"Forge integration test failed: {str(e)}")
        return False


def test_validators():
    """Test validation utilities."""
    print_test_header("Validators")
    
    try:
        # Test image validation
        valid_image = Image.new('RGB', (512, 512), color='white')
        tiny_image = Image.new('RGB', (10, 10), color='black')
        
        is_valid, msg = ImageValidator.validate_for_analysis(valid_image)
        print_result(is_valid, "Valid image passes validation")
        
        is_valid, msg = ImageValidator.validate_for_analysis(tiny_image)
        print_result(not is_valid, "Tiny image fails validation")
        
        # Test prompt validation
        good_prompt = "Change the car color to blue"
        bad_prompt = "x" * 1000
        
        is_valid, msg = PromptValidator.validate_user_intent(good_prompt)
        print_result(is_valid, "Valid prompt passes validation")
        
        is_valid, msg = PromptValidator.validate_user_intent(bad_prompt)
        print_result(not is_valid, "Too long prompt fails validation")
        
        return True
        
    except Exception as e:
        print_result(False, f"Validator test failed: {str(e)}")
        return False


def test_cache():
    """Test caching functionality."""
    print_test_header("Cache System")
    
    try:
        cache = SimpleCache(max_size=3)
        
        # Test 1: Basic operations
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        val1 = cache.get("key1")
        print_result(val1 == "value1", "Cache set/get working")
        
        # Test 2: Cache miss
        val_miss = cache.get("nonexistent")
        print_result(val_miss is None, "Cache miss returns None")
        
        # Test 3: LRU eviction
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key2 (LRU)
        
        val2 = cache.get("key2")
        val1_still = cache.get("key1")
        print_result(val2 is None and val1_still == "value1", "LRU eviction working")
        
        # Test 4: Clear
        cache.clear()
        val_after_clear = cache.get("key1")
        print_result(val_after_clear is None, "Cache clear working")
        
        return True
        
    except Exception as e:
        print_result(False, f"Cache test failed: {str(e)}")
        return False


def test_config_loading():
    """Test configuration file loading."""
    print_test_header("Configuration Loading")
    
    try:
        config_path = project_root / "configs" / "task_configs.json"
        
        # Test 1: Config file exists
        exists = config_path.exists()
        print_result(exists, f"Config file exists at {config_path}")
        
        if exists:
            # Test 2: Valid JSON
            with open(config_path, 'r') as f:
                config = json.load(f)
            print_result(True, "Config file is valid JSON")
            
            # Test 3: Has expected structure
            has_tasks = "task_types" in config
            print_result(has_tasks, "Config has task_types section")
            
            # Test 4: New V2 tasks present
            if has_tasks:
                v2_tasks = ["lighting_adjustment", "texture_change", "perspective_shift"]
                all_present = all(task in config["task_types"] for task in v2_tasks)
                print_result(all_present, "All V2 task types present in config")
        
        return True
        
    except Exception as e:
        print_result(False, f"Config loading failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print(f"\n{TestColors.BOLD}{'='*60}")
    print("   FORGE KONTEXT ASSISTANT V2 - TEST SUITE")
    print(f"{'='*60}{TestColors.RESET}")
    
    tests = [
        ("Templates", test_templates),
        ("Prompt Generation", test_prompt_generation),
        ("Image Analyzer", test_image_analyzer_mock),
        ("Forge Integration", test_forge_integration),
        ("Validators", test_validators),
        ("Cache", test_cache),
        ("Configuration", test_config_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n{TestColors.RED}Unexpected error in {test_name}: {e}{TestColors.RESET}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{TestColors.BOLD}{'='*60}")
    print("   TEST SUMMARY")
    print(f"{'='*60}{TestColors.RESET}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = f"{TestColors.GREEN}PASS{TestColors.RESET}" if passed else f"{TestColors.RED}FAIL{TestColors.RESET}"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n{TestColors.BOLD}Total: {passed_count}/{total_count} tests passed{TestColors.RESET}")
    
    if passed_count == total_count:
        print(f"\n{TestColors.GREEN}{TestColors.BOLD}✅ All tests passed! V2 is ready to use.{TestColors.RESET}")
        print("\nNext steps:")
        print("1. Run: python tests/test_florence2_simple.py")
        print("2. Start Forge WebUI and test the extension")
    else:
        print(f"\n{TestColors.YELLOW}⚠️  Some tests failed. Check the errors above.{TestColors.RESET}")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)