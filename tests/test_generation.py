"""
Test script to verify prompt generation works
"""

# Fix collections compatibility BEFORE any other imports
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ka_modules.templates import PromptTemplates
from ka_modules.prompt_generator import PromptGenerator
from ka_modules.image_analyzer import ImageAnalyzer
from PIL import Image

def test_prompt_generation():
    """Test that prompt generation works without errors"""
    
    print("=== Testing Prompt Generation ===\n")
    
    # Initialize components
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    analyzer = ImageAnalyzer()
    
    # Test cases
    test_cases = [
        {
            "task_type": "object_manipulation",
            "user_intent": "make the car blue",
            "expected_contains": ["blue", "car"]
        },
        {
            "task_type": "style_transfer",
            "user_intent": "oil painting style",
            "expected_contains": ["oil painting"]
        },
        {
            "task_type": "environment_change",
            "user_intent": "change to beach background",
            "expected_contains": ["beach", "background"]
        },
        {
            "task_type": "lighting_adjustment",
            "user_intent": "soft lighting from left",
            "expected_contains": ["soft", "left"]
        },
        {
            "task_type": "state_change",
            "user_intent": "make it look old and weathered",
            "expected_contains": ["old", "weathered"]
        },
        {
            "task_type": "outpainting",
            "user_intent": "extend image to the right",
            "expected_contains": ["extend", "right"]
        }
    ]
    
    # Create a dummy image for analysis
    dummy_image = Image.new('RGB', (512, 512), color='white')
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases):
        print(f"Test {i+1}: {test['task_type']} - '{test['user_intent']}'")
        
        try:
            # Analyze image (mock)
            analysis = analyzer.analyze(dummy_image)
            
            # Generate prompt
            prompt = generator.generate(
                task_type=test['task_type'],
                user_intent=test['user_intent'],
                image_analysis=analysis
            )
            
            print(f"Generated: {prompt}")
            
            # Check if expected words are in the prompt
            prompt_lower = prompt.lower()
            missing = []
            for expected in test['expected_contains']:
                if expected.lower() not in prompt_lower:
                    missing.append(expected)
            
            if missing:
                print(f"❌ FAILED - Missing expected words: {missing}")
                failed += 1
            else:
                print(f"✅ PASSED")
                passed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        print("-" * 50)
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    
    # Test preservation rules
    print("\n=== Testing Preservation Rules ===")
    for task_type in templates.get_all_task_types():
        rules = templates.get_preservation_rules(task_type)
        clause = templates.format_preservation_clause(rules)
        print(f"{task_type}: {clause}")
    
    # Additional specific tests
    print("\n=== Testing Specific Scenarios ===")
    specific_tests = [
        ("object_manipulation", "change the red car to blue"),
        ("style_transfer", "make it look like an oil painting"),
        ("lighting_adjustment", "add dramatic lighting from the left side"),
        ("state_change", "make the building look old and abandoned"),
    ]
    
    for task_type, intent in specific_tests:
        print(f"\nTask: {task_type}")
        print(f"Intent: '{intent}'")
        try:
            prompt = generator.generate(task_type, intent, analyzer.analyze(dummy_image))
            print(f"Result: {prompt}")
        except Exception as e:
            print(f"Error: {e}")
    
    return failed == 0

def test_with_real_images():
    """Test with actual image files if available"""
    
    print("\n\n=== Testing with Real Images ===\n")
    
    # Initialize components
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    analyzer = ImageAnalyzer()
    
    # Look for test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        test_images_dir = Path(".")
    
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if not image_files:
        print("No test images found. Place some .jpg or .png files in ./test_images/")
        return
    
    # Test with each image
    for img_path in image_files[:3]:  # Test up to 3 images
        print(f"\nTesting with: {img_path.name}")
        
        try:
            image = Image.open(img_path)
            
            # Analyze image
            print("Analyzing image...")
            analysis = analyzer.analyze(image)
            
            print(f"  Size: {analysis['size']}")
            print(f"  Main objects: {analysis['objects']['main'][:3]}")
            print(f"  Style: {analysis['style']['artistic']}")
            print(f"  Setting: {analysis['environment']['setting']}")
            
            # Generate prompts for different tasks
            test_intents = [
                ("object_manipulation", "make it more colorful"),
                ("style_transfer", "vintage photograph style"),
                ("environment_change", "sunset lighting"),
            ]
            
            for task_type, intent in test_intents:
                prompt = generator.generate(task_type, intent, analysis)
                print(f"\n  Task: {task_type}")
                print(f"  Intent: {intent}")
                print(f"  Generated: {prompt[:150]}...")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

def test_edge_cases():
    """Test edge cases and error handling"""
    
    print("\n\n=== Testing Edge Cases ===\n")
    
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    analyzer = ImageAnalyzer()
    
    # Test 1: Empty intent
    print("Test 1: Empty intent")
    try:
        prompt = generator.generate("object_manipulation", "", None)
        print(f"Result: {prompt}")
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Test 2: Invalid task type
    print("\nTest 2: Invalid task type")
    try:
        prompt = generator.generate("invalid_task", "test", None)
        print(f"Result: {prompt}")
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Test 3: None image
    print("\nTest 3: None image")
    try:
        analysis = analyzer.analyze(None)
        print(f"Analysis: {analysis['description']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Very long intent
    print("\nTest 4: Very long intent")
    long_intent = "make the car blue and add racing stripes and change the wheels to chrome and add a spoiler and tint the windows and lower the suspension"
    try:
        dummy_image = Image.new('RGB', (100, 100), color='red')
        analysis = analyzer.analyze(dummy_image)
        prompt = generator.generate("object_manipulation", long_intent, analysis)
        print(f"Result: {prompt[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests"""
    
    print("=" * 60)
    print("FLUX.1 Kontext Smart Assistant - Test Suite")
    print("=" * 60)
    
    # Basic tests
    success = test_prompt_generation()
    
    # Real image tests
    test_with_real_images()
    
    # Edge case tests
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)