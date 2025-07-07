"""
Test script to verify prompt generation works
"""

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
            "expected_contains": ["oil", "painting"]
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
    
    return failed == 0

if __name__ == "__main__":
    success = test_prompt_generation()
    sys.exit(0 if success else 1)