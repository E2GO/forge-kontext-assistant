import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from ka_modules.templates import PromptTemplates
        from ka_modules.prompt_generator import PromptGenerator
        from ka_modules.image_analyzer import ImageAnalyzer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_template_generation():
    """Test basic template generation"""
    try:
        from ka_modules.templates import PromptTemplates
        from ka_modules.prompt_generator import PromptGenerator
        
        templates = PromptTemplates()
        generator = PromptGenerator(templates)
        
        # Test simple color change
        prompt = generator.generate(
            task_type="object_manipulation",
            user_intent="make the car blue"
        )
        
        assert "blue" in prompt.lower()
        assert len(prompt) > 10
        
        print("✅ Template generation works")
        return True
    except Exception as e:
        print(f"❌ Template generation error: {e}")
        return False

def test_image_analyzer():
    """Test image analyzer initialization"""
    try:
        from ka_modules.image_analyzer import ImageAnalyzer
        
        analyzer = ImageAnalyzer()
        assert analyzer is not None
        
        print("✅ Image analyzer initialization works")
        return True
    except Exception as e:
        print(f"❌ Image analyzer error: {e}")
        return False

if __name__ == "__main__":
    print("Running basic tests...")
    
    tests = [
        test_imports,
        test_template_generation,
        test_image_analyzer
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed < total:
        sys.exit(1)
