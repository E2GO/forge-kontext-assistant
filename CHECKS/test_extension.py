"""
Quick test script for FluxKontext Smart Assistant
Run this to verify the extension loads correctly
"""

import sys
from pathlib import Path

# Add extension to path
ext_path = Path(__file__).parent
sys.path.insert(0, str(ext_path))

print("Testing FluxKontext Smart Assistant...")
print("=" * 50)

# Test 1: Import modules
try:
    from ka_modules import templates
    print("✅ Templates module loaded")
except Exception as e:
    print(f"❌ Templates module failed: {e}")

try:
    from ka_modules import ui_components  
    print("✅ UI components module loaded")
except Exception as e:
    print(f"❌ UI components failed: {e}")

try:
    from ka_modules import prompt_generator
    print("✅ Prompt generator module loaded")
except Exception as e:
    print(f"❌ Prompt generator failed: {e}")

try:
    from ka_modules import image_analyzer
    print("✅ Image analyzer module loaded")
except Exception as e:
    print(f"❌ Image analyzer failed: {e}")

# Test 2: Test template generation
print("\n" + "=" * 50)
print("Testing template system...")
try:
    from ka_modules.templates import get_template, TemplateBuilder
    
    template = get_template("object_color")
    if template:
        params = {
            "object": "car",
            "current_color": "red", 
            "new_color": "blue"
        }
        prompt = TemplateBuilder.fill_template(template, params)
        print(f"✅ Generated prompt: {prompt[:80]}...")
    else:
        print("❌ Failed to get template")
except Exception as e:
    print(f"❌ Template test failed: {e}")

# Test 3: Test prompt generator
print("\n" + "=" * 50)
print("Testing prompt generator...")
try:
    from ka_modules.prompt_generator import PromptGenerator
    
    generator = PromptGenerator()
    test_prompt = generator.generate(
        task_type="object_color",
        user_intent="make the car blue"
    )
    print(f"✅ Generated from intent: {test_prompt[:80]}...")
except Exception as e:
    print(f"❌ Prompt generator test failed: {e}")

# Test 4: Check if main script loads
print("\n" + "=" * 50)
print("Testing main script...")
try:
    import kontext_assistant
    if hasattr(kontext_assistant, 'script'):
        print("✅ Main script instance created")
        print(f"   Title: {kontext_assistant.script.title()}")
    else:
        print("❌ Script instance not found")
except Exception as e:
    print(f"❌ Main script failed: {e}")

print("\n" + "=" * 50)
print("Test complete!")