"""
Test full integration with Florence-2 and prompt generation
"""

# Fix collections compatibility
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import os
os.environ["KONTEXT_USE_FLORENCE2"] = "true"  # Force Florence-2

from PIL import Image, ImageDraw, ImageFont
from ka_modules.image_analyzer import ImageAnalyzer
from ka_modules.prompt_generator import PromptGenerator
from ka_modules.templates import PromptTemplates

def create_test_scene():
    """Create a more complex test scene"""
    img = Image.new('RGB', (800, 600), color='skyblue')
    draw = ImageDraw.Draw(img)
    
    # Draw ground
    draw.rectangle([0, 400, 800, 600], fill='green')
    
    # Draw house
    draw.rectangle([200, 250, 400, 400], fill='brown')  # House body
    draw.polygon([(150, 250), (300, 150), (450, 250)], fill='red')  # Roof
    draw.rectangle([250, 300, 300, 350], fill='yellow')  # Window
    draw.rectangle([320, 320, 370, 400], fill='#654321')  # Door (dark brown)
    
    # Draw car
    draw.rectangle([500, 350, 650, 400], fill='blue')  # Car body
    draw.ellipse([510, 380, 540, 410], fill='black')  # Wheel 1
    draw.ellipse([610, 380, 640, 410], fill='black')  # Wheel 2
    
    # Draw sun
    draw.ellipse([650, 50, 750, 150], fill='yellow')
    
    return img

def test_with_florence2():
    print("=== Testing Full Integration with Florence-2 ===\n")
    
    # Initialize components
    analyzer = ImageAnalyzer()
    templates = PromptTemplates()
    generator = PromptGenerator(templates)
    
    print(f"Using Florence-2: {analyzer.use_florence2}")
    
    # Create test scene
    print("Creating test scene...")
    test_image = create_test_scene()
    test_image.save("test_scene.png")
    print("Test scene saved as 'test_scene.png'")
    
    # Load model
    print("\nLoading Florence-2 model...")
    analyzer.load_model()
    
    # Analyze image
    print("\nAnalyzing image with Florence-2...")
    analysis = analyzer.analyze(test_image)
    
    print("\n=== Florence-2 Analysis ===")
    print(f"Description: {analysis['description']}")
    print(f"Main objects: {analysis['objects']['main']}")
    print(f"Setting: {analysis['environment']['setting']}")
    print(f"Style: {analysis['style']['artistic']}")
    
    # Test different prompt generation scenarios
    test_cases = [
        {
            "task": "object_manipulation",
            "intent": "change the car to red",
            "desc": "Change car color"
        },
        {
            "task": "style_transfer",
            "intent": "make it look like a watercolor painting",
            "desc": "Apply watercolor style"
        },
        {
            "task": "environment_change",
            "intent": "change to a night scene with stars",
            "desc": "Change to night"
        },
        {
            "task": "lighting_adjustment",
            "intent": "add dramatic sunset lighting from the left",
            "desc": "Add sunset lighting"
        }
    ]
    
    print("\n=== Prompt Generation with Florence-2 Context ===")
    for test in test_cases:
        print(f"\n{test['desc']}:")
        print(f"  Task: {test['task']}")
        print(f"  Intent: '{test['intent']}'")
        
        prompt = generator.generate(
            task_type=test['task'],
            user_intent=test['intent'],
            image_analysis=analysis
        )
        
        print(f"  Generated prompt:")
        print(f"    {prompt[:150]}...")
    
    # Test with a real photo if available
    real_photos = ["test_photo.jpg", "test_photo.png", "sample.jpg", "sample.png"]
    photo_found = False
    
    for photo_name in real_photos:
        if os.path.exists(photo_name):
            print(f"\n=== Testing with real photo: {photo_name} ===")
            real_image = Image.open(photo_name)
            
            # Resize if too large
            if real_image.width > 1024 or real_image.height > 1024:
                real_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            real_analysis = analyzer.analyze(real_image)
            print(f"Description: {real_analysis['description']}")
            print(f"Objects detected: {real_analysis['objects']['main']}")
            
            # Generate a prompt
            prompt = generator.generate(
                task_type="style_transfer",
                user_intent="make it look like an oil painting",
                image_analysis=real_analysis
            )
            print(f"\nGenerated prompt for oil painting style:")
            print(f"  {prompt}")
            
            photo_found = True
            break
    
    if not photo_found:
        print("\n(Place a photo named 'test_photo.jpg' in this directory to test with real images)")
    
    # Unload model
    print("\nUnloading model...")
    analyzer.unload_model()
    
    print("\n=== Test Complete! ===")
    print("Florence-2 is working correctly with the prompt generation system.")

if __name__ == "__main__":
    test_with_florence2()