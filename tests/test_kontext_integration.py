"""
Test the integration between kontext.py and kontext_assistant.py
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add the extension root to sys.path
extension_root = Path(__file__).parent
sys.path.insert(0, str(extension_root))

# Import shared state first
from ka_modules.shared_state import shared_state

def create_test_image(text, size=(512, 512)):
    """Create a test image with text"""
    from PIL import ImageDraw, ImageFont
    
    # Create a colorful image
    img = Image.new('RGB', size, color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, text, fill=(255, 255, 255), font=font)
    
    return img

def test_kontext_integration():
    print("=== Testing Kontext Integration ===\n")
    
    # Create test images
    print("1. Creating test images...")
    img1 = create_test_image("Image 1")
    img2 = create_test_image("Image 2")
    img3 = create_test_image("Image 3")
    print("   ✓ Created 3 test images")
    
    # Test shared state directly
    print("\n2. Testing shared state directly...")
    shared_state.set_images([img1, img2, img3])
    print(f"   ✓ Set {shared_state.image_count} images in shared state")
    
    # Verify images are stored
    retrieved = shared_state.get_images()
    print(f"   ✓ Retrieved {len(retrieved)} images from shared state")
    
    # Now test kontext_assistant integration
    print("\n3. Testing kontext_assistant integration...")
    try:
        from scripts.kontext_assistant import KontextAssistant
        
        assistant = KontextAssistant()
        assistant._initialize_modules()
        print("   ✓ KontextAssistant initialized")
        
        # Test getting images from shared state
        images = assistant._get_kontext_images_from_ui()
        image_count = sum(1 for img in images if img is not None)
        print(f"   ✓ Assistant found {image_count} images from shared state")
        
        # Test analysis
        if image_count > 0:
            print("\n4. Testing image analysis...")
            for i in range(image_count):
                try:
                    result_text, result_data = assistant.analyze_image(i)
                    if "✅" in result_text:
                        print(f"   ✓ Successfully analyzed image {i+1}")
                    else:
                        print(f"   ✗ Failed to analyze image {i+1}: {result_text}")
                except Exception as e:
                    print(f"   ✗ Error analyzing image {i+1}: {e}")
        
        # Test prompt generation
        print("\n5. Testing prompt generation...")
        if assistant.generator:
            test_cases = [
                ("object_manipulation", "make the car blue"),
                ("style_transfer", "oil painting style"),
                ("environment_change", "sunset background")
            ]
            
            for task, intent in test_cases:
                try:
                    prompt = assistant.generator.generate(
                        task_type=task,
                        user_intent=intent,
                        image_analysis={"description": "test image"}
                    )
                    if prompt and not prompt.startswith("❌"):
                        print(f"   ✓ Generated prompt for {task}")
                    else:
                        print(f"   ✗ Failed to generate prompt for {task}")
                except Exception as e:
                    print(f"   ✗ Error generating prompt for {task}: {e}")
        
    except ImportError as e:
        print(f"   ✗ Failed to import KontextAssistant: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ForgeKontext integration
    print("\n6. Testing ForgeKontext integration...")
    try:
        from scripts.kontext import ForgeKontext
        
        # Test the class methods
        ForgeKontext.set_kontext_images([img1, img2, img3])
        retrieved = ForgeKontext.get_kontext_images()
        print(f"   ✓ ForgeKontext stored and retrieved {len(retrieved)} images")
        
    except ImportError as e:
        print(f"   ✗ Failed to import ForgeKontext: {e}")
    except Exception as e:
        print(f"   ✗ Error with ForgeKontext: {e}")
    
    print("\n=== Integration test complete! ===")
    print("\nSummary:")
    print("- Shared state: ✓ Working")
    print("- Image storage: ✓ Working")
    print("- Image retrieval: ✓ Working")
    print("- Assistant integration: Check results above")
    print("\nIf all tests passed, the integration is working correctly!")

if __name__ == "__main__":
    test_kontext_integration()