"""
Test the shared state integration
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add the extension root to sys.path
extension_root = Path(__file__).parent
sys.path.insert(0, str(extension_root))

# Import shared state
from ka_modules.shared_state import shared_state

def create_test_image(color, size=(512, 512)):
    """Create a simple test image"""
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    if color == 'red':
        img_array[:, :, 0] = 255
    elif color == 'green':
        img_array[:, :, 1] = 255
    elif color == 'blue':
        img_array[:, :, 2] = 255
    return Image.fromarray(img_array)

def test_shared_state():
    print("=== Testing Shared State ===\n")
    
    # Create test images
    print("Creating test images...")
    img1 = create_test_image('red')
    img2 = create_test_image('green')
    img3 = create_test_image('blue')
    
    # Test setting images
    print("\nTesting set_images...")
    shared_state.set_images([img1, img2, img3])
    print(f"Image count: {shared_state.image_count}")
    
    # Test getting images
    print("\nTesting get_images...")
    retrieved = shared_state.get_images()
    print(f"Retrieved {len(retrieved)} images")
    print(f"All images valid: {all(img is not None for img in retrieved)}")
    
    # Test getting individual images
    print("\nTesting get_image...")
    for i in range(3):
        img = shared_state.get_image(i)
        print(f"Image {i+1}: {'✓' if img else '✗'}")
    
    # Test clearing
    print("\nTesting clear...")
    shared_state.clear()
    print(f"Image count after clear: {shared_state.image_count}")
    
    # Test partial update
    print("\nTesting partial update...")
    shared_state.set_images([img1, None, img3])
    print(f"Image count with None: {shared_state.image_count}")
    
    print("\n=== Shared State test complete! ===")

if __name__ == "__main__":
    test_shared_state()