"""
Debug script to add extra logging to kontext_assistant.py
Run this to add debug logging
"""

import re
from pathlib import Path

def add_debug_logging():
    # Read the current kontext_assistant.py
    assistant_file = Path("scripts/kontext_assistant.py")
    content = assistant_file.read_text(encoding='utf-8')
    
    # Add debug logging to _get_kontext_images_from_ui
    debug_code = '''    def _get_kontext_images_from_ui(self, *args) -> List[Optional[Image.Image]]:
        """Get kontext images from shared state or UI args"""
        kontext_images = []
        
        # Debug logging
        logger.info(f"[DEBUG] _get_kontext_images_from_ui called with {len(args)} args")
        
        # First try to get from shared state
        if MODULES_AVAILABLE and shared_state:
            kontext_images = shared_state.get_images()
            image_count = sum(1 for img in kontext_images if img is not None)
            logger.info(f"[DEBUG] Shared state has {image_count} images")
            if any(img is not None for img in kontext_images):
                logger.debug(f"Got {shared_state.image_count} images from shared state")
                return kontext_images
        else:
            logger.info("[DEBUG] Shared state not available")
        
        # Fallback to parsing UI args
        logger.debug(f"Total args received: {len(args)}")
        
        # Log arg types
        for i, arg in enumerate(args[:10]):  # First 10 args
            arg_type = type(arg).__name__
            if hasattr(arg, 'size') and hasattr(arg, 'mode'):
                logger.info(f"[DEBUG] Arg {i}: PIL Image {arg.size}")
            else:
                logger.info(f"[DEBUG] Arg {i}: {arg_type}")'''
    
    # Find and replace the method
    pattern = r'def _get_kontext_images_from_ui\(self, \*args\) -> List\[Optional\[Image\.Image\]\]:.*?"""Get kontext images from shared state or UI args"""'
    
    if re.search(pattern, content, re.DOTALL):
        # Replace the method signature and docstring
        content = re.sub(
            pattern,
            debug_code,
            content,
            count=1,
            flags=re.DOTALL
        )
        
        # Write back
        assistant_file.write_text(content, encoding='utf-8')
        print("✓ Added debug logging to kontext_assistant.py")
        print("\nNow restart Forge WebUI and check the console output when clicking 'Analyze Image'")
    else:
        print("✗ Could not find the method to patch")
        print("Make sure kontext_assistant.py has the latest version")

if __name__ == "__main__":
    add_debug_logging()