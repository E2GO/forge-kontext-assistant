"""
Input validation utilities
"""

from typing import Tuple
from PIL import Image

def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """Validate input image"""
    if image is None:
        return False, "Image is None"
    
    if not hasattr(image, 'size'):
        return False, "Invalid image object"
    
    width, height = image.size
    if width < 64 or height < 64:
        return False, f"Image too small: {width}x{height} (min 64x64)"
    
    if width > 4096 or height > 4096:
        return False, f"Image too large: {width}x{height} (max 4096x4096)"
    
    return True, "Valid"

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Validate user prompt"""
    if not prompt:
        return False, "Prompt is empty"
    
    if len(prompt) < 3:
        return False, "Prompt too short (min 3 characters)"
    
    if len(prompt) > 500:
        return False, "Prompt too long (max 500 characters)"
    
    # Check for basic instruction words
    instruction_words = ["change", "make", "add", "remove", "modify", "convert"]
    has_instruction = any(word in prompt.lower() for word in instruction_words)
    
    if not has_instruction:
        return True, "Valid (but consider using instruction words for better results)"
    
    return True, "Valid"

def validate_task_type(task_type: str) -> Tuple[bool, str]:
    """Validate task type"""
    valid_types = [
        "object_color", "object_state", "style_transfer",
        "environment_change", "element_combination", 
        "state_changes", "outpainting"
    ]
    
    if task_type not in valid_types:
        return False, f"Invalid task type. Valid types: {', '.join(valid_types)}"
    
    return True, "Valid"
