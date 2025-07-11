"""
Image utilities for handling Gradio image loading issues
"""

import logging
from typing import Optional, List, Union
from PIL import Image
import numpy as np
import io
import base64

logger = logging.getLogger(__name__)


def validate_and_convert_image(img: Union[Image.Image, np.ndarray, str, dict, None]) -> Optional[Image.Image]:
    """
    Validate and convert various image formats to PIL Image.
    Handles Gradio's different image representations.
    """
    if img is None:
        return None
    
    try:
        # Already a PIL Image
        if isinstance(img, Image.Image):
            # Verify it's accessible
            _ = img.size
            return img
        
        # NumPy array (common from Gradio)
        elif isinstance(img, np.ndarray):
            # Handle different array shapes
            if img.ndim == 2:  # Grayscale
                return Image.fromarray(img.astype('uint8'), 'L')
            elif img.ndim == 3:
                if img.shape[2] == 3:  # RGB
                    return Image.fromarray(img.astype('uint8'), 'RGB')
                elif img.shape[2] == 4:  # RGBA
                    return Image.fromarray(img.astype('uint8'), 'RGBA')
        
        # String path (file path from Gradio)
        elif isinstance(img, str):
            try:
                # Try to open as file
                pil_img = Image.open(img)
                # Convert to RGB to ensure consistency
                if pil_img.mode not in ('RGB', 'RGBA'):
                    pil_img = pil_img.convert('RGB')
                return pil_img
            except Exception as e:
                logger.debug(f"Failed to open image from path: {e}")
                
                # Try base64 decode
                if img.startswith('data:image'):
                    try:
                        # Extract base64 data
                        base64_data = img.split(',')[1]
                        img_data = base64.b64decode(base64_data)
                        pil_img = Image.open(io.BytesIO(img_data))
                        if pil_img.mode not in ('RGB', 'RGBA'):
                            pil_img = pil_img.convert('RGB')
                        return pil_img
                    except Exception as e:
                        logger.debug(f"Failed to decode base64 image: {e}")
        
        # Dictionary (Gradio sometimes returns dict with 'name' key)
        elif isinstance(img, dict):
            if 'name' in img:
                return validate_and_convert_image(img['name'])
            elif 'path' in img:
                return validate_and_convert_image(img['path'])
            elif 'data' in img:
                return validate_and_convert_image(img['data'])
        
        # Try to convert unknown types
        else:
            logger.warning(f"Unknown image type: {type(img)}")
            # Try to convert to numpy array
            try:
                arr = np.array(img)
                return validate_and_convert_image(arr)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Failed to validate/convert image: {e}")
    
    return None


def validate_image_list(images: List[Union[Image.Image, np.ndarray, str, dict, None]]) -> List[Optional[Image.Image]]:
    """
    Validate and convert a list of images.
    Always returns a list of exactly 3 images (None for missing).
    """
    validated = []
    
    # Process provided images
    for i, img in enumerate(images[:3]):
        validated_img = validate_and_convert_image(img)
        validated.append(validated_img)
        
        if validated_img is None and img is not None:
            logger.warning(f"Failed to validate image {i+1}")
    
    # Ensure exactly 3 slots
    while len(validated) < 3:
        validated.append(None)
    
    return validated


def copy_image_safely(img: Optional[Image.Image]) -> Optional[Image.Image]:
    """
    Create a safe copy of an image to avoid reference issues.
    """
    if img is None:
        return None
    
    try:
        # Create a full copy
        return img.copy()
    except Exception as e:
        logger.error(f"Failed to copy image: {e}")
        try:
            # Try alternative copy method
            return Image.fromarray(np.array(img))
        except:
            return None