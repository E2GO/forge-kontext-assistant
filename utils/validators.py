"""
Validation utilities for Kontext Assistant.
Provides validation for images, prompts, and other inputs.
"""

from typing import Tuple, Optional, List, Any
from PIL import Image
import re

# Compatibility
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


class ImageValidator:
    """Validates images for processing."""
    
    # Minimum dimensions
    MIN_WIDTH = 64
    MIN_HEIGHT = 64
    
    # Maximum dimensions (to prevent memory issues)
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096
    
    # Supported formats
    SUPPORTED_MODES = ['RGB', 'RGBA', 'L', 'P']
    
    @classmethod
    def validate_for_analysis(cls, image: Any) -> Tuple[bool, str]:
        """
        Validate image for analysis.
        
        Args:
            image: Image to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if None
        if image is None:
            return False, "Image cannot be None"
        
        # Check if PIL Image
        if not isinstance(image, Image.Image):
            return False, "Image must be a PIL Image object"
        
        # Check dimensions
        width, height = image.size
        
        if width < cls.MIN_WIDTH or height < cls.MIN_HEIGHT:
            return False, f"Image too small: {width}x{height}. Minimum: {cls.MIN_WIDTH}x{cls.MIN_HEIGHT}"
        
        if width > cls.MAX_WIDTH or height > cls.MAX_HEIGHT:
            return False, f"Image too large: {width}x{height}. Maximum: {cls.MAX_WIDTH}x{cls.MAX_HEIGHT}"
        
        # Check mode
        if image.mode not in cls.SUPPORTED_MODES:
            return False, f"Unsupported image mode: {image.mode}. Supported: {cls.SUPPORTED_MODES}"
        
        return True, "Image is valid"
    
    @classmethod
    def validate_for_generation(cls, width: int, height: int) -> Tuple[bool, str]:
        """
        Validate dimensions for generation.
        
        Args:
            width: Target width
            height: Target height
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if dimensions are positive
        if width <= 0 or height <= 0:
            return False, "Dimensions must be positive"
        
        # Check minimum
        if width < cls.MIN_WIDTH or height < cls.MIN_HEIGHT:
            return False, f"Dimensions too small. Minimum: {cls.MIN_WIDTH}x{cls.MIN_HEIGHT}"
        
        # Check maximum
        if width > cls.MAX_WIDTH or height > cls.MAX_HEIGHT:
            return False, f"Dimensions too large. Maximum: {cls.MAX_WIDTH}x{cls.MAX_HEIGHT}"
        
        # Check if dimensions are multiples of 8 (common requirement)
        if width % 8 != 0 or height % 8 != 0:
            return False, "Dimensions should be multiples of 8"
        
        return True, "Dimensions are valid"


class PromptValidator:
    """Validates prompts and user inputs."""
    
    # Length limits
    MIN_LENGTH = 1
    MAX_LENGTH = 500
    MAX_PROMPT_LENGTH = 1000
    
    # Prohibited patterns
    PROHIBITED_PATTERNS = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript protocol
        r'data:text/html',  # Data URLs
    ]
    
    @classmethod
    def validate_user_intent(cls, intent: str) -> Tuple[bool, str]:
        """
        Validate user intent/prompt.
        
        Args:
            intent: User's input text
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if None or empty
        if not intent:
            return False, "Intent cannot be empty"
        
        # Strip whitespace
        intent = intent.strip()
        
        # Check length
        if len(intent) < cls.MIN_LENGTH:
            return False, f"Intent too short. Minimum: {cls.MIN_LENGTH} characters"
        
        if len(intent) > cls.MAX_LENGTH:
            return False, f"Intent too long. Maximum: {cls.MAX_LENGTH} characters"
        
        # Check for prohibited patterns
        for pattern in cls.PROHIBITED_PATTERNS:
            if re.search(pattern, intent, re.IGNORECASE):
                return False, "Intent contains prohibited content"
        
        # Check for reasonable content
        if len(intent.split()) < 2:
            return False, "Intent should contain at least 2 words"
        
        return True, "Intent is valid"
    
    @classmethod
    def validate_generated_prompt(cls, prompt: str) -> Tuple[bool, str]:
        """
        Validate generated prompt.
        
        Args:
            prompt: Generated prompt text
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Basic checks
        if not prompt:
            return False, "Prompt cannot be empty"
        
        # Length check
        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            return False, f"Prompt too long. Maximum: {cls.MAX_PROMPT_LENGTH} characters"
        
        # Should contain instructional language
        instruction_words = ['change', 'add', 'remove', 'transform', 'convert', 'modify']
        has_instruction = any(word in prompt.lower() for word in instruction_words)
        
        if not has_instruction:
            return False, "Prompt should contain instructional language"
        
        return True, "Prompt is valid"
    
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """
        Sanitize user input text.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove any HTML/script tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit length
        if len(text) > cls.MAX_LENGTH:
            text = text[:cls.MAX_LENGTH]
        
        return text


class TaskValidator:
    """Validates task types and configurations."""
    
    VALID_TASK_TYPES = [
        "object_manipulation",
        "style_transfer",
        "environment_change",
        "element_combination",
        "state_change",
        "outpainting",
        "lighting_adjustment",
        "texture_change",
        "perspective_shift",
        "seasonal_change"
    ]
    
    @classmethod
    def validate_task_type(cls, task_type: str) -> Tuple[bool, str]:
        """
        Validate task type.
        
        Args:
            task_type: Task type string
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not task_type:
            return False, "Task type cannot be empty"
        
        if task_type not in cls.VALID_TASK_TYPES:
            return False, f"Invalid task type. Valid types: {', '.join(cls.VALID_TASK_TYPES)}"
        
        return True, "Task type is valid"
    
    @classmethod
    def validate_parameters(cls, params: dict) -> Tuple[bool, str]:
        """
        Validate task parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not isinstance(params, dict):
            return False, "Parameters must be a dictionary"
        
        # Check for required common parameters
        if 'preserve_strength' in params:
            strength = params['preserve_strength']
            if not isinstance(strength, (int, float)):
                return False, "preserve_strength must be a number"
            if not 0 <= strength <= 1:
                return False, "preserve_strength must be between 0 and 1"
        
        return True, "Parameters are valid"


# Helper functions
def validate_all(image: Any, intent: str, task_type: str) -> List[Tuple[bool, str]]:
    """
    Validate all inputs at once.
    
    Args:
        image: Input image
        intent: User intent
        task_type: Selected task type
        
    Returns:
        List of validation results
    """
    results = []
    
    # Validate image
    results.append(("Image", *ImageValidator.validate_for_analysis(image)))
    
    # Validate intent
    results.append(("Intent", *PromptValidator.validate_user_intent(intent)))
    
    # Validate task type
    results.append(("Task Type", *TaskValidator.validate_task_type(task_type)))
    
    return results