# Ka_modules package for Forge Kontext Assistant
"""
Smart Assistant modules for FLUX.1 Kontext
"""

# Make modules easily importable
from .templates import PromptTemplates
from .prompt_generator import PromptGenerator
from .image_analyzer import ImageAnalyzer

__all__ = [
    'PromptTemplates',
    'PromptGenerator', 
    'ImageAnalyzer'
]

__version__ = '2.0.0'
