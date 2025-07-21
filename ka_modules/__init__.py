# Ka_modules package for Forge Kontext Assistant
"""
Smart Assistant modules for FLUX.1 Kontext
"""

# Make modules easily importable
from .templates import PromptTemplates
from .image_analyzer import ImageAnalyzer

__all__ = [
    'PromptTemplates',
     
    'ImageAnalyzer'
]

__version__ = '1.0.1'
