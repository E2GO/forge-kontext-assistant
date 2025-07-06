"""
Kontext Assistant modules package.
"""

# Fix for Python 3.10 compatibility
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

# Version
__version__ = "2.0.0"

# Import main classes for convenience
from .templates import PromptTemplates
from .prompt_generator import PromptGenerator
from .image_analyzer import ImageAnalyzer, Florence2ModelLoader
from .forge_integration import ForgeIntegration, KontextImageFinder

__all__ = [
    'PromptTemplates',
    'PromptGenerator',
    'ImageAnalyzer',
    'Florence2ModelLoader',
    'ForgeIntegration',
    'KontextImageFinder'
]