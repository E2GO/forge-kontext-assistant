"""
FluxKontext Smart Assistant for Forge WebUI

An intelligent prompt generation assistant that analyzes context images
and generates proper instructional prompts for FLUX.1 Kontext model.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

# Package metadata
__all__ = [
    "KontextAssistant",
    "ImageAnalyzer", 
    "PromptGenerator",
    "Phi3Enhancer"
]

# Version check
import sys
if sys.version_info < (3, 10):
    raise RuntimeError("FluxKontext Smart Assistant requires Python 3.10 or higher")

# Import main components when used as a module
try:
    from .kontext_assistant import KontextAssistant
    from .modules.image_analyzer import ImageAnalyzer
    from .modules.prompt_generator import PromptGenerator
    from .modules.llm_enhancer import Phi3Enhancer
except ImportError:
    # This happens when running as Forge extension
    pass