"""
Utility modules for Kontext Assistant.
"""

# Fix for Python 3.10 compatibility
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

from .cache import SimpleCache
from .validators import ImageValidator, PromptValidator

__all__ = ['SimpleCache', 'ImageValidator', 'PromptValidator']