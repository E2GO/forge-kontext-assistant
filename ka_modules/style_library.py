"""
Style Library Proxy for Backward Compatibility
All styles are now organized in the styles/ subdirectory
"""

# Import everything from the new location
from .styles import StyleLibrary, StylePreset, StyleCategory, ALL_STYLES

# For backward compatibility with direct imports
styles = ALL_STYLES

# Re-export everything
__all__ = ['StyleLibrary', 'StylePreset', 'StyleCategory', 'styles']

# Note: This is a proxy module. The actual implementation is in ka_modules/styles/
# This file exists only to maintain backward compatibility with existing code.