"""
Site customization to fix collections compatibility for Python 3.10+
Place this file in your Python site-packages directory to apply globally
"""

import collections
import collections.abc

# Copy all abstract base classes from collections.abc to collections
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

print("✅ Collections compatibility patch applied")