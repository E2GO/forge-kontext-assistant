"""
Cache utility for Kontext Assistant.
Provides simple caching functionality with LRU eviction.
"""

from typing import Dict, Any, Optional, List
from collections import OrderedDict
import time
import hashlib
import json

# Compatibility
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


class SimpleCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache with maximum size.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class TimedCache:
    """Cache with time-based expiration."""
    
    def __init__(self, ttl_seconds: float = 3600.0):
        """
        Initialize timed cache.
        
        Args:
            ttl_seconds: Time to live in seconds
        """
        self.ttl = ttl_seconds
        self.cache: Dict[str, tuple] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # Expired
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value with current timestamp."""
        self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Hash string suitable for cache key
    """
    # Convert to string representation
    key_data = {
        'args': args,
        'kwargs': kwargs
    }
    
    # Create hash
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


# For backward compatibility
class AnalysisCache(SimpleCache):
    """Specialized cache for image analysis results."""
    
    def __init__(self, max_size: int = 50):
        super().__init__(max_size)
    
    def get_image_key(self, image) -> str:
        """Generate cache key for an image."""
        if hasattr(image, 'size') and hasattr(image, 'mode'):
            # PIL Image
            return f"{image.size}_{image.mode}_{hash(image.tobytes()) % 1000000}"
        else:
            # Other image type
            return str(hash(str(image)))