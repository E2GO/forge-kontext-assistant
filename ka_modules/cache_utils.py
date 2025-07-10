"""
Cache utilities for memory management
"""

from collections import OrderedDict
from typing import Any, Optional, Dict
import logging
import sys

logger = logging.getLogger(__name__)


class LimitedCache:
    """Cache with size limit to prevent memory leaks"""
    
    def __init__(self, max_size: int = 10, max_memory_mb: int = 500):
        """
        Initialize cache with size and memory limits
        
        Args:
            max_size: Maximum number of items to cache
            max_memory_mb: Maximum memory usage in MB (approximate)
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._total_size = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving it to end (most recently used)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Add item to cache with LRU eviction"""
        # Estimate size of the value
        size_estimate = self._estimate_size(value)
        
        # If single item is too large, don't cache it
        if size_estimate > self.max_memory_mb * 1024 * 1024:
            logger.warning(f"Item {key} too large to cache ({size_estimate / 1024 / 1024:.1f} MB)")
            return
        
        # Remove items if we're at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
        
        # Remove items if we're over memory limit
        while self._total_size + size_estimate > self.max_memory_mb * 1024 * 1024 and self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
        
        # Add new item
        self.cache[key] = value
        self._total_size += size_estimate
        
    def _remove(self, key: str) -> None:
        """Remove item and update size tracking"""
        if key in self.cache:
            value = self.cache.pop(key)
            self._total_size -= self._estimate_size(value)
            logger.debug(f"Evicted {key} from cache")
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object in bytes"""
        try:
            return sys.getsizeof(obj)
        except:
            # Fallback for complex objects
            if isinstance(obj, dict):
                size = sys.getsizeof(obj)
                for k, v in obj.items():
                    size += sys.getsizeof(k) + self._estimate_size(v)
                return size
            elif isinstance(obj, (list, tuple)):
                size = sys.getsizeof(obj)
                for item in obj:
                    size += self._estimate_size(item)
                return size
            else:
                return 1024  # Default 1KB for unknown objects
    
    def clear(self) -> None:
        """Clear all cached items"""
        self.cache.clear()
        self._total_size = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'items': len(self.cache),
            'max_items': self.max_size,
            'memory_mb': self._total_size / 1024 / 1024,
            'max_memory_mb': self.max_memory_mb,
            'keys': list(self.cache.keys())
        }
    
    def __len__(self) -> int:
        """Number of items in cache"""
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache"""
        return key in self.cache


class ImageHashCache(LimitedCache):
    """Specialized cache for image hashes and analysis results"""
    
    def __init__(self, max_images: int = 9):
        """
        Initialize image cache
        
        Args:
            max_images: Maximum number of images to cache (default 9 for 3 slots × 3 operations)
        """
        # Each image analysis can be ~1-5MB, so limit memory accordingly
        super().__init__(max_size=max_images, max_memory_mb=50)
        
    def get_image_key(self, image_index: int, operation: str = "analysis") -> str:
        """Generate cache key for image operation"""
        return f"img_{image_index}_{operation}"
    
    def invalidate_image(self, image_index: int) -> None:
        """Remove all cached data for a specific image index"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"img_{image_index}_")]
        for key in keys_to_remove:
            self._remove(key)
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for image {image_index}")