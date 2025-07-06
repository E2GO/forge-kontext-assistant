"""
Simple caching utilities for kontext assistant
"""

from functools import lru_cache
from typing import Any, Dict
import hashlib
import json

class ResultCache:
    """Simple result cache for image analysis"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        
    def _get_key(self, image_path: str, task_type: str) -> str:
        """Generate cache key"""
        data = f"{image_path}:{task_type}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, image_path: str, task_type: str) -> Any:
        """Get cached result"""
        key = self._get_key(image_path, task_type)
        return self.cache.get(key)
    
    def set(self, image_path: str, task_type: str, result: Any):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self._get_key(image_path, task_type)
        self.cache[key] = result
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

# Global cache instance
analysis_cache = ResultCache()
