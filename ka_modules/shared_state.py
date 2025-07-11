"""
Shared state management for Kontext Assistant
Provides thread-safe access to kontext images across scripts
"""

import threading
from typing import List, Optional
from PIL import Image
import logging

logger = logging.getLogger("KontextAssistant.SharedState")


class KontextSharedState:
    """Thread-safe shared state for kontext images"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._kontext_images: List[Optional[Image.Image]] = [None, None, None]
        self._image_hashes: List[Optional[str]] = [None, None, None]
        self._last_update_time = 0
        self._image_lock = threading.Lock()
        self._change_callbacks = []  # Callbacks for image changes
        self._initialized = True
        logger.info("KontextSharedState initialized")
    
    def set_images(self, images: List[Optional[Image.Image]]) -> None:
        """Set kontext images thread-safely with validation and change detection"""
        logger.debug(f"set_images called with {len(images)} images: {[img is not None for img in images]}")
        with self._image_lock:
            # Validate images and detect changes
            validated_images = []
            new_hashes = []
            changes = []
            
            for i, img in enumerate(images[:3]):
                if img is not None:
                    if not isinstance(img, Image.Image):
                        logger.warning(f"Invalid image type at index {i}: {type(img)}")
                        validated_images.append(None)
                        new_hashes.append(None)
                        if self._image_hashes[i] is not None:
                            changes.append((i, 'removed'))
                    else:
                        try:
                            # Validate image is accessible
                            _ = img.size
                            _ = img.mode
                            validated_images.append(img)
                            # Calculate hash for change detection
                            import hashlib
                            img_bytes = img.tobytes()
                            img_hash = hashlib.md5(img_bytes).hexdigest()
                            new_hashes.append(img_hash)
                            
                            # Check if image changed
                            if self._image_hashes[i] != img_hash:
                                if self._image_hashes[i] is None:
                                    changes.append((i, 'added'))
                                else:
                                    changes.append((i, 'changed'))
                        except Exception as e:
                            logger.error(f"Image {i} validation failed: {e}")
                            validated_images.append(None)
                            new_hashes.append(None)
                            if self._image_hashes[i] is not None:
                                changes.append((i, 'removed'))
                else:
                    validated_images.append(None)
                    new_hashes.append(None)
                    if self._image_hashes[i] is not None:
                        changes.append((i, 'removed'))
            
            # Ensure we have exactly 3 slots
            while len(validated_images) < 3:
                validated_images.append(None)
                new_hashes.append(None)
            
            self._kontext_images = validated_images
            self._image_hashes = new_hashes[:3]
            
            import time
            self._last_update_time = time.time()
            
            # Log update and changes
            count = sum(1 for img in self._kontext_images if img is not None)
            logger.debug(f"Updated kontext images: {count} images stored")
            
            # Notify callbacks about changes
            if changes:
                logger.info(f"Image changes detected: {changes}")
                for callback in self._change_callbacks:
                    try:
                        callback(changes)
                    except Exception as e:
                        logger.error(f"Error in change callback: {e}")
    
    def get_images(self) -> List[Optional[Image.Image]]:
        """Get kontext images thread-safely"""
        with self._image_lock:
            # Debug logging
            logger.debug(f"get_images called, current state: {[img is not None for img in self._kontext_images]}")
            return self._kontext_images.copy()
    
    def get_image(self, index: int) -> Optional[Image.Image]:
        """Get a specific kontext image by index"""
        with self._image_lock:
            if 0 <= index < len(self._kontext_images):
                return self._kontext_images[index]
            return None
    
    def clear(self) -> None:
        """Clear all stored images"""
        with self._image_lock:
            self._kontext_images = [None, None, None]
            logger.debug("Cleared all kontext images")
    
    @property
    def image_count(self) -> int:
        """Get count of non-None images"""
        with self._image_lock:
            return sum(1 for img in self._kontext_images if img is not None)
    
    def register_change_callback(self, callback) -> None:
        """Register a callback for image changes"""
        self._change_callbacks.append(callback)
        logger.debug(f"Registered change callback: {callback}")
    
    def unregister_change_callback(self, callback) -> None:
        """Unregister a change callback"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            logger.debug(f"Unregistered change callback: {callback}")
    
    def get_image_hashes(self) -> List[Optional[str]]:
        """Get current image hashes"""
        with self._image_lock:
            return self._image_hashes.copy()


# Global instance
shared_state = KontextSharedState()