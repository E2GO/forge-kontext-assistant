"""
Model management for memory-efficient loading/unloading
"""

import torch
import gc
import logging
from typing import Optional, Dict, Any
from threading import Lock

logger = logging.getLogger("ModelManager")

class ModelManager:
    """Manages model loading/unloading for memory efficiency"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.models = {}
        self.memory_threshold = 0.9  # 90% VRAM usage threshold
        self._initialized = True
        logger.info("ModelManager initialized")
    
    def register_model(self, name: str, model_instance: Any):
        """Register a model instance"""
        self.models[name] = {
            "instance": model_instance,
            "loaded": False,
            "priority": 0
        }
        logger.info(f"Registered model: {name}")
    
    def load_model(self, name: str, force: bool = False) -> bool:
        """Load a model, potentially unloading others if needed"""
        if name not in self.models:
            logger.error(f"Model {name} not registered")
            return False
        
        model_info = self.models[name]
        
        # Check if already loaded
        if model_info["loaded"] and not force:
            logger.info(f"Model {name} already loaded")
            return True
        
        # Check memory before loading
        if not self._check_memory_available():
            logger.warning("Memory threshold exceeded, attempting to free memory")
            self._free_memory_for_model(name)
        
        try:
            # Load the model
            logger.info(f"Loading model: {name}")
            model_instance = model_info["instance"]
            
            if hasattr(model_instance, "load_model"):
                model_instance.load_model()
            
            model_info["loaded"] = True
            model_info["last_used"] = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            self._log_memory_usage()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return False
    
    def unload_model(self, name: str) -> bool:
        """Unload a specific model"""
        if name not in self.models:
            logger.error(f"Model {name} not registered")
            return False
        
        model_info = self.models[name]
        
        if not model_info["loaded"]:
            logger.info(f"Model {name} not loaded")
            return True
        
        try:
            logger.info(f"Unloading model: {name}")
            model_instance = model_info["instance"]
            
            if hasattr(model_instance, "unload_model"):
                model_instance.unload_model()
            
            model_info["loaded"] = False
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._log_memory_usage()
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {name}: {e}")
            return False
    
    def _check_memory_available(self) -> bool:
        """Check if enough memory is available"""
        if not torch.cuda.is_available():
            return True
        
        try:
            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            
            usage_ratio = allocated_memory / total_memory
            
            return usage_ratio < self.memory_threshold
            
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return True
    
    def _free_memory_for_model(self, target_model: str):
        """Free memory by unloading other models"""
        # Find loaded models sorted by priority (lower = less important)
        loaded_models = [
            (name, info) for name, info in self.models.items()
            if info["loaded"] and name != target_model
        ]
        
        # Sort by priority (ascending) - unload low priority first
        loaded_models.sort(key=lambda x: x[1]["priority"])
        
        # Unload models until we have enough memory
        for name, _ in loaded_models:
            self.unload_model(name)
            
            if self._check_memory_available():
                logger.info(f"Freed enough memory after unloading {name}")
                break
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    
    def set_model_priority(self, name: str, priority: int):
        """Set model priority (higher = more important)"""
        if name in self.models:
            self.models[name]["priority"] = priority
            logger.info(f"Set priority for {name}: {priority}")
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {name: info["loaded"] for name, info in self.models.items()}

# Global instance
model_manager = ModelManager()