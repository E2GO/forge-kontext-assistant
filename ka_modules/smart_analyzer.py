"""
Smart Analyzer - Simplified dual-model system
Automatically uses Florence-2 Base + PromptGen v2.0 for comprehensive analysis
"""

import logging
import time
import threading
from typing import Dict, Optional, Any, List
from PIL import Image
import torch

from ka_modules.image_analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)


class SmartAnalyzer:
    """
    Simplified analyzer that automatically uses both Florence-2 models:
    - Florence-2 Base for initial analysis and object detection
    - PromptGen v2.0 for detailed tags, mixed caption, and analysis
    """
    
    def __init__(self, device='cuda', force_cpu=False, florence_model_type="base", auto_unload=True, unload_delay=60):
        """
        Initialize smart analyzer
        
        Args:
            device: Device for computation
            force_cpu: Force CPU usage
            florence_model_type: Ignored - always uses dual model approach
            auto_unload: Automatically unload models after analysis to save memory
            unload_delay: Delay in seconds before unloading models (0 = immediate)
        """
        self.device = device
        self.force_cpu = force_cpu
        self.auto_unload = auto_unload
        self.unload_delay = unload_delay
        
        # Initialize both Florence models
        self.florence_base = None
        self.florence_promptgen = None
        
        # Thread locks for safe initialization
        self._base_lock = threading.Lock()
        self._promptgen_lock = threading.Lock()
        
        # Timer for delayed unload
        self._unload_timer = None
        
        logger.info(f"Smart analyzer initialized for device: {device}, auto_unload: {auto_unload}, unload_delay: {unload_delay}s")
    
    def _cancel_unload_timer(self):
        """Cancel any pending unload timer"""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None
    
    def _schedule_unload(self):
        """Schedule model unload after delay"""
        # Cancel any existing timer
        self._cancel_unload_timer()
        
        if self.unload_delay <= 0:
            # Immediate unload
            logger.info("Auto-unloading models immediately...")
            self.unload_models()
        else:
            # Delayed unload
            logger.info(f"Scheduling model unload in {self.unload_delay} seconds...")
            self._unload_timer = threading.Timer(self.unload_delay, self._delayed_unload)
            self._unload_timer.start()
    
    def _delayed_unload(self):
        """Execute delayed unload"""
        logger.info("Executing delayed model unload...")
        self.unload_models()
        self._unload_timer = None
    
    def _ensure_florence_base(self):
        """Lazy initialize Florence-2 Base"""
        # Cancel any pending unload since we're using the models
        self._cancel_unload_timer()
        
        if self.florence_base is None:
            with self._base_lock:
                if self.florence_base is None:
                    # Aggressive memory cleanup before loading
                    logger.info("Performing memory cleanup before loading Florence-2 Base...")
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    logger.info("Initializing Florence-2 Base analyzer...")
                    self.florence_base = ImageAnalyzer(
                        device=self.device,
                        force_cpu=self.force_cpu,
                        model_type="base"
                    )
                    logger.info(f"Florence-2 Base initialized: {self.florence_base.model_name}")
    
    def _ensure_florence_promptgen(self):
        """Lazy initialize Florence-2 PromptGen v2.0"""
        # Cancel any pending unload since we're using the models
        self._cancel_unload_timer()
        
        if self.florence_promptgen is None:
            with self._promptgen_lock:
                if self.florence_promptgen is None:
                    # Aggressive memory cleanup before loading
                    logger.info("Performing memory cleanup before loading PromptGen v2.0...")
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    logger.info("Initializing Florence-2 PromptGen v2.0 analyzer...")
                    self.florence_promptgen = ImageAnalyzer(
                        device=self.device,
                        force_cpu=self.force_cpu,
                        model_type="promptgen_v2"
                    )
                    logger.info(f"PromptGen v2.0 initialized: {self.florence_promptgen.model_name}")
    
    def analyze(self, image: Image.Image, use_florence: bool = True, 
                use_joycaption: bool = False,  # Ignored - for compatibility
                promptgen_instruction: str = "<MORE_DETAILED_CAPTION>") -> Dict[str, Any]:
        """
        Perform comprehensive analysis using both Florence models
        
        Args:
            image: PIL Image to analyze
            use_florence: Ignored - always True
            use_joycaption: Ignored - always False
            promptgen_instruction: Ignored - always uses MORE_DETAILED_CAPTION
            
        Returns:
            Combined analysis results from both models
        """
        start_time = time.time()
        results = {
            'success': True,
            'analyzers_used': {
                'florence2': True,
                'joycaption': False
            }
        }
        
        try:
            # First: Run Florence-2 Base for brief description and object detection
            logger.info("Running Florence-2 Base analysis...")
            self._ensure_florence_base()
            base_result = self.florence_base.analyze(image, detailed=False)
            
            # Clear some memory before running PromptGen
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Second: Run PromptGen v2.0 with MORE_DETAILED_CAPTION
            logger.info("Running PromptGen v2.0 analysis...")
            self._ensure_florence_promptgen()
            promptgen_result = self.florence_promptgen.analyze(
                image, 
                detailed=True, 
                promptgen_instruction="<MORE_DETAILED_CAPTION>"
            )
            
            # Combine results in new format
            results.update(self._format_dual_results(base_result, promptgen_result))
            
            # Add timing
            results['total_analysis_time'] = time.time() - start_time
            
            # Automatically unload models to free memory after analysis if enabled
            if self.auto_unload:
                self._schedule_unload()
            
        except Exception as e:
            logger.error(f"Smart analysis failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            # Try to unload models even on error if auto_unload is enabled
            if self.auto_unload:
                try:
                    self.unload_models()
                except:
                    pass
        
        return results
    
    def _format_dual_results(self, base: Dict[str, Any], promptgen: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results from both models into unified output
        """
        # Extract key information from both models
        result = {
            # Brief description from Florence-2 Base
            'brief_description': base.get('description', ''),
            
            # Detailed description from PromptGen v2.0
            'description': promptgen.get('description', ''),
            
            # Tags from PromptGen v2.0
            'tags': promptgen.get('tags', {}),
            
            # Mixed caption optimized for FLUX
            'mixed_caption': promptgen.get('mixed_caption', ''),
            
            # Technical data from Florence-2 Base
            'technical_data': {
                'size': base.get('size', 'unknown'),
                'mode': base.get('mode', 'unknown'),
                'objects_with_positions': base.get('objects', {}),
                'text_detected': base.get('text', [])
            },
            
            # Object detection from Florence-2 Base
            'objects': base.get('objects', {}),
            
            # Style and composition from PromptGen v2.0
            'style': promptgen.get('style', {}),
            'composition_analysis': promptgen.get('composition_analysis', ''),
            
            # Model information
            'models_used': {
                'base': 'Florence-2 Base (Microsoft)',
                'promptgen': 'Florence-2 PromptGen v2.0'
            },
            
            # Display mode
            'analysis_mode': 'dual_model_automatic',
            
            # For UI compatibility
            'analyzers_used': {
                'florence2': True,
                'joycaption': False
            },
            'model_type': 'dual_model'
        }
        
        # Add mood if available
        if 'mood' in promptgen:
            result['mood'] = promptgen['mood']
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of both analyzers"""
        status = {
            'florence_base': self.florence_base.get_status() if self.florence_base else {'loaded': False},
            'florence_promptgen': self.florence_promptgen.get_status() if self.florence_promptgen else {'loaded': False},
            'joycaption': {'loaded': False}  # Always false now
        }
        return status
    
    def unload_models(self):
        """Unload both models to free memory"""
        logger.info("Starting model unload process...")
        
        # Cancel any pending timer
        self._cancel_unload_timer()
        
        # Unload base model
        with self._base_lock:
            if self.florence_base:
                try:
                    self.florence_base.unload_model()
                except Exception as e:
                    logger.warning(f"Error unloading Florence Base: {e}")
                finally:
                    self.florence_base = None
        
        # Unload PromptGen model
        with self._promptgen_lock:
            if self.florence_promptgen:
                try:
                    self.florence_promptgen.unload_model()
                except Exception as e:
                    logger.warning(f"Error unloading PromptGen: {e}")
                finally:
                    self.florence_promptgen = None
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("All models unloaded and memory cleared")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.unload_models()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.unload_models()
        return False