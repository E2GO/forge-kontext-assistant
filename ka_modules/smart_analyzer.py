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


# Analysis mode configurations
ANALYSIS_MODES = {
    "fast": {
        "name": "Quick Description (fast)",
        "description": "Florence-only quick analysis",
        "florence_tasks": ["<DETAILED_CAPTION>"],
        "promptgen_tasks": [],  # No PromptGen - truly fast mode
        "estimated_time": "1-2 seconds"
    },
    "standard": {
        "name": "Standard", 
        "description": "Balanced analysis",
        "florence_tasks": ["<CAPTION>", "<OD>"],
        "promptgen_tasks": ["<MORE_DETAILED_CAPTION>", "<GENERATE_TAGS>", "<MIXED_CAPTION>"],
        "estimated_time": "5-7 seconds"
    },
    "detailed": {
        "name": "Detailed",
        "description": "Comprehensive analysis",
        "florence_tasks": ["<DETAILED_CAPTION>", "<OD>", "<DENSE_REGION_CAPTION>"],
        "promptgen_tasks": ["<MORE_DETAILED_CAPTION>", "<GENERATE_TAGS>", "<ANALYZE>", "<MIXED_CAPTION_PLUS>"],
        "estimated_time": "10-12 seconds"
    },
    "tags": {
        "name": "Tags Only",
        "description": "Just generate tags",
        "florence_tasks": [],  # Skip Florence for tags-only mode
        "promptgen_tasks": ["<GENERATE_TAGS>"],
        "estimated_time": "1-2 seconds"
    },
    "composition": {
        "name": "Composition",
        "description": "Spatial analysis",
        "florence_tasks": ["<DENSE_REGION_CAPTION>", "<OD>"],
        "promptgen_tasks": ["<ANALYZE>"],
        "estimated_time": "4-5 seconds"
    }
}


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
                promptgen_instruction: str = "<MORE_DETAILED_CAPTION>",
                analysis_mode: str = "standard") -> Dict[str, Any]:
        """
        Perform analysis using specified mode
        
        Args:
            image: PIL Image to analyze
            use_florence: Ignored - determined by mode
            use_joycaption: Ignored - always False
            promptgen_instruction: Ignored - determined by mode
            analysis_mode: Analysis mode (fast, standard, detailed, tags, composition)
            
        Returns:
            Analysis results based on selected mode
        """
        start_time = time.time()
        
        # Get mode configuration
        mode_config = ANALYSIS_MODES.get(analysis_mode, ANALYSIS_MODES["standard"])
        logger.info(f"Running analysis in '{mode_config['name']}' mode")
        
        results = {
            'success': True,
            'analysis_mode': analysis_mode,
            'mode_name': mode_config['name'],
            'analyzers_used': {
                'florence2': len(mode_config['florence_tasks']) > 0,
                'joycaption': False  # Always False for compatibility
            }
        }
        
        try:
            florence_results = {}
            promptgen_results = {}
            
            # Run Florence-2 Base tasks if any
            if mode_config['florence_tasks']:
                logger.info(f"Running Florence-2 Base tasks: {mode_config['florence_tasks']}")
                self._ensure_florence_base()
                
                for task in mode_config['florence_tasks']:
                    if task == "<CAPTION>":
                        # Basic caption
                        result = self.florence_base._run_florence_task(image, task)
                        florence_results['caption'] = result.get('<CAPTION>', '') if result else ''
                    elif task == "<DETAILED_CAPTION>":
                        # Detailed caption
                        result = self.florence_base._run_florence_task(image, task)
                        florence_results['detailed_caption'] = result.get('<DETAILED_CAPTION>', '') if result else ''
                    elif task == "<OD>":
                        # Object detection
                        result = self.florence_base._run_florence_task(image, task)
                        florence_results['object_detection'] = result.get('<OD>', {}) if result else {}
                    elif task == "<DENSE_REGION_CAPTION>":
                        # Dense region caption
                        result = self.florence_base._run_florence_task(image, task)
                        florence_results['dense_regions'] = result.get('<DENSE_REGION_CAPTION>', {}) if result else {}
            
            # Clear some memory before running PromptGen
            if mode_config['promptgen_tasks']:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Run PromptGen v2.0 tasks if any
            if mode_config['promptgen_tasks']:
                logger.info(f"Running PromptGen v2.0 tasks: {mode_config['promptgen_tasks']}")
                self._ensure_florence_promptgen()
                
                for task in mode_config['promptgen_tasks']:
                    if task == "<MORE_DETAILED_CAPTION>":
                        result = self.florence_promptgen.analyze(
                            image, detailed=True, promptgen_instruction=task
                        )
                        promptgen_results['detailed_caption'] = result.get('description', '')
                    elif task == "<GENERATE_TAGS>":
                        # Use _run_florence_task for tags
                        result = self.florence_promptgen._run_florence_task(image, task)
                        if result:
                            # Extract tags from result
                            tags_data = result.get('<GENERATE_TAGS>', '')
                            if isinstance(tags_data, str):
                                promptgen_results['tags'] = {'danbooru': tags_data}
                            else:
                                promptgen_results['tags'] = tags_data
                    elif task == "<ANALYZE>":
                        # Use _run_florence_task for analysis
                        result = self.florence_promptgen._run_florence_task(image, task)
                        if result:
                            analysis_text = result.get('<ANALYZE>', '')
                            promptgen_results['composition_analysis'] = analysis_text
                    elif task == "<MIXED_CAPTION>":
                        # Use _run_florence_task for mixed caption
                        result = self.florence_promptgen._run_florence_task(image, task)
                        if result:
                            # MiaoshouAI compatibility
                            caption = result.get('<MIX_CAPTION>', '') or result.get('<MIXED_CAPTION>', '')
                            promptgen_results['mixed_caption'] = caption
                    elif task == "<MIXED_CAPTION_PLUS>":
                        # Use _run_florence_task for mixed caption plus
                        result = self.florence_promptgen._run_florence_task(image, task)
                        if result:
                            # MiaoshouAI compatibility
                            caption = result.get('<MIX_CAPTION_PLUS>', '') or result.get('<MIXED_CAPTION_PLUS>', '')
                            promptgen_results['mixed_caption_plus'] = caption
            
            # Format results based on mode
            results.update(self._format_mode_results(analysis_mode, florence_results, promptgen_results))
            
            # Add timing
            results['total_analysis_time'] = time.time() - start_time
            results['estimated_time'] = mode_config['estimated_time']
            
            # Automatically unload models to free memory after analysis if enabled
            if self.auto_unload:
                self._schedule_unload()
            
        except Exception as e:
            logger.error(f"Analysis failed in {analysis_mode} mode: {e}")
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
    
    def _format_mode_results(self, mode: str, florence: Dict[str, Any], promptgen: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results based on analysis mode
        """
        if mode == "fast":
            # Fast mode uses only Florence-2 Base with DETAILED_CAPTION
            return {
                'description': florence.get('detailed_caption', ''),
                'brief_description': florence.get('detailed_caption', ''),  # For compatibility
                'tags': {},  # No tags in fast mode
                'analysis_mode': 'fast',
                'mode_display': 'fast_mode',
                'models_used': {
                    'base': 'Florence-2 Base (Microsoft)',
                    'promptgen': None
                }
            }
        
        elif mode == "standard":
            # Standard mode uses CAPTION from Florence and full PromptGen
            return {
                'brief_description': florence.get('caption', ''),  # Using basic CAPTION now
                'description': promptgen.get('detailed_caption', ''),
                'tags': promptgen.get('tags', {}),
                'mixed_caption': promptgen.get('mixed_caption', ''),
                'objects': self._process_object_detection(florence.get('object_detection', {})),
                'analysis_mode': 'dual_model_automatic',  # Keep compatibility
                'mode_display': 'standard_mode'
            }
        
        elif mode == "detailed":
            return {
                'brief_description': florence.get('detailed_caption', ''),
                'description': promptgen.get('detailed_caption', ''),
                'tags': promptgen.get('tags', {}),
                'mixed_caption': promptgen.get('mixed_caption_plus', ''),
                'composition_analysis': promptgen.get('composition_analysis', ''),
                'objects': self._process_object_detection(florence.get('object_detection', {})),
                'dense_regions': florence.get('dense_regions', {}),
                'analysis_mode': 'detailed',
                'mode_display': 'detailed_mode'
            }
        
        elif mode == "tags":
            return {
                'tags': promptgen.get('tags', {}),
                'analysis_mode': 'tags_only',
                'mode_display': 'tags_mode'
            }
        
        elif mode == "composition":
            return {
                'composition_analysis': promptgen.get('composition_analysis', ''),
                'objects': self._process_object_detection(florence.get('object_detection', {})),
                'dense_regions': florence.get('dense_regions', {}),
                'analysis_mode': 'composition',
                'mode_display': 'composition_mode'
            }
        
        else:
            # Fallback to standard
            return self._format_mode_results("standard", florence, promptgen)
    
    def _process_object_detection(self, od_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process object detection results into consistent format"""
        if not od_result:
            return {}
        
        # Extract labels and bboxes
        labels = od_result.get('labels', [])
        bboxes = od_result.get('bboxes', [])
        
        # Create object list
        objects = {
            'all': labels,
            'with_positions': []
        }
        
        # Add position information if available
        for i, label in enumerate(labels):
            if i < len(bboxes):
                bbox = bboxes[i]
                objects['with_positions'].append({
                    'label': label,
                    'bbox': bbox
                })
        
        return objects
    
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