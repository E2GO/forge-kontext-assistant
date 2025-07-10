"""
Smart Analyzer combining Florence-2 and JoyCaption
"""

import logging
import time
import threading
from typing import Dict, Optional, Any, List
from PIL import Image
import concurrent.futures
import torch

from ka_modules.image_analyzer import ImageAnalyzer
from ka_modules.joycaption_analyzer import JoyCaptionAnalyzer

logger = logging.getLogger(__name__)


class SmartAnalyzer:
    """
    Combines Florence-2 technical analysis with JoyCaption artistic analysis
    Provides comprehensive image understanding for prompt generation
    """
    
    def __init__(self, device='cuda', force_cpu=False, florence_model_type="base"):
        """
        Initialize smart analyzer with both models
        
        Args:
            device: Device for computation
            force_cpu: Force CPU usage
            florence_model_type: Type of Florence-2 model ('base' or 'promptgen_v2')
        """
        self.device = device
        self.force_cpu = force_cpu
        self.florence_model_type = florence_model_type
        
        # Initialize analyzers
        self.florence = None
        self.joycaption = None
        
        # Thread pool for parallel execution
        # Use only 1 worker to prevent memory issues when analyzing multiple images
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Add locks for thread-safe model initialization
        self._florence_lock = threading.Lock()
        self._joycaption_lock = threading.Lock()
        
        logger.info(f"Smart analyzer initialized for device: {device}")
    
    def _ensure_florence(self):
        """Lazy initialize Florence-2"""
        # Check if we need to reload due to model type change
        need_reload = False
        if self.florence is None:
            need_reload = True
        elif hasattr(self.florence, 'model_type') and self.florence.model_type != self.florence_model_type:
            logger.info(f"Florence model type changed from {self.florence.model_type} to {self.florence_model_type}, reloading...")
            need_reload = True
            
        if need_reload:
            with self._florence_lock:
                # Double-check after acquiring lock
                if self.florence is not None and hasattr(self.florence, 'model_type') and self.florence.model_type != self.florence_model_type:
                    # Unload old model first
                    logger.info(f"Unloading Florence-2 model: {self.florence.model_id}")
                    self.florence.unload_model()
                    self.florence = None
                    
                if self.florence is None:
                    logger.info(f"Initializing Florence-2 analyzer with model type: {self.florence_model_type}")
                    self.florence = ImageAnalyzer(
                        device=self.device,
                        force_cpu=self.force_cpu,
                        model_type=self.florence_model_type
                    )
                    logger.info(f"Florence-2 analyzer initialized with {self.florence.model_name}")
                else:
                    # Reset initialization if there was a previous error
                    self.florence.reset_initialization()
    
    def _ensure_joycaption(self):
        """Lazy initialize JoyCaption"""
        if self.joycaption is None:
            with self._joycaption_lock:
                # Double-check after acquiring lock
                if self.joycaption is None:
                    try:
                        logger.info("Initializing JoyCaption analyzer...")
                        self.joycaption = JoyCaptionAnalyzer(
                            device=self.device,
                            force_cpu=self.force_cpu,
                            use_gguf=False  # Use full HuggingFace model
                        )
                    except Exception as e:
                        logger.warning(f"Failed to initialize JoyCaption: {e}")
                        logger.info("JoyCaption will not be available")
                        # Don't use fallback - let user know JoyCaption failed
                        self.joycaption = None
    
    def analyze(self, image: Image.Image, use_florence: bool = True, 
                use_joycaption: bool = False,  # Changed default to False
                promptgen_instruction: str = "<MORE_DETAILED_CAPTION>") -> Dict[str, Any]:
        """
        Perform comprehensive analysis using selected models
        
        Args:
            image: PIL Image to analyze
            use_florence: Use Florence-2 for technical analysis
            use_joycaption: Use JoyCaption for artistic analysis
            promptgen_instruction: PromptGen instruction to use (only for promptgen_v2)
            
        Returns:
            Combined analysis results
        """
        if not use_florence and not use_joycaption:
            return {
                'error': 'Please select at least one analysis method',
                'success': False
            }
        
        start_time = time.time()
        results = {
            'success': True,
            'analyzers_used': {
                'florence2': use_florence,
                'joycaption': use_joycaption
            }
        }
        
        try:
            if use_florence and use_joycaption:
                # Run sequentially to avoid memory issues
                logger.info("Running Florence-2 analysis...")
                florence_result = self._run_florence(image, promptgen_instruction)
                
                # Clear some memory before running JoyCaption
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("Running JoyCaption analysis...")
                joy_result = self._run_joycaption(image)
                
                # Merge results
                results.update(self._merge_results(florence_result, joy_result))
                
            elif use_florence:
                # Florence only
                logger.info(f"Running Florence-only analysis with model type: {self.florence_model_type}")
                florence_result = self._run_florence(image, promptgen_instruction)
                results.update(self._format_florence_only(florence_result))
                
            elif use_joycaption:
                # JoyCaption only
                joy_result = self._run_joycaption(image)
                results.update(self._format_joy_only(joy_result))
            
            # Add timing
            results['total_analysis_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Smart analysis failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _run_florence(self, image: Image.Image, promptgen_instruction: str = "<MORE_DETAILED_CAPTION>") -> Dict[str, Any]:
        """Run Florence-2 analysis"""
        self._ensure_florence()
        # Log which model is being used
        if self.florence:
            logger.info(f"Running analysis with Florence-2 model: {self.florence.model_id}")
            logger.info(f"Florence model type: {self.florence.model_type}")
        else:
            logger.error("Florence analyzer is None!")
            return {}
        
        try:
            # Pass promptgen_instruction if using promptgen_v2
            if self.florence_model_type == "promptgen_v2":
                logger.info(f"Using PromptGen v2.0 with instruction: {promptgen_instruction}")
                result = self.florence.analyze(image, detailed=True, promptgen_instruction=promptgen_instruction)
            else:
                result = self.florence.analyze(image, detailed=True)
            logger.info(f"Florence analysis returned: {list(result.keys()) if result else 'None'}")
            return result
        except Exception as e:
            logger.error(f"Florence analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _run_joycaption(self, image: Image.Image) -> Dict[str, Any]:
        """Run JoyCaption analysis"""
        try:
            self._ensure_joycaption()
            if self.joycaption is None:
                # JoyCaption not available
                logger.warning("JoyCaption is not available")
                return {
                    'success': False,
                    'error': 'JoyCaption not available',
                    'descriptive': 'JoyCaption analysis not available',
                    'categories': {}
                }
            return self.joycaption.analyze(image)
        except Exception as e:
            logger.error(f"JoyCaption analysis failed: {e}")
            # Return minimal valid result
            return {
                'success': False,
                'error': str(e),
                'descriptive': 'JoyCaption analysis failed',
                'categories': {}
            }
    
    def _merge_results(self, florence: Dict[str, Any], joy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently merge results from both analyzers
        
        Florence provides: technical data, positions, OCR
        JoyCaption provides: descriptions, tags, artistic analysis
        """
        # Store both descriptions separately when both models are used
        florence_description = florence.get('description', '') if florence else ''
        joy_description = ''
        
        if joy and joy.get('success', True):
            # Check all possible description fields
            joy_description = (joy.get('descriptive_casual_medium', '') or 
                              joy.get('descriptive_casual', '') or
                              joy.get('descriptive', ''))
        
        # Determine which description(s) to use
        if florence and joy:
            # Both models - keep both descriptions separate
            merged = {
                'description': joy_description or florence_description,  # Primary description
                'florence_description': florence_description,
                'joycaption_description': joy_description,
                'technical_data': {},
                'tags': {},
                'categories': {},
                'style': {},
                'florence_analysis': florence,
                'joy_analysis': joy
            }
        else:
            # Single model - use single description
            merged = {
                'description': joy_description or florence_description,
                'technical_data': {},
                'tags': {},
                'categories': {},
                'style': {},
                'florence_analysis': florence,
                'joy_analysis': joy
            }
        
        # Technical data from Florence
        if florence:
            merged['technical_data'] = {
                'size': florence.get('size', 'unknown'),
                'mode': florence.get('mode', 'unknown'),
                'objects_with_positions': florence.get('objects', {}),
                'text_detected': florence.get('text', []),
                'regions': florence.get('regions', [])
            }
            
            # Extract object counts
            if 'objects' in florence and isinstance(florence['objects'], dict):
                if 'counts' in florence['objects']:
                    merged['technical_data']['object_counts'] = florence['objects']['counts']
        
        # Artistic data from JoyCaption
        if joy:
            # Tags - get from any available field
            tags = (joy.get('booru_tags_medium', '') or
                   joy.get('booru_tags', '') or
                   joy.get('danbooru_tags', ''))
            
            merged['tags'] = {
                'danbooru': tags,
                'general': '',  # No longer used
                'all': self._merge_tags(tags, '')
            }
            
            # Categories from parsed tags
            if 'categories' in joy:
                merged['categories'] = joy['categories']
            
            # Style info (can be enhanced from tags)
            merged['style'] = self._extract_style_info(joy)
        
        # If only Florence-2 is used, use its generated tags
        elif florence and 'tags' in florence:
            florence_tags = florence['tags']
            # Convert Florence tags to similar format
            merged['tags'] = {
                'danbooru': ', '.join(florence_tags.get('general', [])),
                'general': ', '.join(florence_tags.get('general', [])),
                'all': florence_tags.get('general', [])
            }
            
            # Also populate categories from Florence tags
            merged['categories'] = {
                'character': florence_tags.get('character', []),
                'clothing': florence_tags.get('clothing', []),
                'style': florence_tags.get('style', []),
                'environment': florence_tags.get('environment', []),
                'objects': florence_tags.get('objects', []),
                'colors': florence_tags.get('colors', []),
                'composition': florence_tags.get('composition', [])
            }
        
        # Combined object list (Florence technical + JoyCaption semantic)
        merged['objects'] = self._merge_object_lists(florence, joy)
        
        return merged
    
    def _merge_tags(self, danbooru_tags: str, general_tags: str) -> List[str]:
        """Merge tags from different sources, removing duplicates"""
        all_tags = set()
        
        # Parse Danbooru tags
        if danbooru_tags:
            tags = [tag.strip() for tag in danbooru_tags.replace(',', ' ').split()]
            all_tags.update(tags)
        
        # Parse general tags
        if general_tags:
            tags = [tag.strip() for tag in general_tags.split(',')]
            all_tags.update(tags)
        
        # Sort and return
        return sorted(list(all_tags))
    
    def _merge_object_lists(self, florence: Dict, joy: Dict) -> Dict[str, List[str]]:
        """Merge object lists from both analyzers"""
        objects = {
            'florence': [],
            'joycaption': [],
            'combined': []
        }
        
        # Get Florence objects
        if florence and 'objects' in florence:
            if isinstance(florence['objects'], dict) and 'all' in florence['objects']:
                objects['florence'] = florence['objects']['all']
            elif isinstance(florence['objects'], list):
                objects['florence'] = florence['objects']
        
        # Get JoyCaption objects from categories
        if joy and 'categories' in joy:
            if 'characters' in joy['categories']:
                objects['joycaption'].extend(joy['categories']['characters'])
            if 'objects' in joy['categories']:
                objects['joycaption'].extend(joy['categories']['objects'])
        
        # Combine without duplicates
        combined_set = set(objects['florence'] + objects['joycaption'])
        objects['combined'] = sorted(list(combined_set))
        
        return objects
    
    def _extract_style_info(self, joy: Dict) -> Dict[str, Any]:
        """Extract style information from JoyCaption results"""
        style_info = {
            'artistic_style': 'unknown',
            'mood': 'unknown',
            'composition': 'unknown'
        }
        
        if 'categories' in joy:
            categories = joy['categories']
            
            if 'style' in categories and categories['style']:
                style_info['artistic_style'] = ', '.join(categories['style'])
            
            if 'mood' in categories and categories['mood']:
                style_info['mood'] = ', '.join(categories['mood'])
            
            if 'composition' in categories and categories['composition']:
                style_info['composition'] = ', '.join(categories['composition'])
        
        return style_info
    
    def _format_florence_only(self, florence: Dict) -> Dict[str, Any]:
        """Format results when only Florence is used"""
        result = {
            'description': florence.get('description', ''),
            'technical_data': {
                'size': florence.get('size', 'unknown'),
                'mode': florence.get('mode', 'unknown'),
                'objects_with_positions': florence.get('objects', {}),
                'text_detected': florence.get('text', [])
            },
            'objects': florence.get('objects', {}),
            'style': florence.get('style', {}),
            'environment': florence.get('environment', {}),
            'analysis_mode': 'florence_only'
        }
        
        # Preserve important fields from PromptGen v2.0
        if 'model_type' in florence:
            result['model_type'] = florence['model_type']
        if 'tags' in florence:
            result['tags'] = florence['tags']
        if 'mixed_caption' in florence:
            result['mixed_caption'] = florence['mixed_caption']
        if 'composition_analysis' in florence:
            result['composition_analysis'] = florence['composition_analysis']
        
        # IMPORTANT: Preserve the analyzers_used flag
        # This was missing and causing the UI to show wrong model info
        result['analyzers_used'] = {
            'florence2': True,
            'joycaption': False
        }
        
        return result
    
    def _format_joy_only(self, joy: Dict) -> Dict[str, Any]:
        """Format results when only JoyCaption is used"""
        # Get description from any available field
        description = (joy.get('descriptive_casual_medium', '') or 
                      joy.get('descriptive_casual', '') or
                      joy.get('descriptive', ''))
        
        # Get tags from any available field
        tags = (joy.get('booru_tags_medium', '') or
               joy.get('booru_tags', '') or
               joy.get('danbooru_tags', ''))
        
        return {
            'description': description,
            'tags': {
                'danbooru': tags,
                'general': ''  # No longer use general tags
            },
            'categories': joy.get('categories', {}),
            'analysis_mode': 'joycaption_only'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of both analyzers"""
        status = {
            'florence': self.florence.get_status() if self.florence else {'loaded': False},
            'joycaption': self.joycaption.get_status() if self.joycaption else {'loaded': False}
        }
        return status
    
    def unload_models(self):
        """Unload both models to free memory"""
        logger.info("Starting model unload process...")
        
        # Use locks to ensure thread-safe unloading
        with self._florence_lock:
            if self.florence:
                try:
                    self.florence.unload_model()
                except Exception as e:
                    logger.warning(f"Error unloading Florence: {e}")
                finally:
                    self.florence = None
        
        with self._joycaption_lock:
            if self.joycaption:
                try:
                    self.joycaption.unload_model()
                except Exception as e:
                    logger.warning(f"Error unloading JoyCaption: {e}")
                finally:
                    self.joycaption = None
        
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
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.unload_models()
        return False