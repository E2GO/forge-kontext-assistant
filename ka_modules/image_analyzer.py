"""
Florence-2 based image analyzer for Kontext Assistant.
Provides detailed image analysis for prompt generation.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import time

# Compatibility with Python 3.10.17
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

logger = logging.getLogger("KontextAssistant.ImageAnalyzer")


@dataclass
class AnalysisResult:
    """Structured analysis result from Florence-2."""
    objects: Dict[str, Any]
    style: Dict[str, Any]
    environment: Dict[str, Any]
    lighting: Dict[str, Any]
    composition: Dict[str, Any]
    raw_caption: str
    processing_time: float


class Florence2ModelLoader:
    """Manages Florence-2 model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._initialized = False
        
    def initialize(self, device: Optional[str] = None):
        """Lazy load Florence-2 model when needed."""
        if self._initialized:
            return
            
        try:
            logger.info("Loading Florence-2 Large model...")
            start_time = time.time()
            
            # Import here to avoid loading if not needed
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            # Determine device
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
                
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                torch_dtype=torch.float32,  # Using float32 to avoid dtype mismatch
                                trust_remote_code=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True
            )
            
            # Move to device if needed
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self._initialized = True
            load_time = time.time() - start_time
            logger.info(f"Florence-2 loaded successfully in {load_time:.2f}s on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {str(e)}")
            raise RuntimeError(f"Florence-2 initialization failed: {str(e)}")
    
    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Florence-2 model unloaded")
    
    def process_task(self, image: Image.Image, task: str, text_input: Optional[str] = None) -> Dict[str, Any]:
        """Process image with specified Florence-2 task."""
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Prepare inputs
            if text_input and task in ["<CAPTION>", "<DETAILED_CAPTION>"]:
                prompt = f"{task}{text_input}"
            else:
                prompt = task
                
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=3,
                    early_stopping=True
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Parse output
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height)
            )
            
            return parsed
            
        except Exception as e:
            logger.error(f"Task {task} failed: {str(e)}")
            return {}


class ImageAnalyzer:
    """Main image analyzer using Florence-2."""
    
    # Florence-2 task mappings
    TASKS = {
        'caption': '<CAPTION>',
        'detailed_caption': '<DETAILED_CAPTION>',
        'object_detection': '<OD>',
        'dense_region_caption': '<DENSE_REGION_CAPTION>',
        'region_proposal': '<REGION_PROPOSAL>',
        'ocr': '<OCR>',
        'ocr_with_region': '<OCR_WITH_REGION>'
    }
    
    def __init__(self):
        self.model_loader = Florence2ModelLoader()
        self._cache = {}
        self.enable_caching = True
        
    def _get_cache_key(self, image: Image.Image) -> str:
        """Generate cache key for image."""
        import hashlib
        # Simple hash based on image size and mode
        return f"{image.size}_{image.mode}_{hashlib.md5(image.tobytes()).hexdigest()[:8]}"
    
    def analyze(self, image: Image.Image, use_cache: bool = True) -> AnalysisResult:
        """
        Perform comprehensive image analysis using Florence-2.
        
        Args:
            image: PIL Image to analyze
            use_cache: Whether to use cached results
            
        Returns:
            AnalysisResult with structured analysis
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(image)
        if use_cache and self.enable_caching and cache_key in self._cache:
            logger.info("Using cached analysis result")
            return self._cache[cache_key]
        
        # Ensure model is loaded
        if not self.model_loader._initialized:
            self.model_loader.initialize()
        
        try:
            # 1. Get detailed caption for overall understanding
            caption_result = self.model_loader.process_task(image, self.TASKS['detailed_caption'])
            detailed_caption = caption_result.get(self.TASKS['detailed_caption'], "")
            
            # 2. Detect objects
            od_result = self.model_loader.process_task(image, self.TASKS['object_detection'])
            
            # 3. Get dense region captions for better understanding
            dense_result = self.model_loader.process_task(image, self.TASKS['dense_region_caption'])
            
            # Process results into structured format
            analysis = self._structure_analysis(
                detailed_caption,
                od_result,
                dense_result
            )
            
            # Add raw caption and timing
            analysis.raw_caption = detailed_caption
            analysis.processing_time = time.time() - start_time
            
            # Cache result
            if self.enable_caching:
                self._cache[cache_key] = analysis
            
            logger.info(f"Analysis completed in {analysis.processing_time:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            # Return basic analysis on failure
            return self._get_fallback_analysis(str(e))
    
    def _structure_analysis(self, caption: str, od_result: Dict, dense_result: Dict) -> AnalysisResult:
        """Structure raw Florence-2 outputs into organized analysis."""
        
        # Extract objects from detection results
        objects = self._extract_objects(od_result)
        
        # Analyze style from caption
        style = self._analyze_style(caption)
        
        # Extract environment details
        environment = self._analyze_environment(caption, dense_result)
        
        # Infer lighting conditions
        lighting = self._analyze_lighting(caption)
        
        # Analyze composition
        composition = self._analyze_composition(od_result, dense_result)
        
        return AnalysisResult(
            objects=objects,
            style=style,
            environment=environment,
            lighting=lighting,
            composition=composition,
            raw_caption=caption,
            processing_time=0.0  # Will be set by caller
        )
    
    def _extract_objects(self, od_result: Dict) -> Dict[str, Any]:
        """Extract and categorize detected objects."""
        objects = {
            "main": [],
            "secondary": [],
            "count": {},
            "all_detected": []
        }
        
        if not od_result or self.TASKS['object_detection'] not in od_result:
            return objects
        
        detections = od_result[self.TASKS['object_detection']]
        if 'labels' in detections and 'bboxes' in detections:
            labels = detections['labels']
            bboxes = detections['bboxes']
            
            # Count occurrences
            for label in labels:
                objects['count'][label] = objects['count'].get(label, 0) + 1
                objects['all_detected'].append(label)
            
            # Categorize by size/position (larger/centered = main)
            if bboxes:
                bbox_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes]
                sorted_indices = sorted(range(len(bbox_sizes)), key=lambda i: bbox_sizes[i], reverse=True)
                
                # Top 2-3 largest objects as main
                for i in sorted_indices[:min(3, len(sorted_indices))]:
                    objects['main'].append(labels[i])
                
                # Rest as secondary
                for i in sorted_indices[3:]:
                    objects['secondary'].append(labels[i])
        
        return objects
    
    def _analyze_style(self, caption: str) -> Dict[str, Any]:
        """Extract style information from caption."""
        style = {
            "artistic": "photorealistic",  # default
            "mood": "neutral",
            "color_palette": {
                "dominant": [],
                "temperature": "neutral"
            }
        }
        
        # Style keywords
        if any(word in caption.lower() for word in ['painting', 'artistic', 'abstract', 'drawn']):
            style['artistic'] = 'artistic'
        elif any(word in caption.lower() for word in ['photo', 'realistic', 'photograph']):
            style['artistic'] = 'photorealistic'
        elif any(word in caption.lower() for word in ['cartoon', 'anime', 'illustration']):
            style['artistic'] = 'illustration'
        
        # Mood analysis
        if any(word in caption.lower() for word in ['bright', 'cheerful', 'vibrant', 'sunny']):
            style['mood'] = 'bright, energetic'
        elif any(word in caption.lower() for word in ['dark', 'moody', 'dramatic', 'somber']):
            style['mood'] = 'dark, dramatic'
        elif any(word in caption.lower() for word in ['calm', 'serene', 'peaceful', 'tranquil']):
            style['mood'] = 'calm, serene'
        
        # Color analysis
        color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
                      'black', 'white', 'gray', 'brown', 'golden', 'silver']
        detected_colors = [color for color in color_words if color in caption.lower()]
        if detected_colors:
            style['color_palette']['dominant'] = detected_colors[:3]
        
        # Temperature
        if any(word in caption.lower() for word in ['warm', 'sunset', 'sunrise', 'golden']):
            style['color_palette']['temperature'] = 'warm'
        elif any(word in caption.lower() for word in ['cool', 'cold', 'blue', 'icy']):
            style['color_palette']['temperature'] = 'cool'
        
        return style
    
    def _analyze_environment(self, caption: str, dense_result: Dict) -> Dict[str, Any]:
        """Extract environment and setting information."""
        environment = {
            "setting": "unknown",
            "time_of_day": "unknown",
            "weather": "clear",
            "season": "unknown"
        }
        
        # Setting detection
        settings = {
            'indoor': ['room', 'indoor', 'inside', 'interior', 'house', 'building'],
            'outdoor': ['outdoor', 'outside', 'landscape', 'nature', 'street', 'park'],
            'urban': ['city', 'urban', 'street', 'building', 'downtown'],
            'nature': ['forest', 'mountain', 'beach', 'lake', 'desert', 'field']
        }
        
        for setting_type, keywords in settings.items():
            if any(word in caption.lower() for word in keywords):
                environment['setting'] = setting_type
                break
        
        # Time of day
        if any(word in caption.lower() for word in ['morning', 'sunrise', 'dawn']):
            environment['time_of_day'] = 'morning'
        elif any(word in caption.lower() for word in ['afternoon', 'midday', 'noon']):
            environment['time_of_day'] = 'afternoon'
        elif any(word in caption.lower() for word in ['evening', 'sunset', 'dusk']):
            environment['time_of_day'] = 'evening'
        elif any(word in caption.lower() for word in ['night', 'dark', 'midnight']):
            environment['time_of_day'] = 'night'
        
        # Weather
        weather_conditions = ['rain', 'snow', 'fog', 'cloudy', 'sunny', 'storm']
        for condition in weather_conditions:
            if condition in caption.lower():
                environment['weather'] = condition
                break
        
        # Season
        seasons = ['spring', 'summer', 'autumn', 'fall', 'winter']
        for season in seasons:
            if season in caption.lower():
                environment['season'] = season
                break
        
        return environment
    
    def _analyze_lighting(self, caption: str) -> Dict[str, Any]:
        """Analyze lighting conditions from caption."""
        lighting = {
            "primary_source": "natural",
            "direction": "ambient",
            "intensity": "medium",
            "shadows": "soft",
            "contrast": "medium"
        }
        
        # Light source
        if any(word in caption.lower() for word in ['lamp', 'light', 'bulb', 'neon']):
            lighting['primary_source'] = 'artificial'
        elif any(word in caption.lower() for word in ['sun', 'daylight', 'sunlight']):
            lighting['primary_source'] = 'sun'
        
        # Intensity
        if any(word in caption.lower() for word in ['bright', 'intense', 'glaring']):
            lighting['intensity'] = 'bright'
        elif any(word in caption.lower() for word in ['dim', 'dark', 'low light']):
            lighting['intensity'] = 'low'
        
        # Shadows
        if any(word in caption.lower() for word in ['harsh', 'hard shadow', 'stark']):
            lighting['shadows'] = 'hard'
        elif any(word in caption.lower() for word in ['soft', 'diffused', 'gentle']):
            lighting['shadows'] = 'soft'
        
        # Contrast
        if any(word in caption.lower() for word in ['high contrast', 'dramatic']):
            lighting['contrast'] = 'high'
        elif any(word in caption.lower() for word in ['low contrast', 'flat']):
            lighting['contrast'] = 'low'
        
        return lighting
    
    def _analyze_composition(self, od_result: Dict, dense_result: Dict) -> Dict[str, Any]:
        """Analyze image composition."""
        composition = {
            "perspective": "eye-level",
            "focal_point": "center",
            "depth_layers": ["foreground", "background"],
            "balance": "balanced"
        }
        
        # Get main objects for focal point
        if od_result and self.TASKS['object_detection'] in od_result:
            detections = od_result[self.TASKS['object_detection']]
            if 'labels' in detections and detections['labels']:
                composition['focal_point'] = detections['labels'][0]
        
        # Analyze depth from dense captions
        if dense_result and self.TASKS['dense_region_caption'] in dense_result:
            regions = dense_result[self.TASKS['dense_region_caption']]
            if 'labels' in regions:
                # More regions suggest more depth layers
                if len(regions['labels']) > 3:
                    composition['depth_layers'] = ["foreground", "midground", "background"]
        
        return composition
    
    def _get_fallback_analysis(self, error_msg: str) -> AnalysisResult:
        """Return basic analysis when Florence-2 fails."""
        return AnalysisResult(
            objects={
                "main": ["unknown"],
                "secondary": [],
                "count": {},
                "all_detected": []
            },
            style={
                "artistic": "unknown",
                "mood": "neutral",
                "color_palette": {"dominant": [], "temperature": "neutral"}
            },
            environment={
                "setting": "unknown",
                "time_of_day": "unknown",
                "weather": "unknown",
                "season": "unknown"
            },
            lighting={
                "primary_source": "unknown",
                "direction": "ambient",
                "intensity": "medium",
                "shadows": "soft",
                "contrast": "medium"
            },
            composition={
                "perspective": "eye-level",
                "focal_point": "unknown",
                "depth_layers": ["foreground", "background"],
                "balance": "balanced"
            },
            raw_caption=f"Analysis failed: {error_msg}",
            processing_time=0.0
        )
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()
        logger.info("Analysis cache cleared")
    
    def unload_model(self):
        """Unload Florence-2 model to free memory."""
        self.model_loader.unload()
        self.clear_cache()