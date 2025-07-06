"""
Florence-2 image analysis module for FluxKontext Smart Assistant.

This module provides comprehensive image analysis using Microsoft's Florence-2
vision-language model, extracting objects, style, environment, and composition details.
"""

# Python 3.10+ compatibility patch
import sys
import collections.abc
if sys.version_info >= (3, 10):
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Iterable = collections.abc.Iterable
    collections.MutableSet = collections.abc.MutableSet
    collections.Callable = collections.abc.Callable


import torch
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from PIL import Image
import numpy as np
from pathlib import Path

logger = logging.getLogger("KontextAssistant.ImageAnalyzer")

# Try to import transformers and Florence-2
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    FLORENCE_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Florence-2 features will be disabled.")
    FLORENCE_AVAILABLE = False


class ImageAnalyzer:
    """
    Analyzes images using Florence-2 to extract structured information
    for FLUX.1 Kontext prompt generation.
    """
    
    def __init__(self, device: Optional[str] = None, use_fp16: bool = True):
        """
        Initialize the image analyzer.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_fp16: Whether to use FP16 for memory efficiency
        """
        self.device = device or self._get_device()
        self.dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32
        
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        logger.info(f"ImageAnalyzer initialized - Device: {self.device}, Dtype: {self.dtype}")
    
    def _get_device(self) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """
        Load Florence-2 model and processor.
        
        Returns:
            Boolean indicating success
        """
        if not FLORENCE_AVAILABLE:
            logger.error("Transformers library not available. Cannot load Florence-2.")
            return False
        
        if self.model_loaded:
            return True
        
        try:
            logger.info("Loading Florence-2 Large model...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self.dtype
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                **model_kwargs
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            self.model_loaded = True
            logger.info("Florence-2 model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {str(e)}")
            self.model = None
            self.processor = None
            return False
    
    def analyze_comprehensive(self, image: Union[Image.Image, str, Path]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of an image.
        
        Args:
            image: PIL Image, path to image, or Path object
            
        Returns:
            Dictionary with structured analysis results
        """
        # Ensure image is PIL Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or path")
        
        # Load model if needed
        if not self.model_loaded:
            if not self.load_model():
                return self._get_fallback_analysis()
        
        try:
            # Perform multiple analysis tasks
            analysis_results = {
                "objects": self._detect_objects(image),
                "style": self._analyze_style(image),
                "environment": self._analyze_environment(image),
                "lighting": self._analyze_lighting(image),
                "composition": self._analyze_composition(image)
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            return self._get_fallback_analysis()
    
    def _run_florence_task(self, image: Image.Image, task: str, 
                          text_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a specific Florence-2 task on the image.
        
        Args:
            image: PIL Image
            task: Task prompt (e.g., '<OD>', '<CAPTION>')
            text_input: Optional text input for tasks that require it
            
        Returns:
            Task results dictionary
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
        
        # Prepare inputs
        if text_input:
            prompt = task + text_input
        else:
            prompt = task
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        # Decode results
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse task results
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer
    
    def _detect_objects(self, image: Image.Image) -> Dict[str, List[str]]:
        """Detect and categorize objects in the image"""
        try:
            # Object detection
            od_results = self._run_florence_task(image, "<OD>")
            
            # Get labels with confidence
            labels = od_results.get("<OD>", {}).get("labels", [])
            
            # Dense region caption for more context
            region_results = self._run_florence_task(image, "<DENSE_REGION_CAPTION>")
            region_labels = region_results.get("<DENSE_REGION_CAPTION>", {}).get("labels", [])
            
            # Categorize objects
            main_objects = []
            secondary_objects = []
            
            # Simple heuristic: larger bounding boxes = main objects
            if "bboxes" in od_results.get("<OD>", {}):
                bboxes = od_results["<OD>"]["bboxes"]
                
                # Calculate areas
                areas = []
                for bbox in bboxes:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    areas.append(area)
                
                # Sort by area
                if areas:
                    sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
                    
                    # Top 3 are main objects
                    for i, idx in enumerate(sorted_indices):
                        if i < 3 and idx < len(labels):
                            main_objects.append(labels[idx])
                        elif idx < len(labels):
                            secondary_objects.append(labels[idx])
            
            # Add unique region descriptions
            for region_label in region_labels[:5]:  # Top 5 regions
                if region_label not in main_objects and region_label not in secondary_objects:
                    secondary_objects.append(region_label)
            
            # Remove duplicates while preserving order
            main_objects = list(dict.fromkeys(main_objects))
            secondary_objects = list(dict.fromkeys(secondary_objects))
            
            return {
                "main": main_objects[:5],  # Limit to top 5
                "secondary": secondary_objects[:10],  # Limit to top 10
                "count": {obj: labels.count(obj) for obj in set(labels)}
            }
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return {"main": ["objects"], "secondary": [], "count": {}}
    
    def _analyze_style(self, image: Image.Image) -> Dict[str, str]:
        """Analyze artistic style and mood"""
        try:
            # Get detailed caption
            caption_result = self._run_florence_task(image, "<DETAILED_CAPTION>")
            detailed_caption = caption_result.get("<DETAILED_CAPTION>", "")
            
            # Get more detailed description
            more_caption = self._run_florence_task(image, "<MORE_DETAILED_CAPTION>")
            more_detailed = more_caption.get("<MORE_DETAILED_CAPTION>", "")
            
            # Extract style information from captions
            style_info = {
                "artistic": "photorealistic",  # Default
                "mood": "neutral",
                "color_palette": ""
            }
            
            # Style keywords
            artistic_styles = {
                "photorealistic": ["photo", "realistic", "real", "photograph"],
                "artistic": ["painting", "art", "artistic", "painted"],
                "cartoon": ["cartoon", "animated", "illustration"],
                "sketch": ["sketch", "drawing", "pencil", "drawn"]
            }
            
            # Mood keywords
            moods = {
                "bright": ["bright", "sunny", "cheerful", "vibrant", "colorful"],
                "dark": ["dark", "gloomy", "moody", "shadowy", "dim"],
                "serene": ["calm", "peaceful", "serene", "tranquil"],
                "dramatic": ["dramatic", "intense", "striking", "bold"]
            }
            
            # Analyze captions for style
            combined_text = (detailed_caption + " " + more_detailed).lower()
            
            for style, keywords in artistic_styles.items():
                if any(keyword in combined_text for keyword in keywords):
                    style_info["artistic"] = style
                    break
            
            for mood, keywords in moods.items():
                if any(keyword in combined_text for keyword in keywords):
                    style_info["mood"] = mood
                    break
            
            # Extract color information
            colors = self._extract_colors_from_text(combined_text)
            if colors:
                style_info["color_palette"] = ", ".join(colors[:3])
            
            return style_info
            
        except Exception as e:
            logger.error(f"Error in style analysis: {str(e)}")
            return {"artistic": "unknown", "mood": "neutral", "color_palette": ""}
    
    def _analyze_environment(self, image: Image.Image) -> Dict[str, str]:
        """Analyze environment and setting"""
        try:
            # Get caption for context
            caption_result = self._run_florence_task(image, "<CAPTION>")
            caption = caption_result.get("<CAPTION>", "")
            
            # Analyze for environment cues
            env_info = {
                "setting": "unknown",
                "time_of_day": "day",
                "weather": "clear",
                "season": "unknown"
            }
            
            caption_lower = caption.lower()
            
            # Setting detection
            settings = {
                "indoor": ["indoor", "inside", "room", "interior", "office", "home"],
                "outdoor": ["outdoor", "outside", "street", "nature", "park", "city"],
                "urban": ["city", "street", "building", "urban", "downtown"],
                "nature": ["forest", "mountain", "beach", "lake", "nature", "tree"]
            }
            
            for setting, keywords in settings.items():
                if any(keyword in caption_lower for keyword in keywords):
                    env_info["setting"] = setting
                    break
            
            # Time detection
            times = {
                "dawn": ["dawn", "sunrise", "morning light", "early morning"],
                "day": ["day", "daylight", "afternoon", "noon", "bright"],
                "sunset": ["sunset", "dusk", "golden hour", "evening light"],
                "night": ["night", "dark", "evening", "nighttime"]
            }
            
            for time, keywords in times.items():
                if any(keyword in caption_lower for keyword in keywords):
                    env_info["time_of_day"] = time
                    break
            
            # Weather detection
            weather_conditions = {
                "clear": ["clear", "sunny", "bright"],
                "cloudy": ["cloudy", "overcast", "gray sky"],
                "rainy": ["rain", "rainy", "wet"],
                "snowy": ["snow", "snowy", "winter"]
            }
            
            for weather, keywords in weather_conditions.items():
                if any(keyword in caption_lower for keyword in keywords):
                    env_info["weather"] = weather
                    break
            
            return env_info
            
        except Exception as e:
            logger.error(f"Error in environment analysis: {str(e)}")
            return {
                "setting": "unknown",
                "time_of_day": "day",
                "weather": "clear",
                "season": "unknown"
            }
    
    def _analyze_lighting(self, image: Image.Image) -> Dict[str, str]:
        """Analyze lighting conditions"""
        try:
            # This is simplified - in a real implementation, 
            # we might analyze the image pixels directly
            
            # Get caption for lighting cues
            caption_result = self._run_florence_task(image, "<CAPTION>")
            caption = caption_result.get("<CAPTION>", "").lower()
            
            lighting_info = {
                "primary_source": "natural",
                "direction": "diffuse",
                "intensity": "moderate",
                "shadows": "soft",
                "contrast": "normal"
            }
            
            # Source detection
            if any(word in caption for word in ["sun", "sunlight", "daylight"]):
                lighting_info["primary_source"] = "sun"
            elif any(word in caption for word in ["lamp", "light", "artificial"]):
                lighting_info["primary_source"] = "artificial"
            
            # Direction detection
            if "backlit" in caption or "silhouette" in caption:
                lighting_info["direction"] = "backlight"
            elif any(word in caption for word in ["side light", "side-lit"]):
                lighting_info["direction"] = "side"
            
            # Intensity
            if any(word in caption for word in ["bright", "harsh", "strong"]):
                lighting_info["intensity"] = "bright"
                lighting_info["shadows"] = "hard"
            elif any(word in caption for word in ["dim", "soft", "gentle"]):
                lighting_info["intensity"] = "soft"
                lighting_info["shadows"] = "soft"
            
            return lighting_info
            
        except Exception as e:
            logger.error(f"Error in lighting analysis: {str(e)}")
            return {
                "primary_source": "unknown",
                "direction": "unknown",
                "intensity": "moderate",
                "shadows": "normal",
                "contrast": "normal"
            }
    
    def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image composition"""
        try:
            # Get object detection results for composition
            od_results = self._run_florence_task(image, "<OD>")
            
            composition_info = {
                "perspective": "eye-level",
                "focal_point": "center",
                "depth_layers": ["foreground", "background"],
                "balance": "balanced"
            }
            
            # Analyze bounding boxes for composition
            if "bboxes" in od_results.get("<OD>", {}):
                bboxes = od_results["<OD>"]["bboxes"]
                if bboxes:
                    # Find the largest/most prominent object
                    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
                    largest_idx = areas.index(max(areas))
                    largest_bbox = bboxes[largest_idx]
                    
                    # Determine focal point
                    center_x = (largest_bbox[0] + largest_bbox[2]) / 2
                    center_y = (largest_bbox[1] + largest_bbox[3]) / 2
                    
                    img_center_x = image.width / 2
                    img_center_y = image.height / 2
                    
                    # Focal point position
                    if abs(center_x - img_center_x) < image.width * 0.2:
                        composition_info["focal_point"] = "center"
                    elif center_x < img_center_x:
                        composition_info["focal_point"] = "left"
                    else:
                        composition_info["focal_point"] = "right"
                    
                    # Perspective based on vertical position
                    if center_y < image.height * 0.3:
                        composition_info["perspective"] = "high-angle"
                    elif center_y > image.height * 0.7:
                        composition_info["perspective"] = "low-angle"
            
            return composition_info
            
        except Exception as e:
            logger.error(f"Error in composition analysis: {str(e)}")
            return {
                "perspective": "eye-level",
                "focal_point": "center",
                "depth_layers": ["foreground", "background"],
                "balance": "balanced"
            }
    
    def _extract_colors_from_text(self, text: str) -> List[str]:
        """Extract color mentions from text"""
        colors = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "gray", "brown", "gold", "silver"
        ]
        
        found_colors = []
        text_lower = text.lower()
        
        for color in colors:
            if color in text_lower:
                found_colors.append(color)
        
        return found_colors
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Return fallback analysis when model is not available"""
        return {
            "objects": {
                "main": ["subject"],
                "secondary": ["background"],
                "count": {}
            },
            "style": {
                "artistic": "photorealistic",
                "mood": "neutral",
                "color_palette": ""
            },
            "environment": {
                "setting": "unknown",
                "time_of_day": "day",
                "weather": "clear",
                "season": "unknown"
            },
            "lighting": {
                "primary_source": "natural",
                "direction": "diffuse",
                "intensity": "moderate",
                "shadows": "soft",
                "contrast": "normal"
            },
            "composition": {
                "perspective": "eye-level",
                "focal_point": "center",
                "depth_layers": ["foreground", "background"],
                "balance": "balanced"
            }
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        
        self.model_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded and memory cleared")



    def analyze(self, image, task_type: str) -> dict:
        """Analyze image (mock implementation)"""
        return {
            'objects': {
                'main': ['object', 'subject'],
                'secondary': ['background', 'elements']
            },
            'style': {
                'artistic': 'photorealistic',
                'mood': 'neutral'
            },
            'environment': {
                'setting': 'indoor/outdoor',
                'time_of_day': 'daytime'
            }
        }

# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = ImageAnalyzer()
    
    # Mock image for testing
    test_image = Image.new('RGB', (512, 512), color='red')
    
    # Analyze
    results = analyzer.analyze_comprehensive(test_image)
    
    print("Analysis results:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)