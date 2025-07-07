"""
Image analysis module for Kontext Assistant
Supports both mock mode and real Florence-2 analysis
"""

# Fix collections compatibility BEFORE any other imports
import collections
import collections.abc
for attr in dir(collections.abc):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

import logging
import torch
import time
import gc
import os
import random
from typing import Dict, Any, Optional, List
from PIL import Image
from pathlib import Path

logger = logging.getLogger("ImageAnalyzer")

# Check if we should use Florence-2
USE_FLORENCE2 = os.environ.get("KONTEXT_USE_FLORENCE2", "auto").lower()

# Try to import required libraries
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("transformers not available - using mock mode")


class Florence2Analyzer:
    """Real Florence-2 based image analyzer"""
    
    def __init__(self, device: Optional[str] = None, load_in_8bit: bool = False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
        self.model_id = "microsoft/Florence-2-large"
        self.initialized = False
        
        logger.info(f"Florence2Analyzer initialized for device: {self.device}")
    
    def load_model(self, force_reload: bool = False):
        """Load Florence-2 model and processor"""
        if self.initialized and not force_reload:
            logger.info("Model already loaded")
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available!")
            return
        
        try:
            logger.info(f"Loading Florence-2 model: {self.model_id}")
            start_time = time.time()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with memory optimization
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.load_in_8bit and self.device == "cuda":
                model_kwargs["load_in_8bit"] = True
                logger.info("Loading model in 8-bit for memory efficiency")
            
            # Load model without device_map for Florence-2
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Move model to device manually
            if self.device == "cuda":
                if not self.load_in_8bit:
                    self.model = self.model.half()  # Use half precision
                self.model = self.model.to(self.device)
                logger.info(f"Model moved to {self.device}")
            
            # Ensure model is in eval mode
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU memory used: {memory_used:.2f} GB")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            self.initialized = False
            raise
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            logger.info("Unloading Florence-2 model")
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.initialized = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model unloaded and memory cleared")
    
    def analyze(self, image: Image.Image, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze image using Florence-2
        
        Args:
            image: PIL Image to analyze
            tasks: List of tasks to perform (if None, performs default set)
        
        Returns:
            Dictionary with structured analysis results
        """
        if not self.initialized:
            logger.warning("Model not loaded, using fallback analysis")
            return self._fallback_analysis(image)
        
        if tasks is None:
            tasks = ["<OD>", "<CAPTION>", "<DETAILED_CAPTION>"]
        
        results = {}
        
        try:
            # Basic image info
            results["size"] = f"{image.width}x{image.height}"
            results["aspect_ratio"] = round(image.width / image.height, 2)
            results["mode"] = image.mode
            
            # Run Florence-2 tasks
            for task in tasks:
                task_result = self._run_task(image, task)
                if task_result:
                    results[task] = task_result
            
            # Convert to structured format
            structured = self._structure_results(results, image)
            return structured
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return self._fallback_analysis(image)
    
    def _run_task(self, image: Image.Image, task: str) -> Any:
        """Run a specific Florence-2 task"""
        try:
            inputs = self.processor(
                text=task,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device and ensure correct dtype
            if self.device == "cuda":
                # Convert inputs to the same dtype as model
                model_dtype = next(self.model.parameters()).dtype
                inputs = {
                    k: v.to(self.device).to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                    else v.to(self.device) if isinstance(v, torch.Tensor)
                    else v
                    for k, v in inputs.items()
                }
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )[0]
            
            # Post-process
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height)
            )
            
            return parsed[task]
            
        except Exception as e:
            logger.error(f"Error running task {task}: {e}")
            return None
    
    def _structure_results(self, raw_results: Dict, image: Image.Image) -> Dict[str, Any]:
        """Convert Florence-2 results to structured format"""
        structured = {
            "size": raw_results.get("size", f"{image.width}x{image.height}"),
            "aspect_ratio": raw_results.get("aspect_ratio", round(image.width / image.height, 2)),
            "mode": raw_results.get("mode", image.mode),
            "description": "",
            "objects": {"main": [], "secondary": [], "count": {}},
            "style": {"artistic": "unknown", "mood": "neutral", "color_palette": {"dominant": [], "temperature": "neutral"}},
            "environment": {"setting": "unknown", "time_of_day": "unknown", "weather": "unknown", "season": "unknown"},
            "lighting": {"primary_source": "unknown", "direction": "unknown", "intensity": "unknown", "shadows": "unknown", "contrast": "unknown"},
            "composition": {"perspective": "unknown", "focal_point": "unknown", "depth_layers": [], "balance": "unknown"}
        }
        
        # Extract from CAPTION
        if "<CAPTION>" in raw_results:
            structured["description"] = raw_results["<CAPTION>"]
        
        # Extract from DETAILED_CAPTION
        if "<DETAILED_CAPTION>" in raw_results:
            detailed = raw_results["<DETAILED_CAPTION>"]
            structured["description"] = detailed
            
            # Parse style hints
            if "painting" in detailed.lower() or "artistic" in detailed.lower():
                structured["style"]["artistic"] = "painterly"
            elif "photo" in detailed.lower() or "realistic" in detailed.lower():
                structured["style"]["artistic"] = "photorealistic"
            
            # Parse mood
            mood_words = {
                "bright": "bright", "dark": "moody", "colorful": "vibrant",
                "serene": "calm", "dramatic": "dramatic"
            }
            for word, mood in mood_words.items():
                if word in detailed.lower():
                    structured["style"]["mood"] = mood
                    break
        
        # Extract from OD (Object Detection)
        if "<OD>" in raw_results:
            od_result = raw_results["<OD>"]
            if "bboxes" in od_result and "labels" in od_result:
                labels = od_result["labels"]
                
                # Count objects
                label_counts = {}
                for label in labels:
                    label_lower = label.lower()
                    label_counts[label_lower] = label_counts.get(label_lower, 0) + 1
                
                structured["objects"]["count"] = label_counts
                
                # Separate main and secondary objects
                # Main objects are those with larger bounding boxes or more instances
                sorted_labels = sorted(set(labels), key=lambda x: label_counts.get(x.lower(), 0), reverse=True)
                
                if sorted_labels:
                    structured["objects"]["main"] = sorted_labels[:3]
                    structured["objects"]["secondary"] = sorted_labels[3:8]
        
        # Infer environment and lighting from description
        if structured["description"]:
            desc_lower = structured["description"].lower()
            
            # Environment
            if any(word in desc_lower for word in ["indoor", "room", "interior"]):
                structured["environment"]["setting"] = "indoor"
            elif any(word in desc_lower for word in ["outdoor", "outside", "street", "nature"]):
                structured["environment"]["setting"] = "outdoor"
            
            # Time of day
            time_words = {
                "morning": "morning", "sunrise": "morning",
                "afternoon": "afternoon", "midday": "afternoon",
                "evening": "evening", "sunset": "evening",
                "night": "night", "dark": "night"
            }
            for word, time in time_words.items():
                if word in desc_lower:
                    structured["environment"]["time_of_day"] = time
                    break
            
            # Weather
            weather_words = {
                "sunny": "sunny", "clear": "clear",
                "cloudy": "cloudy", "overcast": "cloudy",
                "rain": "rainy", "snow": "snowy"
            }
            for word, weather in weather_words.items():
                if word in desc_lower:
                    structured["environment"]["weather"] = weather
                    break
        
        # Extract dominant colors from description
        color_words = ["red", "blue", "green", "yellow", "orange", "purple", "pink", 
                      "brown", "black", "white", "gray", "beige"]
        found_colors = []
        desc_lower = structured["description"].lower()
        for color in color_words:
            if color in desc_lower:
                found_colors.append(color)
        
        if found_colors:
            structured["style"]["color_palette"]["dominant"] = found_colors[:4]
            
            # Determine temperature
            warm_colors = ["red", "orange", "yellow", "pink"]
            cool_colors = ["blue", "green", "purple"]
            warm_count = sum(1 for c in found_colors if c in warm_colors)
            cool_count = sum(1 for c in found_colors if c in cool_colors)
            
            if warm_count > cool_count:
                structured["style"]["color_palette"]["temperature"] = "warm"
            elif cool_count > warm_count:
                structured["style"]["color_palette"]["temperature"] = "cool"
        
        return structured
    
    def _fallback_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback analysis when model is not available"""
        return MockImageAnalyzer()._create_mock_analysis(image)


class MockImageAnalyzer:
    """Mock analyzer for testing and fallback"""
    
    def __init__(self):
        self.initialized = False
        self.model = None
        logger.info("MockImageAnalyzer initialized")
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image and return structured information"""
        return self._create_mock_analysis(image)
    
    def _create_mock_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Create mock analysis data"""
        if image is None:
            return self._get_empty_analysis()
        
        # Get basic image info
        width, height = image.size
        aspect_ratio = width / height
        
        # Mock analysis with realistic structure
        analysis = {
            "size": f"{width}x{height}",
            "aspect_ratio": round(aspect_ratio, 2),
            "mode": image.mode,
            "description": "Mock analysis - Florence-2 integration available",
            
            "objects": {
                "main": self._get_mock_main_objects(),
                "secondary": self._get_mock_secondary_objects(),
                "count": {
                    "person": random.randint(0, 3),
                    "vehicle": random.randint(0, 2),
                    "animal": random.randint(0, 1),
                    "furniture": random.randint(1, 4)
                }
            },
            
            "style": {
                "artistic": random.choice(["photorealistic", "painterly", "sketch", "digital art"]),
                "mood": random.choice(["bright", "moody", "neutral", "dramatic"]),
                "color_palette": {
                    "dominant": self._get_mock_colors(),
                    "temperature": random.choice(["warm", "cool", "neutral"])
                }
            },
            
            "environment": {
                "setting": random.choice(["indoor", "outdoor", "urban", "natural", "studio"]),
                "time_of_day": random.choice(["morning", "afternoon", "evening", "night"]),
                "weather": random.choice(["clear", "cloudy", "rainy", "foggy"]),
                "season": random.choice(["spring", "summer", "fall", "winter"])
            },
            
            "lighting": {
                "primary_source": random.choice(["natural sunlight", "artificial", "mixed"]),
                "direction": random.choice(["front", "side", "back", "top"]),
                "intensity": random.choice(["bright", "moderate", "dim"]),
                "shadows": random.choice(["soft", "hard", "minimal"]),
                "contrast": random.choice(["high", "medium", "low"])
            },
            
            "composition": {
                "perspective": random.choice(["eye-level", "low-angle", "high-angle", "bird's-eye"]),
                "focal_point": "center",
                "depth_layers": ["foreground", "midground", "background"],
                "balance": random.choice(["centered", "rule-of-thirds", "asymmetric"])
            }
        }
        
        return analysis
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            "size": "0x0",
            "aspect_ratio": 0,
            "mode": "unknown",
            "description": "No image provided",
            "objects": {"main": [], "secondary": [], "count": {}},
            "style": {"artistic": "unknown", "mood": "neutral", "color_palette": {"dominant": [], "temperature": "neutral"}},
            "environment": {"setting": "unknown", "time_of_day": "unknown", "weather": "unknown", "season": "unknown"},
            "lighting": {"primary_source": "unknown", "direction": "unknown", "intensity": "unknown", "shadows": "unknown", "contrast": "unknown"},
            "composition": {"perspective": "unknown", "focal_point": "unknown", "depth_layers": [], "balance": "unknown"}
        }
    
    def _get_mock_main_objects(self) -> list:
        """Get mock main objects"""
        possible_objects = [
            "person", "car", "building", "tree", "dog", "cat",
            "chair", "table", "laptop", "phone", "book", "cup"
        ]
        num_objects = random.randint(1, 3)
        return random.sample(possible_objects, num_objects)
    
    def _get_mock_secondary_objects(self) -> list:
        """Get mock secondary objects"""
        possible_objects = [
            "window", "door", "lamp", "plant", "picture frame",
            "rug", "curtain", "shelf", "clock", "vase"
        ]
        num_objects = random.randint(2, 5)
        return random.sample(possible_objects, num_objects)
    
    def _get_mock_colors(self) -> list:
        """Get mock dominant colors"""
        colors = [
            "red", "blue", "green", "yellow", "orange", "purple",
            "brown", "gray", "black", "white", "pink", "beige"
        ]
        num_colors = random.randint(2, 4)
        return random.sample(colors, num_colors)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load model (placeholder for compatibility)"""
        logger.info("Mock mode - no model to load")
        self.initialized = True
    
    def unload_model(self):
        """Unload model to free memory"""
        logger.info("Mock mode - no model to unload")
        self.initialized = False
        self.model = None


# Main ImageAnalyzer class that switches between implementations
class ImageAnalyzer:
    """Image analyzer with automatic Florence-2/mock switching"""
    
    def __init__(self):
        self.initialized = False
        self.use_florence2 = False
        self.analyzer = None
        
        # Determine which analyzer to use
        if USE_FLORENCE2 == "false":
            self.use_florence2 = False
        elif USE_FLORENCE2 == "true" and TRANSFORMERS_AVAILABLE:
            self.use_florence2 = True
        elif USE_FLORENCE2 == "auto":
            # Auto mode - use Florence-2 if available and system has enough resources
            if TRANSFORMERS_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        # Check VRAM
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        if vram_gb >= 4:  # Need at least 4GB for comfortable operation
                            self.use_florence2 = True
                            logger.info(f"Auto mode: Using Florence-2 (VRAM: {vram_gb:.1f}GB)")
                        else:
                            logger.info(f"Auto mode: Insufficient VRAM ({vram_gb:.1f}GB), using mock")
                    else:
                        logger.info("Auto mode: No CUDA available, using mock")
                except:
                    logger.info("Auto mode: Could not check VRAM, using mock")
        
        # Initialize the appropriate analyzer
        if self.use_florence2:
            try:
                self.analyzer = Florence2Analyzer()
                logger.info("Using Florence-2 analyzer")
            except Exception as e:
                logger.warning(f"Failed to initialize Florence-2: {e}")
                self.use_florence2 = False
        
        if not self.use_florence2:
            self.analyzer = MockImageAnalyzer()
            logger.info("Using mock analyzer")
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image using the selected analyzer"""
        if self.analyzer:
            return self.analyzer.analyze(image)
        else:
            # Fallback
            mock = MockImageAnalyzer()
            return mock.analyze(image)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load model if using Florence-2"""
        if self.analyzer and hasattr(self.analyzer, 'load_model'):
            self.analyzer.load_model(model_path)
        self.initialized = True
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.analyzer and hasattr(self.analyzer, 'unload_model'):
            self.analyzer.unload_model()
        self.initialized = False