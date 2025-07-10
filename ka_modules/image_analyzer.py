"""
Smart Image Analyzer with automatic GPU compatibility detection
Supports RTX 4090/5090 with automatic fallback to mock mode
"""

import logging
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import torch
from pathlib import Path
import json
import time
from functools import lru_cache

# Fix for Python 3.10 collections compatibility
import collections
import collections.abc
for attr_name in dir(collections.abc):
    attr = getattr(collections.abc, attr_name)
    if not hasattr(collections, attr_name):
        setattr(collections, attr_name, attr)

# Try to import transformers for Florence-2
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
# Configure logging
logger = logging.getLogger(__name__)

class SmartImageAnalyzer:
    """
    Smart analyzer that automatically handles GPU compatibility issues
    Falls back to mock mode when Florence-2 fails
    """
    
    def __init__(self, device: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize smart image analyzer
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            force_cpu: Force CPU mode for compatibility
        """
        self.model = None
        self.processor = None
        self.device = device
        self.force_cpu = force_cpu
        # Mock mode removed - always use real analysis
        self._initialized = False
        self._init_attempted = False
        self._init_error = None
        
        # Model configuration
        self.model_id = "microsoft/Florence-2-large"
        self.cache_dir = Path.home() / ".cache" / "kontext_assistant"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detection settings
        self.gpu_compatibility_mode = False
        self.detected_gpu = None
        
        self._detect_gpu_compatibility()
    
    def _detect_gpu_compatibility(self):
        """Detect GPU and set compatibility mode"""
        if not torch.cuda.is_available():
            logger.info("No CUDA GPU detected, will use CPU mode")
            return
            
        try:
            gpu_name = torch.cuda.get_device_name(0)
            self.detected_gpu = gpu_name
            logger.info(f"Detected GPU: {gpu_name}")
            
            # Check for problematic GPUs
            problematic_gpus = ["RTX 4090", "RTX 5090", "4090", "5090"]
            if any(gpu in gpu_name for gpu in problematic_gpus):
                logger.warning(f"{gpu_name} detected - enabling compatibility mode")
                self.gpu_compatibility_mode = True
                
                # Force CPU for these GPUs unless explicitly set
                if self.device is None and not self.force_cpu:
                    logger.info("Auto-enabling CPU mode for compatibility")
                    self.force_cpu = True
                    
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")
    
    # Method _ensure_initialized:

    def _ensure_initialized(self, progress_callback=None):
        """Lazy loading of Florence-2 model with automatic fallback"""
        if self._initialized:
            return
            
        if self._init_attempted and self._init_error:
            # Don't retry if we already failed
            raise RuntimeError(f"Previous init failed: {self._init_error}")
            
        self._init_attempted = True
        
        if not TRANSFORMERS_AVAILABLE:
            self._init_error = "Transformers not available"
            logger.error("Transformers not installed")
            raise RuntimeError("Transformers library is required for Florence-2")
            
        try:
            logger.info("Attempting to load Florence-2 model...")
            start_time = time.time()
            
            # Determine device
            if self.device is None:
                if self.force_cpu or self.gpu_compatibility_mode:
                    self.device = "cpu"
                    logger.info("Using CPU mode for compatibility")
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if progress_callback:
                progress_callback("Loading Florence-2 model...", 0.1)
            
            # Load processor
            if progress_callback:
                progress_callback("Loading processor...", 0.3)
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Load model with appropriate dtype
            if progress_callback:
                progress_callback("Loading model weights...", 0.5)
            
            if self.device == "cuda":
                # Try different dtypes for GPU - adding bfloat16
                dtypes_to_try = [torch.float32, torch.float16, torch.bfloat16]
                
                for dtype in dtypes_to_try:
                    try:
                        logger.info(f"Trying to load model with dtype {dtype}")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            torch_dtype=dtype,
                            trust_remote_code=True,
                            cache_dir=self.cache_dir
                        ).to(self.device)
                        self.model.eval()
                        
                        # Test inference to ensure it works
                        self._test_inference()
                        
                        logger.info(f"Model loaded successfully with {dtype}")
                        break
                        
                    except RuntimeError as e:
                        logger.warning(f"Failed with {dtype}: {e}")
                        if dtype == dtypes_to_try[-1]:
                            raise
                        continue
            else:
                # CPU mode - always use float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    cache_dir=self.cache_dir
                )
                self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Florence-2 loaded successfully in {load_time:.1f}s on {self.device}")
            
            if progress_callback:
                progress_callback("Model loaded successfully!", 1.0)
            
            self._initialized = True
            
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"Failed to load Florence-2: {e}")
            # No fallback to mock mode anymore
            raise
            
            # Clean up partial loads
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    def _real_analyze(self, image: Image.Image, detailed: bool) -> Dict[str, Any]:
        """Perform real Florence-2 analysis with timing"""
        import time
        start_time = time.time()
        
        analysis = {}
        
        # Basic image info
        analysis['size'] = f"{image.width}x{image.height}"
        analysis['mode'] = image.mode
        analysis['analysis_mode'] = 'florence2'
        
        # Get detailed caption
        caption_result = self._run_florence_task(image, "<DETAILED_CAPTION>")
        if caption_result:
            raw_description = caption_result.get('<DETAILED_CAPTION>', 'No description available')
            analysis['description'] = self._clean_description(raw_description)
        
        # Get objects with bounding boxes
        od_result = self._run_florence_task(image, "<OD>")
        if od_result and '<OD>' in od_result:
            objects_data = od_result['<OD>']
            analysis['objects'] = self._process_objects(objects_data)
        
        # Analyze regions for composition
        if detailed:
            region_result = self._run_florence_task(image, "<DENSE_REGION_CAPTION>")
            if region_result and '<DENSE_REGION_CAPTION>' in region_result:
                analysis['regions'] = region_result['<DENSE_REGION_CAPTION>']
        
        # Get detailed objects analysis
        detailed_objects = self._extract_detailed_objects(
            analysis.get('description', ''), 
            objects_data if 'od_result' in locals() and od_result else {}
        )
        
        # Update objects with detailed categories
        if detailed_objects:
            analysis['objects_detailed'] = detailed_objects
            # Keep backward compatibility
            if not analysis.get('objects'):
                analysis['objects'] = {
                    'main': detailed_objects.get('main', []),
                    'all': detailed_objects.get('main', [])
                }
        
        # Extract style characteristics
        analysis['style'] = self._extract_style_info(analysis.get('description', ''))
        
        # Environment info
        analysis['environment'] = self._extract_enhanced_environment(analysis.get('description', ''), analysis.get('regions'))
        
        # Generate tags similar to JoyCaption - DISABLED for performance
        # Tags are now only generated when specifically needed
        # analysis['tags'] = self._generate_tags_from_analysis(analysis)
        
        # Add timing info
        analysis['analysis_time'] = time.time() - start_time
        logger.info(f"Analysis completed in {analysis['analysis_time']:.2f} seconds")
        
        return analysis
    def _test_inference(self):
        """Test inference to ensure model works"""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (224, 224), color='white')
            inputs = self.processor(text="<CAPTION>", images=test_image, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                # Ensure dtype compatibility
                if 'pixel_values' in inputs:
                    model_dtype = next(self.model.parameters()).dtype
                    inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model_dtype)
            
            with torch.no_grad():
                # Very short generation for testing
                self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=5
                )
            
            logger.info("Test inference successful")
            
        except Exception as e:
            logger.error(f"Test inference failed: {e}")
            raise
    
    def analyze(self, image: Image.Image, detailed: bool = True) -> Dict[str, Any]:
        """
        Analyze image with automatic fallback to mock if needed
        """
        # Ensure model is loaded
        self._ensure_initialized()
        
        try:
            # Real analysis only
            return self._real_analyze(image, detailed)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _run_florence_task(self, image: Image.Image, task: str) -> Optional[Dict]:
        """Run a specific Florence-2 task with proper error handling"""
        try:
            inputs = self.processor(text=task, images=image, return_tensors="pt")
            
            # Move to device and ensure dtype compatibility
            if self.device == "cuda":
                device_inputs = {}
                model_dtype = next(self.model.parameters()).dtype
                
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                        if k == 'pixel_values':
                            v = v.to(dtype=model_dtype)
                        device_inputs[k] = v
                    else:
                        device_inputs[k] = v
                inputs = device_inputs
            
            with torch.no_grad():
                # Deterministic parameters for accuracy
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,     # Enough for full description
                    min_new_tokens=20,       # Minimum for basic description
                    do_sample=False,         # Disable randomness for accuracy
                    num_beams=5,            # More beams for better quality
                    repetition_penalty=1.1,  # Small penalty for repetitions
                    length_penalty=1.0,      # Neutral length
                    early_stopping=True      # Stop when complete
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height)
            )
            
            return parsed_answer
            
        except Exception as e:
            logger.error(f"Error in Florence task {task}: {e}")
            return None
    
    def _clean_description(self, description: str) -> str:
        """Clean description from common artifacts"""
        if not description:
            return description
        
        # List of phrases to remove
        artifacts_to_remove = [
            "ready to be downloaded",
            "download for free",
            "free download",
            "stock photo",
            "watermark",
            "shutterstock",
            "getty images",
            "©",
            "copyright",
            "all rights reserved",
            "illustration of",
            "photo of",
            "image of",
            "picture of",
            "rendering of",
            "3d render of",
            "digital art of"
        ]
        
        # Clean the description
        cleaned = description
        for artifact in artifacts_to_remove:
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(artifact), re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)
        
        # Remove double spaces and dots
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\.+', '.', cleaned)
        cleaned = cleaned.strip()
        
        # If description became too short, return original
        if len(cleaned) < 10 and len(description) > 10:
            return description
        
        return cleaned
    
    def _process_objects(self, objects_data: Dict) -> Dict[str, List]:
        """Process object detection results"""
        if not objects_data or 'bboxes' not in objects_data:
            return {'main': [], 'secondary': [], 'all': []}
        
        labels = objects_data.get('labels', [])
        
        # Count occurrences
        object_counts = {}
        for label in labels:
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Sort by frequency
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize
        main_objects = [obj[0] for obj in sorted_objects[:3]]
        secondary_objects = [obj[0] for obj in sorted_objects[3:6]]
        
        return {
            'main': main_objects,
            'secondary': secondary_objects,
            'all': labels,
            'counts': object_counts
        }
    
    
    def _extract_detailed_objects(self, description: str, objects_data: Dict) -> Dict[str, List]:
        """Extract detailed object categories from description and detection results"""
        import re
        
        detailed = {
            'main': [],
            'clothing': [],
            'accessories': [],
            'props': [],
            'background': [],
            'architecture': []
        }
        
        # Get main objects from detection
        if objects_data and 'labels' in objects_data:
            detailed['main'] = list(set(objects_data['labels'][:3]))
        
        desc_lower = description.lower()
        
        # Extract clothing
        clothing_patterns = [
            r'wearing\s+(?:a\s+)?([^,\.]+(?:robe|dress|shirt|suit|coat|jacket|cloak|armor))',
            r'dressed in\s+(?:a\s+)?([^,\.]+)',
            r'(?:red|blue|green|white|black|gold|silver)\s+(\w+(?:robe|dress|shirt|suit|coat))'
        ]
        for pattern in clothing_patterns:
            matches = re.findall(pattern, desc_lower)
            detailed['clothing'].extend(matches)
        
        # Extract accessories
        accessory_patterns = [
            r'(?:with|wearing|has)\s+(?:a\s+)?([^,\.]+(?:belt|rope|chain|necklace|ring|crown|tiara|hat|glasses))',
            r'holding\s+(?:a\s+)?([^,\.]+)',
            r'(?:gold|silver|leather|rope)\s+(\w+)'
        ]
        for pattern in accessory_patterns:
            matches = re.findall(pattern, desc_lower)
            detailed['accessories'].extend(matches)
        
        # Extract props and items
        prop_patterns = [
            r'(?:holding|carrying|with)\s+(?:a\s+)?([^,\.]+(?:book|staff|sword|cup|orb|scroll|bag))',
            r'(?:sitting on|standing near|next to)\s+(?:a\s+)?([^,\.]+)'
        ]
        for pattern in prop_patterns:
            matches = re.findall(pattern, desc_lower)
            detailed['props'].extend(matches)
        
        # Extract background elements with improved patterns
        background_patterns = [
            r'(?:in front of|against|behind)\s+(?:a\s+)?([^,\.]+?)(?=\s*(?:,|\.|and|with|$))',
            r'(?:background shows|background with|background of)\s+(?:a\s+)?([^,\.]+?)(?=\s*(?:,|\.|and|with|$))',
            r'set against\s+(?:a\s+)?([^,\.]+?)(?=\s*(?:,|\.|and|with|$))',
            r'in the background(?:,?\s*)([^,\.]+?)(?=\s*(?:,|\.|$))',
            r'([^,\.]+?)\s+in the background'
        ]
        for pattern in background_patterns:
            matches = re.findall(pattern, desc_lower)
            for match in matches:
                # Clean up the match
                cleaned = match.strip()
                # Skip if it's too short or starts with problematic words
                if len(cleaned) > 3 and not cleaned.startswith(('it ', 'a ', 'an ', 'the ')):
                    detailed['background'].append(cleaned)
                elif cleaned.startswith(('a ', 'an ', 'the ')) and len(cleaned) > 5:
                    detailed['background'].append(cleaned)
        
        # Extract architectural elements
        arch_keywords = ['wall', 'column', 'arch', 'door', 'window', 'castle', 'building', 'tower', 'bridge']
        for keyword in arch_keywords:
            if keyword in desc_lower:
                # Find context around keyword
                # Better pattern for architectural elements
                pattern = rf'\b(\w+\s+)?{keyword}s?(?:\s+(\w+))?\b'
                matches = re.findall(pattern, desc_lower)
                for match in matches:
                    if match[0] and match[1]:
                        full_match = f"{match[0].strip()} {keyword} {match[1].strip()}"
                    elif match[0]:
                        full_match = f"{match[0].strip()} {keyword}"
                    elif match[1]:
                        full_match = f"{keyword} {match[1].strip()}"
                    else:
                        full_match = keyword
                    
                    # Only add if it's meaningful
                    if len(full_match) > 3 and full_match != keyword:
                        detailed['architecture'].append(full_match.strip())
        
        # Clean up duplicates and empty strings
        for key in detailed:
            detailed[key] = list(set([item.strip() for item in detailed[key] if item and item.strip()]))
        
        return detailed
    
    def _extract_enhanced_environment(self, description: str, regions: List = None) -> Dict[str, str]:
        """Extract enhanced environment information"""
        env = {
            'setting': 'unknown',
            'time_of_day': 'unknown',
            'weather': 'unknown',
            'lighting_quality': 'unknown',
            'atmosphere': 'unknown'
        }
        
        desc_lower = description.lower()
        
        # Enhanced setting detection
        indoor_keywords = ['indoor', 'room', 'interior', 'inside', 'hall', 'chamber', 'office', 'studio']
        outdoor_keywords = ['outdoor', 'outside', 'street', 'nature', 'garden', 'forest', 'mountain', 'sky']
        urban_keywords = ['city', 'urban', 'building', 'street', 'town']
        fantasy_keywords = ['castle', 'tower', 'magical', 'fantasy', 'kingdom']
        
        for keyword in indoor_keywords:
            if keyword in desc_lower:
                env['setting'] = 'indoor'
                break
        
        for keyword in outdoor_keywords:
            if keyword in desc_lower:
                env['setting'] = 'outdoor'
                break
                
        for keyword in urban_keywords:
            if keyword in desc_lower:
                env['setting'] = 'urban'
                break
                
        for keyword in fantasy_keywords:
            if keyword in desc_lower:
                env['setting'] = 'fantasy/medieval'
                break
        
        # Lighting quality
        if any(word in desc_lower for word in ['bright', 'well-lit', 'sunny', 'illuminated']):
            env['lighting_quality'] = 'bright'
        elif any(word in desc_lower for word in ['dark', 'dim', 'shadowy', 'gloomy']):
            env['lighting_quality'] = 'dark'
        elif any(word in desc_lower for word in ['warm', 'golden', 'amber', 'orange']):
            env['lighting_quality'] = 'warm'
        elif any(word in desc_lower for word in ['cool', 'blue', 'cold']):
            env['lighting_quality'] = 'cool'
        
        # Atmosphere
        if any(word in desc_lower for word in ['mysterious', 'eerie', 'spooky', 'haunting']):
            env['atmosphere'] = 'mysterious'
        elif any(word in desc_lower for word in ['peaceful', 'serene', 'calm', 'tranquil']):
            env['atmosphere'] = 'peaceful'
        elif any(word in desc_lower for word in ['dramatic', 'epic', 'grand', 'majestic']):
            env['atmosphere'] = 'dramatic'
        elif any(word in desc_lower for word in ['cozy', 'warm', 'inviting', 'comfortable']):
            env['atmosphere'] = 'cozy'
        
        return env
    
    def _extract_style_info(self, description: str) -> Dict[str, str]:
        """Extract style information from description"""
        style = {
            'type': 'photographic',
            'mood': 'neutral',
            'lighting': 'natural',
            'color_palette': 'varied'
        }
        
        desc_lower = description.lower()
        
        # Style detection
        if any(word in desc_lower for word in ['painting', 'artistic', 'abstract', 'illustration']):
            style['type'] = 'artistic'
        elif any(word in desc_lower for word in ['cartoon', 'anime', 'animated']):
            style['type'] = 'cartoon'
        elif any(word in desc_lower for word in ['render', '3d', 'cgi']):
            style['type'] = '3d_render'
        
        # Mood detection
        if any(word in desc_lower for word in ['dark', 'moody', 'dramatic']):
            style['mood'] = 'dramatic'
        elif any(word in desc_lower for word in ['bright', 'cheerful', 'vibrant']):
            style['mood'] = 'cheerful'
        elif any(word in desc_lower for word in ['calm', 'serene', 'peaceful']):
            style['mood'] = 'serene'
        
        # Lighting detection
        if any(word in desc_lower for word in ['sunset', 'sunrise', 'golden']):
            style['lighting'] = 'golden_hour'
        elif any(word in desc_lower for word in ['night', 'dark', 'dim']):
            style['lighting'] = 'low_light'
        
        return style
    
    def _extract_environment_info(self, description: str) -> Dict[str, str]:
        """Extract environment information"""
        env = {
            'setting': 'unknown',
            'time_of_day': 'unknown',
            'weather': 'unknown'
        }
        
        desc_lower = description.lower()
        
        # Setting detection
        if any(word in desc_lower for word in ['indoor', 'room', 'interior']):
            env['setting'] = 'indoor'
        elif any(word in desc_lower for word in ['outdoor', 'outside', 'street', 'nature']):
            env['setting'] = 'outdoor'
        elif any(word in desc_lower for word in ['city', 'urban', 'building']):
            env['setting'] = 'urban'
        
        # Time detection
        if any(word in desc_lower for word in ['morning', 'dawn', 'sunrise']):
            env['time_of_day'] = 'morning'
        elif any(word in desc_lower for word in ['evening', 'dusk', 'sunset']):
            env['time_of_day'] = 'evening'
        elif any(word in desc_lower for word in ['night', 'dark']):
            env['time_of_day'] = 'night'
        elif any(word in desc_lower for word in ['day', 'afternoon', 'bright']):
            env['time_of_day'] = 'day'
        
        return env
    
    
    def _generate_tags_from_analysis(self, analysis: Dict) -> Dict[str, List[str]]:
        """Generate tags similar to JoyCaption from Florence-2 analysis"""
        tags = {
            'general': [],
            'character': [],
            'clothing': [],
            'style': [],
            'environment': [],
            'objects': [],
            'colors': [],
            'composition': []
        }
        
        description = analysis.get('description', '').lower()
        
        # Extract character tags
        character_keywords = {
            '1girl': ['girl', 'woman', 'female', 'lady'],
            '1boy': ['boy', 'man', 'male', 'guy'],
            'multiple_girls': ['girls', 'women'],
            'multiple_boys': ['boys', 'men'],
            'solo': ['alone', 'single person', 'one person']
        }
        
        for tag, keywords in character_keywords.items():
            if any(kw in description for kw in keywords):
                tags['character'].append(tag)
        
        # Extract clothing from detailed objects
        if 'objects_detailed' in analysis:
            tags['clothing'].extend(analysis['objects_detailed'].get('clothing', []))
        
        # Extract style tags
        style_mapping = {
            'realistic': ['realistic', 'photorealistic', 'photo'],
            'anime': ['anime', 'manga'],
            'digital_art': ['digital art', 'digital painting'],
            'watercolor': ['watercolor'],
            'oil_painting': ['oil painting'],
            '3d': ['3d render', '3d model', 'cgi']
        }
        
        for tag, keywords in style_mapping.items():
            if any(kw in description for kw in keywords):
                tags['style'].append(tag)
        
        # Environment tags
        env_info = analysis.get('environment', {})
        setting = env_info.get('setting', 'unknown')
        if setting != 'unknown':
            tags['environment'].append(setting)
        
        time_of_day = env_info.get('time_of_day', 'unknown')
        if time_of_day != 'unknown':
            tags['environment'].append(time_of_day)
        
        # Object tags from detection
        if 'objects' in analysis:
            main_objects = analysis['objects'].get('main', [])
            tags['objects'].extend(main_objects[:5])  # Top 5 objects
        
        # Color tags
        color_keywords = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 
                         'black', 'white', 'brown', 'gray', 'pink', 'gold', 'silver']
        for color in color_keywords:
            if color in description:
                tags['colors'].append(color)
        
        # Composition tags
        composition_keywords = {
            'close-up': ['close up', 'closeup', 'face focus'],
            'full_body': ['full body', 'whole body', 'entire figure'],
            'upper_body': ['upper body', 'torso', 'bust'],
            'portrait': ['portrait', 'head shot'],
            'from_above': ['from above', 'top view', 'bird eye view'],
            'from_below': ['from below', 'low angle']
        }
        
        for tag, keywords in composition_keywords.items():
            if any(kw in description for kw in keywords):
                tags['composition'].append(tag)
        
        # Combine all tags into general tags
        for category, tag_list in tags.items():
            if category != 'general' and tag_list:
                tags['general'].extend(tag_list)
        
        # Remove duplicates
        for category in tags:
            tags[category] = list(dict.fromkeys(tags[category]))
        
        return tags
    
    def get_status(self) -> Dict[str, Any]:
        """Get current analyzer status"""
        return {
            'mode': 'florence2',
            'device': self.device,
            'gpu_detected': self.detected_gpu,
            'compatibility_mode': self.gpu_compatibility_mode,
            'initialized': self._initialized,
            'error': self._init_error
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")


# Backward compatibility
ImageAnalyzer = SmartImageAnalyzer
