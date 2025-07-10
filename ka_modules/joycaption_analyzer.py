"""
JoyCaption Analyzer for artistic image analysis and categorization
"""

import torch
import logging
import time
import sys
from typing import Dict, List, Any, Optional, Set
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

logger = logging.getLogger(__name__)

# Optional Liger Kernel optimization
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    LIGER_AVAILABLE = True
    logger.info("Liger Kernel available for optimization")
except ImportError:
    LIGER_AVAILABLE = False
    logger.info("Liger Kernel not available, using standard inference")


class JoyCaptionAnalyzer:
    """
    JoyCaption analyzer for comprehensive image captioning and tagging
    Supports multiple caption styles and automatic categorization
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'beta-one': 'fancyfeast/llama-joycaption-beta-one-hf-llava',
        'alpha-two': 'fancyfeast/llama-joycaption-alpha-two-hf-llava',
    }
    
    # Prompt templates for different modes (from official application)
    PROMPTS = {
        # Descriptive modes with length support
        'descriptive': "Write a detailed description for this image.",
        'descriptive_casual': "Write a descriptive caption for this image in a casual tone.",
        'descriptive_casual_medium': "Write a medium-length descriptive caption for this image in a casual tone.",  # Your choice
        
        # Tag-based modes with length support
        'danbooru_tags': """Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.""",
        'booru_tags': "Write a list of Booru-like tags for this image.",
        'booru_tags_medium': "Write a medium-length list of Booru-like tags for this image.",  # Your choice
        
        # Other useful modes
        'straightforward': """Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with "This image is…" or similar phrasing.""",
    }
    
    # Extra options for better results
    EXTRA_OPTIONS = {
        'default': [
            "Include information about art style and medium.",
            "Include information about composition and camera angle.",
            "Include information about lighting and atmosphere.",
            "Focus on visual elements suitable for image generation.",
            "Be specific about colors, textures, and materials."
        ],
        'detailed': [
            "Include information about lighting.",
            "Include information about camera angle.",
            "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
            "Specify the depth of field and whether the background is in focus or blurred.",
            "Include information about the ages of any people/characters when applicable.",
        ],
        'technical': [
            "Include information about whether there is a watermark or not.",
            "Include information about whether there are JPEG artifacts or not.",
            "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
            "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
        ],
        'concise': [
            "ONLY describe the most important elements of the image.",
            "Do NOT use any ambiguous language.",
            "Do not mention the mood/feeling/etc of the image.",
            """Your response will be used by a text-to-image model, so avoid useless meta phrases like "This image shows…", "You are looking at...", etc.""",
        ]
    }
    
    def __init__(self, model_version='beta-one', device='cuda', force_cpu=False, 
                 temperature=0.6, top_p=0.9, max_new_tokens=512, use_liger=True, use_gguf=False):
        """
        Initialize JoyCaption analyzer
        
        Args:
            model_version: Version of JoyCaption to use
            device: Device for computation
            force_cpu: Force CPU usage
            temperature: Generation temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            use_liger: Use Liger Kernel optimization if available
            use_gguf: Use GGUF quantized model (automatic download)
        """
        self.model_version = model_version
        self.device = 'cpu' if force_cpu else device
        self.model_id = self.MODEL_CONFIGS.get(model_version, self.MODEL_CONFIGS['beta-one'])
        self.use_gguf = use_gguf
        
        # Generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.use_liger = use_liger and LIGER_AVAILABLE
        
        self.processor = None
        self.model = None
        self._initialized = False
        self._init_error = None
        
        logger.info(f"JoyCaption analyzer created with model: {self.model_id} (GGUF: {use_gguf})")
    
    def _ensure_initialized(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return
            
        if self._init_error:
            raise RuntimeError(f"JoyCaption initialization failed: {self._init_error}")
        
        try:
            start_time = time.time()
            
            # Use GGUF if requested (currently disabled)
            if self.use_gguf:
                raise RuntimeError("GGUF support has been disabled. Please use the full HuggingFace model.")
                return
            
            # Regular HuggingFace model loading
            logger.info(f"Loading JoyCaption model: {self.model_id}")
            start_time = time.time()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Fix tokenizer padding token (required for LLaMA tokenizer)
            # The tokenizer might not have pad_token set, which causes warnings
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                logger.info("Set padding token to EOS token")
            
            # Also ensure pad_token_id is set (sometimes just setting pad_token isn't enough)
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
                logger.info(f"Set pad_token_id to eos_token_id: {self.processor.tokenizer.eos_token_id}")
            
            # Ensure padding side is set correctly for LLaMA models
            self.processor.tokenizer.padding_side = 'left'  # LLaMA models use left padding
            
            # Log tokenizer configuration for debugging
            logger.info(f"Tokenizer configuration:")
            logger.info(f"  - pad_token: {self.processor.tokenizer.pad_token}")
            logger.info(f"  - pad_token_id: {self.processor.tokenizer.pad_token_id}")
            logger.info(f"  - eos_token: {self.processor.tokenizer.eos_token}")
            logger.info(f"  - eos_token_id: {self.processor.tokenizer.eos_token_id}")
            logger.info(f"  - padding_side: {self.processor.tokenizer.padding_side}")
                
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                device_map=0 if self.device == 'cuda' else None
            )
            
            # Apply Liger Kernel optimization if available and enabled
            if self.use_liger and self.device == 'cuda':
                try:
                    apply_liger_kernel_to_llama(model=self.model.language_model)
                    logger.info("Liger Kernel optimization applied successfully")
                except Exception as e:
                    logger.warning(f"Failed to apply Liger Kernel optimization: {e}")
            
            # Note: When using device_map=0, the model is already on the correct device
            # No need to explicitly move it
                
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"JoyCaption loaded successfully in {load_time:.1f}s on {self.device}")
            
            self._initialized = True
            
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"Failed to load JoyCaption: {e}")
            raise
    
    def _generate(self, image: Image.Image, prompt: str, extra_options: List[str] = None, extra_mode: str = 'default') -> str:
        """
        Generate caption/tags using JoyCaption with proper LLaVA format
        
        Args:
            image: PIL Image to analyze
            prompt: The prompt to use
            extra_options: Additional instructions
            
        Returns:
            Generated text
        """
        # Combine prompt with extra options
        full_prompt = prompt
        if extra_options:
            full_prompt += " " + " ".join(extra_options)
        elif extra_mode and extra_mode in self.EXTRA_OPTIONS:
            full_prompt += " " + " ".join(self.EXTRA_OPTIONS[extra_mode])
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Build conversation format for JoyCaption
        convo = [
            {
                "role": "system",
                # Beta One supports a wider range of system prompts
                "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
            },
            {
                "role": "user",
                "content": full_prompt,
            },
        ]
        
        # Apply chat template - CRITICAL for JoyCaption
        try:
            convo_string = self.processor.apply_chat_template(
                convo,
                tokenize=False,  # Returns string, not tokens
                add_generation_prompt=True
            )
            
            # Debug: Log the conversation string to check for multiple <bos> tokens
            if convo_string.count('<|begin_of_text|>') > 1:
                logger.warning(f"Multiple <bos> tokens detected in conversation string. This may degrade performance.")
                logger.debug(f"Conversation string: {convo_string[:200]}...")
                
        except Exception as e:
            logger.error(f"Failed to apply chat template: {e}")
            # Fallback to simple format
            convo_string = f"System: You are a helpful image captioner.\nUser: {full_prompt}\nAssistant:"
        
        # Process inputs with conversation string
        inputs = self.processor(
            text=[convo_string],  # As list
            images=[image],       # As list
            return_tensors="pt"
        )
        
        # Move to device and convert dtypes
        if self.device == 'cuda':
            # Move all inputs to CUDA
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            # Convert pixel values to bfloat16 (native dtype for JoyCaption)
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        # Generate
        try:
            with torch.no_grad():
                # Use sampling only if temperature > 0
                do_sample = self.temperature > 0
                
                generate_kwargs = {
                    'max_new_tokens': self.max_new_tokens,
                    'do_sample': do_sample,
                    'pad_token_id': self.processor.tokenizer.pad_token_id,
                }
                
                # Add sampling parameters only if sampling is enabled
                if do_sample:
                    generate_kwargs['temperature'] = self.temperature
                    generate_kwargs['top_p'] = self.top_p
                
                # Add timeout mechanism for generation
                import signal
                from contextlib import contextmanager
                
                @contextmanager
                def timeout(seconds):
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Generation timed out after {seconds} seconds")
                    
                    # Set the signal handler and alarm
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                
                # Use timeout only on Unix systems
                if hasattr(signal, 'SIGALRM'):
                    with timeout(60):  # 60 second timeout
                        generate_ids = self.model.generate(
                            **inputs,
                            **generate_kwargs
                        )[0]  # Get first (and only) generation
                else:
                    # Fallback for Windows/non-Unix
                    generate_ids = self.model.generate(
                        **inputs,
                        **generate_kwargs
                    )[0]  # Get first (and only) generation
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.error(f"Input keys: {list(inputs.keys())}")
            logger.error(f"Input shapes: {[(k, v.shape if torch.is_tensor(v) else type(v)) for k, v in inputs.items()]}")
            raise
        
        # Trim off the prompt tokens
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
        
        # Decode
        caption = self.processor.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return caption.strip()
    
    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured analysis from JoyCaption"""
        categories = {
            'characters': [],
            'objects': [],
            'colors': [],
            'materials': [],
            'environment': '',
            'lighting': '',
            'pose': '',
            'style': '',
            'mood': '',
            'text': '',
            'genre': '',
            'subjects_actions': ''
        }
        
        if not analysis_text:
            return categories
            
        # Split by lines and process each section
        lines = analysis_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            line_lower = line.lower()
            if 'character' in line_lower or 'subject' in line_lower:
                current_section = 'characters'
            elif 'object' in line_lower or 'item' in line_lower or 'prop' in line_lower:
                current_section = 'objects'
            elif 'color' in line_lower:
                current_section = 'colors'
            elif 'material' in line_lower or 'texture' in line_lower:
                current_section = 'materials'
            elif 'environment' in line_lower or 'setting' in line_lower or 'background' in line_lower:
                current_section = 'environment'
            elif 'lighting' in line_lower or 'light' in line_lower:
                current_section = 'lighting'
            elif 'pose' in line_lower or 'position' in line_lower:
                current_section = 'pose'
            elif 'style' in line_lower or 'art' in line_lower:
                current_section = 'style'
            elif 'mood' in line_lower or 'atmosphere' in line_lower:
                current_section = 'mood'
            elif 'text' in line_lower or 'symbol' in line_lower:
                current_section = 'text'
            elif 'genre' in line_lower or 'period' in line_lower:
                current_section = 'genre'
            elif 'doing' in line_lower or 'action' in line_lower:
                current_section = 'subjects_actions'
            elif current_section:
                # Process content for current section
                content = line.strip('- •·:')
                if not content:  # Skip empty lines
                    continue
                    
                if current_section in ['characters', 'objects', 'colors', 'materials']:
                    # List fields - split by commas and semicolons
                    items = []
                    # Handle both comma and semicolon separators
                    for separator in [',', ';', ' and ']:
                        if separator in content:
                            items.extend([item.strip() for item in content.split(separator)])
                            break
                    else:
                        # No separator found, add as single item
                        items.append(content)
                    
                    # Filter out empty items
                    items = [item for item in items if item and len(item) > 1]
                    categories[current_section].extend(items)
                else:
                    # String fields - append if already has content
                    if categories[current_section]:
                        categories[current_section] += f", {content}"
                    else:
                        categories[current_section] = content
                    
        return categories
    
    def _parse_structured_tags(self, tags_text: str) -> Dict[str, Any]:
        """Parse structured tags format"""
        categories = {
            'characters': [],
            'objects': [],
            'colors': [],
            'materials': [],
            'environment': '',
            'lighting': '',
            'pose': '',
            'style': '',
            'mood': '',
        }
        
        if not tags_text:
            return categories
            
        # Split by lines
        lines = tags_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if it's a category line
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    category = parts[0].strip().lower()
                    content = parts[1].strip()
                    
                    # Map to our categories
                    if category in ['character', 'characters']:
                        current_category = 'characters'
                    elif category in ['object', 'objects']:
                        current_category = 'objects'
                    elif category in ['color', 'colors']:
                        current_category = 'colors'
                    elif category in ['material', 'materials']:
                        current_category = 'materials'
                    elif category == 'environment':
                        current_category = 'environment'
                    elif category == 'lighting':
                        current_category = 'lighting'
                    elif category == 'pose':
                        current_category = 'pose'
                    elif category == 'style':
                        current_category = 'style'
                    elif category == 'mood':
                        current_category = 'mood'
                    
                    # Process content
                    if current_category and content:
                        if current_category in ['characters', 'objects', 'colors', 'materials']:
                            # List fields
                            items = [item.strip() for item in content.split(',')]
                            categories[current_category].extend([item for item in items if item])
                        else:
                            # String fields
                            categories[current_category] = content
        
        return categories
    
    def _parse_tags(self, danbooru_tags: str, general_tags: str) -> Dict[str, Set[str]]:
        """
        Parse tags into categories
        
        Args:
            danbooru_tags: Danbooru style tags
            general_tags: General descriptive tags
            
        Returns:
            Categorized tags
        """
        categories = {
            'characters': set(),
            'objects': set(),
            'colors': set(),
            'materials': set(),
            'clothing': set(),
            'style': set(),
            'mood': set(),
            'environment': set(),
            'lighting': set(),
            'pose': set(),
            'all_tags': set()
        }
        
        # Parse Danbooru tags (comma-separated in JoyCaption output)
        if danbooru_tags:
            # Split by comma to preserve multi-word tags
            tags = [tag.strip() for tag in danbooru_tags.split(',')]
            for tag in tags:
                tag_lower = tag.lower()
                categories['all_tags'].add(tag)
                
                # Categorize based on common patterns
                # Characters (check first as it's most specific)
                if any(x in tag_lower for x in ['girl', 'boy', 'man', 'woman', 'person', 'people', 
                                               'female', 'male', 'mage', 'sorceress', 'warrior',
                                               'eagle', 'bird', 'animal', 'cat', 'dog', 'creature']):
                    categories['characters'].add(tag)
                # Specific objects (not generic descriptors)
                elif any(x in tag_lower for x in ['staff', 'sword', 'shield', 'weapon', 'tool',
                                                 'crystal', 'orb', 'pentagram', 'circle']):
                    categories['objects'].add(tag)
                # Clothing and armor
                elif any(x in tag_lower for x in ['hat', 'cap', 'glasses', 'shirt', 'dress', 'suit',
                                                 'clothing', 'wear', 'armor', 'uniform', 'costume',
                                                 'robe', 'helmet', 'boots', 'gloves']):
                    categories['clothing'].add(tag)
                # Colors (only pure color words)
                elif tag_lower in ['red', 'blue', 'green', 'white', 'black', 'yellow',
                                  'orange', 'purple', 'pink', 'brown', 'gray', 'grey',
                                  'gold', 'silver', 'cyan', 'magenta']:
                    categories['colors'].add(tag)
                # Materials
                elif any(x in tag_lower for x in ['metal', 'wood', 'glass', 'fabric', 'leather', 'stone',
                                                 'plastic', 'paper', 'cloth', 'feather', 'fur', 'crystal']):
                    categories['materials'].add(tag)
                elif any(x in tag_lower for x in ['cartoon', 'anime', 'realistic', 'style', 'art',
                                                 'illustration', 'painting', 'sketch', 'render']):
                    categories['style'].add(tag)
                elif any(x in tag_lower for x in ['background', 'landscape', 'indoor', 'outdoor',
                                                 'forest', 'city', 'room', 'sky', 'ground']):
                    categories['environment'].add(tag)
                elif any(x in tag_lower for x in ['happy', 'sad', 'serious', 'mood', 'atmosphere',
                                                 'cheerful', 'dark', 'bright', 'calm']):
                    categories['mood'].add(tag)
                elif any(x in tag_lower for x in ['lighting', 'light', 'shadow', 'bright', 'dark',
                                                 'sunlight', 'moonlight', 'glow']):
                    categories['lighting'].add(tag)
                elif any(x in tag_lower for x in ['standing', 'sitting', 'walking', 'running', 'pose',
                                                 'facing', 'looking', 'holding', 'position']):
                    categories['pose'].add(tag)
                else:
                    # Skip generic descriptors and adjectives
                    skip_words = ['digital', 'detailed', 'intricate', 'vibrant', 'dynamic', 
                                 'ethereal', 'mystical', 'majestic', 'powerful', 'epic',
                                 'full body', 'front view', 'low angle', 'high contrast',
                                 'skin', 'view', 'angle', 'celestial', 'dramatic', 'glowing',
                                 'starry', 'cosmic', 'fantasy', 'sci-fi', 'otherworldly',
                                 'regal', 'textures', 'patterns', 'gemstones', 'elements']
                    # Skip if it's a generic word or contains only generic descriptors
                    is_generic = (tag_lower in skip_words or 
                                 any(skip in tag_lower for skip in skip_words) or
                                 len(tag_lower) < 3)  # Skip very short tags
                    
                    if not is_generic:
                        # Default to objects if no other category matches
                        categories['objects'].add(tag)
        
        # Additional processing - extract from tag content
        self._extract_additional_categories(categories)
        
        # Remove any empty strings or None values
        for key in categories:
            if isinstance(categories[key], set):
                categories[key] = {item for item in categories[key] if item and item.strip()}
        
        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in categories.items()}
    
    def _extract_additional_categories(self, categories: Dict[str, Set[str]]):
        """Extract additional categorization from existing tags"""
        # Extract colors from compound tags
        color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
                      'brown', 'black', 'white', 'gray', 'grey', 'gold', 'silver',
                      'cyan', 'magenta', 'crimson', 'azure', 'emerald']
        
        # Process all tags to extract additional info
        all_tags = []
        for tag_set in categories.values():
            if isinstance(tag_set, set):
                all_tags.extend(list(tag_set))
                
        for tag in all_tags:
            tag_lower = tag.lower()
            
            # Extract colors from compound tags like "blue skin", "dark purple robe"
            for color in color_words:
                if color in tag_lower and color not in categories['colors']:
                    categories['colors'].add(color)
                    
            # Extract specific objects from compound tags
            if 'skin' in tag_lower and 'blue skin' in tag_lower:
                categories['characters'].add('blue-skinned character')
            if 'staff' in tag_lower and tag not in categories['objects']:
                categories['objects'].add(tag)
                            
        # Extract materials from descriptive tags
        material_indicators = {
            'feather': ['feathered', 'feathers', 'plumage'],
            'metal': ['metallic', 'steel', 'iron', 'bronze'],
            'fabric': ['cloth', 'textile', 'cotton', 'silk'],
            'glass': ['glassy', 'transparent', 'crystal'],
            'leather': ['leathery', 'hide'],
            'wood': ['wooden', 'timber'],
        }
        
        # Collect materials to add after iteration
        materials_to_add = set()
        for material, indicators in material_indicators.items():
            for tag_set in categories.values():
                if isinstance(tag_set, set):
                    for tag in list(tag_set):  # Convert to list to avoid modification during iteration
                        if any(ind in tag.lower() for ind in indicators):
                            materials_to_add.add(material)
        
        # Add collected materials after iteration
        categories['materials'].update(materials_to_add)
    
    def analyze(self, image: Image.Image, modes: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using JoyCaption
        
        Args:
            image: PIL Image to analyze
            modes: List of analysis modes to use (default: ['descriptive', 'danbooru_tags', 'general_tags'])
            
        Returns:
            Analysis results with description, tags, and categories
        """
        # Ensure model is loaded
        self._ensure_initialized()
        
        # GGUF support disabled
        if self.use_gguf:
            raise RuntimeError("GGUF support has been disabled. Please use the full HuggingFace model.")
        
        # Default modes - use exactly the same prompts as in the online interface
        if modes is None:
            modes = ['descriptive_casual_medium', 'booru_tags_medium']
        
        start_time = time.time()
        results = {
            'analysis_mode': 'joycaption',
            'model_version': self.model_version
        }
        
        try:
            # Generate for each mode
            for mode in modes:
                if mode in self.PROMPTS:
                    logger.info(f"Generating {mode} caption...")
                    result = self._generate(image, self.PROMPTS[mode])
                    results[mode] = result
                    # Debug log for analysis
                    if mode == 'analysis':
                        logger.info(f"Analysis result preview: {result[:200]}...")
            
            # Parse tags into categories if we have them
            if 'danbooru_tags' in results or 'booru_tags_medium' in results or 'booru_tags' in results:
                # Get tags from any available source
                tags = (results.get('danbooru_tags', '') or 
                       results.get('booru_tags_medium', '') or 
                       results.get('booru_tags', ''))
                
                tag_categories = self._parse_tags(tags, '')
                # Store tag-based categories
                results['tag_categories'] = tag_categories
                
            # Parse structured tags if we have them
            if 'structured_tags' in results:
                structured_categories = self._parse_structured_tags(results['structured_tags'])
                results['categories'] = structured_categories
            # Fallback to old analysis format
            elif 'analysis' in results:
                analysis_categories = self._parse_analysis(results['analysis'])
                results['categories'] = analysis_categories
            else:
                # If no analysis, use tag categories as base
                if 'tag_categories' in results:
                    results['categories'] = results['tag_categories'].copy()
                
            # Merge categories from both sources if available
            if 'tag_categories' in results and 'categories' in results and 'analysis' in results:
                # Merge lists
                for key in ['characters', 'objects', 'colors', 'materials']:
                    if key in results['tag_categories']:
                        existing = results['categories'].get(key, [])
                        from_tags = list(results['tag_categories'][key])
                        # Combine and deduplicate
                        combined = list(set(existing + from_tags))
                        results['categories'][key] = combined
            
            # Analysis metadata
            results['analysis_time'] = time.time() - start_time
            results['success'] = True
            
        except Exception as e:
            logger.error(f"JoyCaption analysis failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'loaded': self._initialized,
            'model': self.model_id,
            'device': self.device,
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
        logger.info("JoyCaption model unloaded")