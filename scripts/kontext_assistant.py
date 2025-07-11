import os
import sys
import gc
import torch
import time
import gradio
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# H11 fix no longer needed after cache cleanup
# Keeping comment for reference: use clear_all_caches.bat if h11 errors return

# Get the extension directory
extension_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add ka_modules to path
modules_path = os.path.join(extension_dir, 'ka_modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Try to import modules
MODULES_AVAILABLE = False
try:
    from ka_modules.shared_state import shared_state
    from ka_modules.smart_analyzer import SmartAnalyzer
    from ka_modules.prompt_generator import PromptGenerator
    from ka_modules.text_utils import DescriptionCleaner
    from ka_modules.templates import PromptTemplates
    from ka_modules.image_utils import validate_image_list, validate_and_convert_image
    
    MODULES_AVAILABLE = True
    logger.info(f"✅ Kontext Assistant modules loaded successfully from {modules_path}")
except ImportError as e:
    logger.error(f"❌ Failed to import Kontext Assistant modules: {e}")
    logger.error(f"Extension directory: {extension_dir}")
    logger.error(f"Modules path: {modules_path}")
    logger.error(f"sys.path: {sys.path}")

# Gradio module compatibility check
GRADIO_AVAILABLE = False
try:
    from modules import scripts, shared
    from modules.scripts import AlwaysVisible
    
    # Gradio is available for UI
    
    GRADIO_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import WebUI modules: {e}")
    logger.error("This extension must be run within the Automatic1111 WebUI environment")
    AlwaysVisible = object  # Fallback for class definition


class AnalysisCache:
    """Simple cache for analysis results"""
    def __init__(self):
        self._cache = {}
        self._max_size = 10  # Keep last 10 analyses
        
    def get(self, key: str) -> Optional[Dict]:
        """Get cached analysis"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Dict):
        """Cache analysis result"""
        # Limit cache size
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def clear(self):
        """Clear all cached results"""
        self._cache.clear()
    
    def invalidate_image(self, image_index: int):
        """Invalidate all cache entries for a specific image index"""
        keys_to_remove = [key for key in self._cache if key.startswith(f"img_{image_index}_")]
        for key in keys_to_remove:
            del self._cache[key]
    
    def get_image_key(self, image_index: int, params: str) -> str:
        """Generate cache key for an image"""
        return f"img_{image_index}_{params}"


import threading

class KontextAssistant(scripts.Script if GRADIO_AVAILABLE else object):
    """Smart assistant for context-aware prompt generation"""
    
    # Class-level shared analyzer and lock
    _shared_analyzer = None
    _analyzer_lock = threading.Lock()
    _analyzer_settings = None
    _analysis_lock = threading.Lock()  # Lock for preventing concurrent analysis
    
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.prompt_gen = None
        self.force_cpu = False
        self._analysis_cache = AnalysisCache()
        self._image_hashes = {}  # Track image hashes for change detection
        self._analysis_states = [{}, {}, {}]  # Store analysis states for each image
        self._analysis_texts = ["", "", ""]  # Store formatted analysis texts
        self._analysis_displays = []  # Will store the textbox references
        self._analysis_data_states = []  # Will store the gradio State components
        self._callback_registered = False
        
    def title(self):
        return "Kontext Assistant - Smart Image Analysis & Prompt Generation"
    
    def show(self, is_img2img):
        return AlwaysVisible if GRADIO_AVAILABLE else False
    
    def _get_analyzer_settings(self, force_cpu: bool, florence_model_type: str = "base", auto_unload: bool = True, unload_delay: int = 60):
        """Get current analyzer settings"""
        return {
            'force_cpu': force_cpu,
            'florence_model_type': florence_model_type,
            'auto_unload': auto_unload,
            'unload_delay': unload_delay
        }
    
    def _initialize_modules(self, force_cpu=False, florence_model_type="base", auto_unload=True, unload_delay=60):
        """Initialize modules if needed, reusing shared analyzer when possible"""
        if not MODULES_AVAILABLE:
            return False
            
        try:
            # Get device
            device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
            
            # Current settings
            current_settings = self._get_analyzer_settings(force_cpu, florence_model_type, auto_unload, unload_delay)
            
            # Check if we need to reinitialize analyzer
            need_new_analyzer = (
                KontextAssistant._shared_analyzer is None or
                KontextAssistant._analyzer_settings != current_settings
            )
            
            if need_new_analyzer:
                with KontextAssistant._analyzer_lock:
                    # Double-check after acquiring lock
                    if KontextAssistant._shared_analyzer is None or KontextAssistant._analyzer_settings != current_settings:
                        logger.info(f"Creating new SmartAnalyzer with device={device}, florence_model_type={florence_model_type}")
                        
                        # Unload existing analyzer if settings changed
                        if KontextAssistant._shared_analyzer and KontextAssistant._analyzer_settings != current_settings:
                            logger.info("Settings changed, unloading existing analyzer...")
                            KontextAssistant._shared_analyzer.unload_models()
                            KontextAssistant._shared_analyzer = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Create new analyzer with delayed auto_unload for memory efficiency
                        KontextAssistant._shared_analyzer = SmartAnalyzer(
                            device=device,
                            force_cpu=force_cpu,
                            florence_model_type=florence_model_type,
                            auto_unload=auto_unload,  # Auto unload models after analysis
                            unload_delay=unload_delay    # Wait before unloading
                        )
                        KontextAssistant._analyzer_settings = current_settings
            
            # Use the shared analyzer
            self.analyzer = KontextAssistant._shared_analyzer
            
            # Initialize prompt generator if needed
            if self.prompt_gen is None:
                templates = PromptTemplates()
                self.prompt_gen = PromptGenerator(templates)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_model_status(self) -> str:
        """Get current model loading status"""
        if not self.analyzer:
            return "Not initialized"
        
        try:
            status = self.analyzer.get_status()
            loaded_models = []
            
            # Check Florence Base
            if status.get('florence_base', {}).get('loaded', False):
                loaded_models.append("Florence-2 Base")
            
            # Check PromptGen
            if status.get('florence_promptgen', {}).get('loaded', False):
                loaded_models.append("PromptGen v2.0")
            
            if loaded_models:
                return f"Loaded: {', '.join(loaded_models)}"
            else:
                return "No models loaded"
                
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return "Status unavailable"
    
    def _get_performance_info(self) -> str:
        """Get detailed performance and model information"""
        info = []
        
        # Header
        info.append("### 🚀 Model & Performance Information\n")
        
        # Model status
        try:
            if self.analyzer:
                status = self.analyzer.get_status()
                
                # Florence Base info - check if ever loaded
                base_status = status.get('florence_base', {})
                if base_status:
                    info.append("**Florence-2 Base**")
                    if base_status.get('loaded', False):
                        info.append(f"- Model: {base_status.get('model_id', 'Unknown')}")
                        info.append(f"- Device: {base_status.get('device', 'Unknown')}")
                        info.append(f"- dtype: {base_status.get('dtype', 'Unknown')}")
                        info.append(f"- Compiled: {'Yes' if base_status.get('compiled', False) else 'No'}")
                        if 'gpu_memory' in base_status:
                            info.append(f"- GPU Memory: {base_status['gpu_memory']['allocated_gb']:.2f} GB")
                    else:
                        info.append("- Status: Not loaded (will load on first use)")
                    info.append("")
                
                # PromptGen info - check if ever loaded
                pg_status = status.get('florence_promptgen', {})
                if pg_status:
                    info.append("**PromptGen v2.0**")
                    if pg_status.get('loaded', False):
                        info.append(f"- Model: {pg_status.get('model_id', 'Unknown')}")
                        info.append(f"- Device: {pg_status.get('device', 'Unknown')}")
                        info.append(f"- dtype: {pg_status.get('dtype', 'Unknown')}")
                        info.append(f"- Compiled: {'Yes' if pg_status.get('compiled', False) else 'No'}")
                        if 'gpu_memory' in pg_status:
                            info.append(f"- GPU Memory: {pg_status['gpu_memory']['allocated_gb']:.2f} GB")
                    else:
                        info.append("- Status: Not loaded (will load on first use)")
                    info.append("")
        except Exception as e:
            logger.error(f"Error getting model status in performance info: {e}")
            pass
        
        # Performance metrics from last analyses
        info.append("### ⏱️ Recent Performance Metrics\n")
        
        perf_data = []
        for i in range(3):
            if self._analysis_states[i] and self._analysis_states[i].get('success', False):
                analysis = self._analysis_states[i]
                img_num = i + 1
                
                # Overall time - check both processing_time and total_analysis_time
                if 'processing_time' in analysis:
                    perf_data.append(f"**Image {img_num}**: {analysis['processing_time']}")
                elif 'total_analysis_time' in analysis:
                    perf_data.append(f"**Image {img_num}**: {analysis['total_analysis_time']:.2f}s")
                
                # Task breakdown
                if 'task_times' in analysis and analysis['task_times']:
                    tasks = analysis['task_times']
                    task_str = ", ".join([f"{k}: {v:.2f}s" for k, v in tasks.items()])
                    perf_data.append(f"  - Tasks: {task_str}")
        
        if perf_data:
            info.extend(perf_data)
        else:
            info.append("*No performance data yet - analyze images to see metrics*")
        
        # GPU memory if available
        if torch.cuda.is_available():
            info.append("\n### 💾 GPU Memory Usage\n")
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info.append(f"- Allocated: {allocated:.2f} GB")
            info.append(f"- Reserved: {reserved:.2f} GB")
            info.append(f"- Total VRAM: {total:.2f} GB")
        
        # Version info
        info.append("\n### ℹ️ Version Info\n")
        info.append("- Kontext Assistant: v2.77")
        info.append(f"- PyTorch: {torch.__version__}")
        info.append("- Optimizations: FP16 + torch.compile")
        
        # Get current unload settings from analyzer
        if self.analyzer:
            info.append(f"- Auto-unload: {'Enabled' if self.analyzer.auto_unload else 'Disabled'}")
            if self.analyzer.auto_unload:
                info.append(f"- Unload delay: {self.analyzer.unload_delay} seconds")
        
        return "\n".join(info)
    
    def _has_image_changed(self, image: Image.Image, image_index: int) -> bool:
        """Check if an image has changed since last analysis"""
        if image is None:
            return False
            
        # Calculate image hash
        import hashlib
        img_bytes = image.tobytes()
        current_hash = hashlib.md5(img_bytes).hexdigest()
        
        # Check if changed
        prev_hash = self._image_hashes.get(image_index)
        if prev_hash != current_hash:
            self._image_hashes[image_index] = current_hash
            if self._analysis_cache:
                self._analysis_cache.invalidate_image(image_index)
            return True
        
        return False
    
    def _get_kontext_images_from_ui(self, *args) -> List[Optional[Image.Image]]:
        """Get kontext images from shared state or UI args"""
        kontext_images = []
        
        # Debug logging
        logger.info(f"[DEBUG] _get_kontext_images_from_ui called with {len(args)} args")
        
        # First try to get from shared state
        if MODULES_AVAILABLE and shared_state:
            try:
                kontext_images = shared_state.get_images()
                
                # Validate images to handle any format issues
                if validate_image_list:
                    kontext_images = validate_image_list(kontext_images)
                
                image_count = sum(1 for img in kontext_images if img is not None)
                logger.info(f"[DEBUG] Shared state has {image_count} valid images after validation")
                
                # Log details about each image
                for i, img in enumerate(kontext_images):
                    if img is not None:
                        try:
                            logger.info(f"[DEBUG] Image {i+1}: size={img.size}, mode={img.mode}")
                        except Exception as e:
                            logger.error(f"[DEBUG] Image {i+1} error: {e}")
                    else:
                        logger.info(f"[DEBUG] Image {i+1}: None")
                
                if any(img is not None for img in kontext_images):
                    logger.debug(f"Got {image_count} images from shared state")
                    return kontext_images
            except Exception as e:
                logger.error(f"[DEBUG] Error getting images from shared state: {e}")
        else:
            logger.info("[DEBUG] Shared state not available")
        
        # Fallback to parsing UI args
        logger.debug(f"Total args received: {len(args)}")
        
        # Log arg types for debugging
        for i, arg in enumerate(args[:10]):  # First 10 args
            arg_type = type(arg).__name__
            if hasattr(arg, 'size') and hasattr(arg, 'mode'):
                logger.info(f"[DEBUG] Arg {i}: PIL Image {arg.size}")
            else:
                logger.info(f"[DEBUG] Arg {i}: {arg_type}")
        
        # Try to find kontext images in args
        for i in range(len(args)):
            arg = args[i]
            # Try to validate and convert the argument to an image
            if validate_and_convert_image:
                validated_img = validate_and_convert_image(arg)
                if validated_img is not None:
                    logger.debug(f"Found and validated image at arg index {i}: {validated_img.size}")
                    kontext_images.append(validated_img)
            else:
                # Fallback to old method
                if hasattr(arg, 'mode') and hasattr(arg, 'size'):
                    logger.debug(f"Found image at arg index {i}: {arg.size}")
                    kontext_images.append(arg)
                elif isinstance(arg, Image.Image):
                    logger.debug(f"Found PIL Image at arg index {i}: {arg.size}")
                    kontext_images.append(arg)
        
        # If we didn't find images this way, try the expected positions
        if not kontext_images and len(args) >= 4:
            for i in [1, 2, 3]:
                if i < len(args):
                    img = args[i]
                    if img is not None and hasattr(img, 'mode'):
                        kontext_images.append(img)
                    else:
                        kontext_images.append(None)
        
        # Pad to ensure we have 3 slots
        while len(kontext_images) < 3:
            kontext_images.append(None)
        
        # Only keep first 3
        kontext_images = kontext_images[:3]
        
        # Log what we found
        image_count = sum(1 for img in kontext_images if img is not None)
        logger.info(f"Found {image_count} kontext images from UI args")
        
        return kontext_images
    
    def analyze_image(self, image_index: int, force_cpu: bool, 
                     use_florence: bool, use_joycaption: bool, 
                     florence_model: str, show_detailed: bool, 
                     promptgen_instruction: str, auto_unload: bool,
                     unload_delay: int, *args):
        """Analyze a kontext image with timing"""
        logger.info(f"analyze_image called: image_index={image_index}, florence_model={florence_model}, use_florence={use_florence}, use_joycaption={use_joycaption}, promptgen_instruction={promptgen_instruction}")
        
        # Use lock to prevent concurrent analysis
        with KontextAssistant._analysis_lock:
            try:
                start_time = time.time()
                # Update settings
                self.force_cpu = force_cpu
                
                # Reinitialize if needed
                self._initialize_modules(force_cpu=force_cpu, florence_model_type=florence_model, 
                                       auto_unload=auto_unload, unload_delay=unload_delay)
                
                # Get kontext images from UI args
                kontext_images = self._get_kontext_images_from_ui(*args)
                
                if image_index >= len(kontext_images):
                    return f"❌ Invalid image index {image_index + 1}", {}
            
                image = kontext_images[image_index]
                if image is None:
                    # Clear analysis cache for this slot
                    self._analysis_cache.invalidate_image(image_index)
                    return f"❌ No image in slot {image_index + 1} - please load an image in Forge FluxKontext Pro first", {}
                
                # Validate image is accessible
                try:
                    # Try to get image size to ensure it's valid
                    _ = image.size
                except Exception as e:
                    logger.error(f"Invalid image in slot {image_index + 1}: {e}")
                    return f"❌ Invalid or inaccessible image in slot {image_index + 1}", {}
            
                # Check if we should clear due to image change
                if self._has_image_changed(image, image_index):
                    logger.info(f"Image {image_index + 1} has changed, invalidating cache")
                
                # Check cache
                cache_key = f"{use_florence}_{use_joycaption}_{florence_model}_{promptgen_instruction}"
                cached_result = self._analysis_cache.get(self._analysis_cache.get_image_key(image_index, cache_key))
                if cached_result:
                    logger.info(f"Using cached analysis for image {image_index + 1}")
                    return self._format_analysis_output(cached_result, show_detailed), cached_result
                
                # Use SmartAnalyzer with selected models
                if not use_florence and not use_joycaption:
                    return "❌ Please select at least one analysis method", {}
                
                try:
                    # Log current analyzer state
                    if self.analyzer and hasattr(self.analyzer, 'florence_model_type'):
                        logger.info(f"Current SmartAnalyzer florence_model_type: {self.analyzer.florence_model_type}")
                    
                    # Run analysis with SmartAnalyzer
                    analysis = self.analyzer.analyze(image, use_florence=use_florence, use_joycaption=use_joycaption, promptgen_instruction=promptgen_instruction)
                    
                    # Check for errors
                    if not analysis.get('success', True):
                        error_msg = analysis.get('error', 'Unknown error')
                        return f"❌ Analysis failed: {error_msg}", {}
                    
                    # Skip progress for now
                    # if hasattr(progress, '__call__'):
                    #     progress(1.0, desc="Analysis complete!")
                        
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    return f"❌ Analysis failed: {str(e)}", {}
            
                # Calculate analysis time
                analysis_time = time.time() - start_time
                logger.info(f"Analysis completed in {analysis_time:.2f}s")
                
                # Cache the result
                self._analysis_cache.set(self._analysis_cache.get_image_key(image_index, cache_key), analysis)
                
                # Format output using the method
                return self._format_analysis_output(analysis, show_detailed), analysis
                
            except Exception as e:
                logger.error(f"Error analyzing image {image_index + 1}: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Error analyzing image: {str(e)}", {}
    
    def ui(self, is_img2img):
        """Create Smart Assistant UI"""
        
        if not MODULES_AVAILABLE:
            with gradio.Accordion(label="❌ " + self.title() + " (Modules Missing)", open=False):
                gradio.Markdown("""
                ### ⚠️ Smart Assistant modules not found!
                
                Please ensure:
                1. The `ka_modules` folder exists in the extension directory
                2. All required files are present (templates.py, prompt_generator.py, etc.)
                3. Python dependencies are installed
                
                Check the console for detailed error messages.
                """)
            return []
        
        # Initialize modules on first UI creation with default settings
        self._initialize_modules(force_cpu=False)
        
        with gradio.Accordion(label="🤖 " + self.title(), open=False):
            # Add custom CSS for the refresh button
            custom_css = """
            <style>
            .refresh-button {
                color: #2196F3 !important;
                font-size: 1.2em !important;
                transition: transform 0.3s ease;
            }
            .refresh-button:hover {
                color: #1976D2 !important;
                transform: rotate(180deg);
            }
            </style>
            """
            gradio.HTML(custom_css)
            
            # Hidden values for compatibility
            use_florence = gradio.State(value=True)
            use_joycaption = gradio.State(value=False)
            florence_model = gradio.State(value="base")  # Always use base for initial analysis
            promptgen_instruction = gradio.State(value="<MORE_DETAILED_CAPTION>")
            
            # Image analysis section in a frame
            with gradio.Group():
                gradio.Markdown("""
                ### 📸 Context Image Analysis <span style="font-size: 0.8em; color: #888;">(Automatic Dual-Model Analysis)</span>
                """)
                
                # Analyze and refresh buttons
                with gradio.Row():
                    with gradio.Column(scale=3):
                        analyze_all_btn = gradio.Button(
                            "🔍 Analyze",
                            variant="primary",
                            size="sm"
                        )
                    with gradio.Column(scale=1):
                        refresh_btn = gradio.Button(
                            "🔄",
                            size="sm",
                            elem_classes="refresh-button"
                        )
                    with gradio.Column(scale=6):
                        analyze_status = gradio.Markdown("", elem_id="analyze_status")
                
                analysis_displays = []
                analysis_data = []
                
                for i in range(3):
                    analysis_text = gradio.Textbox(
                        value="",
                        label=f"Analysis {i+1}",
                        interactive=False,
                        lines=4,  # Increased from 3 to 4 lines
                        placeholder=f"Click 🔍 Analyze to analyze kontext image {i+1}...",
                        elem_id=f"analysis_text_{i}"
                    )
                    
                    analysis_displays.append(analysis_text)
                    # Hidden state to store analysis data
                    analysis_state = gradio.State(value={})
                    analysis_data.append(analysis_state)
            
            # Prompt generation section in a separate frame
            with gradio.Group():
                gradio.Markdown("### ✨ Prompt Generation")
                
                with gradio.Row():
                    task_type = gradio.Dropdown(
                        label="Task Type",
                        choices=[
                            ("🎨 Change Object Color/State", "object_manipulation"),
                            ("🖼️ Apply Artistic Style", "style_transfer"),
                            ("🌅 Change Environment/Background", "environment_change"),
                            ("💡 Adjust Lighting", "lighting_adjustment"),
                            ("🔄 Transform State (new/old/broken)", "state_change"),
                            ("↔️ Extend Image (Outpainting)", "outpainting")
                        ],
                        value="object_manipulation",
                        scale=2
                    )
                    
                    user_intent = gradio.Textbox(
                        label="What do you want to change?",
                        placeholder="e.g., 'make the car blue', 'sunset lighting', 'oil painting style'...",
                        lines=1,
                        scale=3
                    )
                
                with gradio.Row():
                    generate_btn = gradio.Button(
                        "🎯 Generate Kontext Prompt",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gradio.Button(
                        "🗑️ Clear",
                        variant="secondary"
                    )
                
                generated_prompt = gradio.Textbox(
                    label="Generated FLUX.1 Kontext Prompt",
                    lines=4,
                    interactive=True,
                    placeholder="Your optimized prompt will appear here..."
                )
                
                # Optimization tips
                with gradio.Accordion("💡 Tips for Best Results", open=False):
                    gradio.Markdown("""
                    ### 🎯 For Best Results:
                    1. **Be Specific**: "golden hour sunset lighting" > "different lighting"
                    2. **Use Context**: The assistant understands what's in your images
                    3. **Combine Changes**: You can change multiple aspects at once
                    4. **Iterate**: Refine your prompts based on results
                    
                    ### 🎨 Task Types Explained:
                    - **Object Manipulation**: Change colors, materials, or states of objects
                    - **Style Transfer**: Apply artistic styles (oil painting, anime, photorealistic)
                    - **Environment Change**: Modify backgrounds, weather, time of day
                    - **Lighting**: Adjust mood through lighting changes
                    - **State Change**: Transform objects (new→old, clean→dirty)
                    - **Outpainting**: Extend image boundaries intelligently
                    """)
            
            # Advanced options
            with gradio.Accordion("⚙️ Advanced Options", open=False):
                with gradio.Row():
                    force_cpu = gradio.Checkbox(
                        label="Force CPU Mode",
                        value=False,
                        info="Use CPU instead of GPU (slower but more compatible)"
                    )
                    
                    show_detailed = gradio.Checkbox(
                        label="Show Detailed Output",
                        value=False,
                        info="Display additional technical information"
                    )
                
                with gradio.Row():
                    auto_unload = gradio.Checkbox(
                        label="Auto-unload Models",
                        value=True,
                        info="Automatically unload models after 60s to save VRAM"
                    )
                    
                    unload_delay = gradio.Slider(
                        label="Unload Delay (seconds)",
                        minimum=0,
                        maximum=300,
                        value=60,
                        step=15,
                        info="Time to wait before unloading models (0 = immediate)",
                        visible=True
                    )
                
                # Model status display
                model_status = gradio.Markdown(
                    value=self._get_model_status(),
                    elem_id="model_status"
                )
            
            # Extra inputs for image data
            extra_inputs = []
            if is_img2img:
                # In img2img mode, we might get different inputs
                extra_inputs = []
            
            # Event handlers
            def generate_prompt(task_type, user_intent, *analysis_states):
                """Generate prompt using all analysis data"""
                if not user_intent:
                    return "❌ Please describe what you want to change"
                
                # Collect all analysis data
                all_analyses = []
                for i, state in enumerate(analysis_states):
                    if state and isinstance(state, dict) and state.get('success', False):
                        all_analyses.append((i, state))
                
                if not all_analyses:
                    return "❌ Please analyze at least one image first"
                
                # Generate prompt
                try:
                    prompt = self.prompt_gen.generate_kontext_prompt(
                        task_type=task_type,
                        user_intent=user_intent,
                        analysis_data=all_analyses
                    )
                    return prompt
                except Exception as e:
                    logger.error(f"Prompt generation failed: {e}")
                    return f"❌ Failed to generate prompt: {str(e)}"
            
            # Individual buttons removed - using only "Analyze" button for all images
            
            def analyze_all_images(*args):
                """Analyze all available images"""
                results = []
                all_analysis_data = []
                
                # Extract the inputs we need
                force_cpu_val = args[0]
                use_florence_val = args[1]
                use_joycaption_val = args[2]
                florence_model_val = args[3]
                show_detailed_val = args[4]
                promptgen_instruction_val = args[5]
                auto_unload_val = args[6]
                unload_delay_val = args[7]
                
                # The rest are extra inputs for images
                extra_args = args[8:]
                
                # Get current images
                kontext_images = self._get_kontext_images_from_ui(*extra_args)
                
                for i in range(3):
                    current_image = kontext_images[i] if i < len(kontext_images) else None
                    
                    # Check if image changed or was removed
                    if current_image is None:
                        if i in self._image_hashes:
                            # Image was removed
                            logger.info(f"Image {i+1} was removed")
                            del self._image_hashes[i]
                            self._analysis_states[i] = {}
                            self._analysis_texts[i] = ""  # Clear stored text
                            if self._analysis_cache:
                                self._analysis_cache.invalidate_image(i)
                        results.extend(["", {}])
                    else:
                        # Check if image changed
                        if self._has_image_changed(current_image, i):
                            logger.info(f"Image {i+1} has changed - will re-analyze")
                            self._analysis_states[i] = {}
                            self._analysis_texts[i] = ""  # Clear stored text
                            if self._analysis_cache:
                                self._analysis_cache.invalidate_image(i)
                        
                        # Analyze the image
                        text_result, data_result = self.analyze_image(
                            i, force_cpu_val, use_florence_val, use_joycaption_val,
                            florence_model_val, show_detailed_val, promptgen_instruction_val,
                            auto_unload_val, unload_delay_val, *extra_args
                        )
                        results.extend([text_result, data_result])
                        self._analysis_states[i] = data_result
                        self._analysis_texts[i] = text_result  # Store formatted text
                
                return results
            
            
            # Generate prompt button
            generate_btn.click(
                fn=generate_prompt,
                inputs=[task_type, user_intent] + analysis_data,
                outputs=generated_prompt
            )
            
            # Clear button
            clear_btn.click(
                fn=lambda: "",
                outputs=generated_prompt
            )
            
            # No longer need to toggle visibility of checkboxes since they're hidden
            
            # Event handlers removed - no more dynamic UI updates needed
            # Models are managed automatically through SmartAnalyzer lifecycle
            
            # Periodic status update
            def update_status():
                """Update model status display"""
                return self._get_model_status()
            
            # Update status when force_cpu changes
            force_cpu.change(
                fn=update_status,
                outputs=model_status
            )
            
            # Store references to analysis displays and data states for later updates
            self._analysis_displays = analysis_displays
            self._analysis_data_states = analysis_data
            
            # Refresh button function
            def refresh_missing_analyses(*args):
                """Clear analysis for missing images and generated prompt"""
                # Get extra inputs for images
                extra_args = args[0:]
                
                # Get current images
                kontext_images = self._get_kontext_images_from_ui(*extra_args)
                
                results = []
                cleared_count = 0
                has_empty_slots = False
                
                for i in range(3):
                    current_image = kontext_images[i] if i < len(kontext_images) else None
                    
                    if current_image is None:
                        # No image in this slot - clear analysis
                        has_empty_slots = True
                        if self._analysis_states[i]:
                            logger.info(f"Clearing analysis for empty slot {i+1}")
                            cleared_count += 1
                        self._analysis_states[i] = {}
                        self._analysis_texts[i] = ""  # Clear stored text
                        if i in self._image_hashes:
                            del self._image_hashes[i]
                        if self._analysis_cache:
                            self._analysis_cache.invalidate_image(i)
                        results.extend(["", {}])
                    else:
                        # Image exists - keep current analysis
                        if self._analysis_states[i] and self._analysis_states[i].get('success', False):
                            # Use stored formatted text
                            results.extend([self._analysis_texts[i], self._analysis_states[i]])
                        else:
                            # No analysis yet
                            results.extend(["", self._analysis_states[i]])
                
                # Clear generated prompt if there are empty slots
                prompt_text = "" if has_empty_slots else generated_prompt.value
                
                logger.info(f"Refresh completed: cleared {cleared_count} empty slots" + (" and prompt" if has_empty_slots else ""))
                return results + [prompt_text]
            
            # Set up refresh button
            refresh_btn.click(
                fn=refresh_missing_analyses,
                inputs=extra_inputs,
                outputs=[item for pair in zip(analysis_displays, analysis_data) for item in pair] + [generated_prompt]
            )
            
            # Performance metrics section at the bottom (moved before button setup)
            with gradio.Accordion("📊 Performance & Model Info", open=False):
                performance_info = gradio.Markdown(
                    value=self._get_performance_info(),
                    elem_id="performance_info"
                )
            
            # Update the analyze button to also update performance info
            analyze_all_btn.click(
                fn=analyze_all_images,
                inputs=[
                    force_cpu, use_florence, use_joycaption, florence_model, show_detailed, promptgen_instruction,
                    auto_unload, unload_delay
                ] + extra_inputs,
                outputs=[item for pair in zip(analysis_displays, analysis_data) for item in pair]
            ).then(
                fn=lambda: self._get_performance_info(),
                inputs=[],
                outputs=performance_info
            )
            
            # Also update performance info after refresh
            refresh_btn.click(
                fn=lambda: self._get_performance_info(),
                inputs=[],
                outputs=performance_info,
                show_progress=False
            )
        
        # Return list of UI elements that need to be tracked
        return []
    
    def _format_analysis_output(self, analysis: dict, show_detailed: bool = False) -> str:
        """Format analysis results into readable output for dual-model system"""
        output = ""
        
        # Check if this is the new dual-model format
        if analysis.get('analysis_mode') == 'dual_model_automatic':
            # Brief description from Florence-2 Base
            if analysis.get('brief_description'):
                output += f"📝 Brief Description:\n{analysis['brief_description']}"
            
            # Detailed description from PromptGen v2.0
            if analysis.get('description'):
                if output:
                    output += "\n\n"
                desc = DescriptionCleaner.clean(analysis['description'])
                output += f"📝 Detailed Description:\n{desc}"
            
            # Tags from PromptGen v2.0
            if analysis.get('tags') and isinstance(analysis['tags'], dict):
                tags = analysis['tags']
                if tags.get('danbooru'):
                    if output:
                        output += "\n\n"
                    output += f"🏷️ Tags:\n{tags['danbooru']}"
            
            # Mixed caption from PromptGen - now formatted as FULL with multiple parts
            if analysis.get('mixed_caption'):
                # First check if there's any meaningful content
                caption_parts = analysis['mixed_caption'].split('\n\n')
                has_content = False
                content_parts = []
                
                for part in caption_parts:
                    part_cleaned = part.strip()
                    # Skip empty parts and pure tag lists
                    if not part_cleaned or (', ' in part_cleaned and any(tag_word in part_cleaned.lower() for tag_word in ['solo', 'background', 'looking', 'standing', 'sitting', 'full body', '1girl', '1boy'])):
                        continue
                    has_content = True
                    content_parts.append(part_cleaned)
                
                # Only show Mixed Caption section if there's actual content
                if has_content and content_parts:
                    if output:
                        output += "\n\n"
                    output += "📝 Mixed Caption (FULL):\n"
                    
                    description_shown = False
                    for i, part_cleaned in enumerate(content_parts):
                        # Replace NA with circle icon
                        part_cleaned = part_cleaned.replace('NA;NA', '⚪').replace(';NA', ';⚪').replace('NA;', '⚪;').replace(': NA', ': ⚪').replace(' NA ', ' ⚪ ').replace('(NA)', '(⚪)')
                        
                        # Check if this is a structured data part (contains colons and semicolons)
                        if ':' in part_cleaned and ';' in part_cleaned:
                            if not description_shown:
                                output += "🎨 Structured prompt elements:\n"
                                description_shown = True
                            else:
                                output += "\n\n🎨 Additional elements:\n"
                            output += part_cleaned
                        else:
                            # This is a natural language description
                            if description_shown:
                                continue  # Skip duplicate descriptions
                            output += f"🎨 {part_cleaned}"
                            description_shown = True
            
            # Composition analysis from PromptGen
            if analysis.get('composition_analysis'):
                if output:
                    output += "\n\n"
                output += "📸 Composition Analysis:\n"
                # Replace NA with circle icon
                comp_analysis = analysis['composition_analysis']
                comp_analysis = comp_analysis.replace('NA;NA', '⚪').replace(';NA', ';⚪').replace('NA;', '⚪;').replace(': NA', ': ⚪').replace(' NA ', ' ⚪ ').replace('(NA)', '(⚪)')
                output += comp_analysis
            
            # Object detection from Florence-2 Base (if detailed mode)
            if show_detailed and analysis.get('objects'):
                objects_data = analysis['objects']
                if isinstance(objects_data, dict) and 'all' in objects_data:
                    object_list = objects_data['all']
                    if object_list:
                        if output:
                            output += "\n\n"
                        output += f"🎯 Detected Objects:\n{', '.join(object_list[:10])}"
            
            # Mood if available
            if 'mood' in analysis and analysis['mood']:
                if output:
                    output += "\n\n"
                output += f"💭 Mood: {analysis['mood']}"
        else:
            # Fallback to original formatting for backward compatibility
            # Check which analyzers were used
            analyzers_used = analysis.get('analyzers_used', {})
            is_florence_only = analyzers_used.get('florence2', False) and not analyzers_used.get('joycaption', False)
            
            # Check for different types of output based on what's in the analysis
            # Description
            if 'description' in analysis:
                desc = DescriptionCleaner.clean(analysis['description'])
                if desc:
                    output += f"📝 Description:\n{desc}"
            
            # Tags
            if 'tags' in analysis and isinstance(analysis['tags'], dict):
                tags = analysis['tags']
                if tags.get('danbooru'):
                    if output:
                        output += "\n\n"
                    output += f"🏷️ Tags:\n{tags['danbooru']}"
            
            # Mixed caption
            if 'mixed_caption' in analysis:
                # Check for meaningful content first
                caption_parts = analysis['mixed_caption'].split('\n\n')
                has_content = False
                content_parts = []
                
                for part in caption_parts:
                    part_cleaned = part.strip()
                    # Skip empty and tag lists
                    if not part_cleaned or (', ' in part_cleaned and any(tag_word in part_cleaned.lower() for tag_word in ['solo', 'background', 'looking', 'standing', 'sitting', 'full body', '1girl', '1boy'])):
                        continue
                    has_content = True
                    content_parts.append(part_cleaned)
                
                if has_content and content_parts:
                    if output:
                        output += "\n\n"
                    output += "📝 Mixed Caption (FULL):\n"
                    
                    description_shown = False
                    for i, part_cleaned in enumerate(content_parts):
                        # Replace NA with circle icon
                        part_cleaned = part_cleaned.replace('NA;NA', '⚪').replace(';NA', ';⚪').replace('NA;', '⚪;').replace(': NA', ': ⚪').replace(' NA ', ' ⚪ ').replace('(NA)', '(⚪)')
                        
                        # Check if this is structured data
                        if ':' in part_cleaned and ';' in part_cleaned:
                            if not description_shown:
                                output += "🎨 Structured prompt elements:\n"
                                description_shown = True
                            else:
                                output += "\n\n🎨 Additional elements:\n"
                            output += part_cleaned
                        else:
                            # Natural language description
                            if description_shown:
                                continue
                            output += f"🎨 {part_cleaned}"
                            description_shown = True
            
            # Composition analysis
            if 'composition_analysis' in analysis:
                if output:
                    output += "\n\n"
                output += "📸 Composition Analysis:\n"
                comp_analysis = analysis['composition_analysis']
                comp_analysis = comp_analysis.replace('NA;NA', '⚪').replace(';NA', ';⚪').replace('NA;', '⚪;').replace(': NA', ': ⚪').replace(' NA ', ' ⚪ ').replace('(NA)', '(⚪)')
                output += comp_analysis
        
        return output or "❌ No analysis data available"


# Create script instance for WebUI
def create_script():
    """Factory function to create script instance"""
    return KontextAssistant()


# For WebUI registration
if GRADIO_AVAILABLE:
    # The WebUI will automatically create an instance
    pass