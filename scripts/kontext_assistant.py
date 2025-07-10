"""
Kontext Smart Assistant for Forge WebUI
Provides AI-powered prompt generation for FLUX.1 Kontext
"""

import gradio
import torch
import logging
from pathlib import Path
import sys
import time
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

# Forge imports
from modules import scripts, shared
from modules.ui_components import InputAccordion

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KontextAssistant")

# Setup imports for ka_modules
script_dir = Path(__file__).parent
extension_root = script_dir.parent

# Add paths for imports safely
if extension_root.exists() and str(extension_root) not in sys.path:
    sys.path.insert(0, str(extension_root))

# Import our modules
try:
    from ka_modules.templates import PromptTemplates
    from ka_modules.prompt_generator import PromptGenerator
    from ka_modules.smart_analyzer import SmartAnalyzer
    from ka_modules.shared_state import shared_state
    from ka_modules.text_utils import DescriptionCleaner
    from ka_modules.cache_utils import ImageHashCache
    MODULES_AVAILABLE = True
    logger.info("Smart Assistant modules loaded successfully")
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.error(f"Failed to load Smart Assistant modules: {e}")
    
    # Try to run the installer
    install_script = extension_root / "install.py"
    if install_script.exists():
        try:
            import subprocess
            logger.info("Attempting to install missing dependencies...")
            result = subprocess.run([sys.executable, str(install_script)], capture_output=True, text=True)
            if result.returncode == 0:
                # Try importing again
                try:
                    from ka_modules.templates import PromptTemplates
                    from ka_modules.prompt_generator import PromptGenerator
                    from ka_modules.smart_analyzer import SmartAnalyzer
                    from ka_modules.shared_state import shared_state
                    from ka_modules.text_utils import DescriptionCleaner
                    from ka_modules.cache_utils import ImageHashCache
                    MODULES_AVAILABLE = True
                    logger.info("Dependencies installed successfully! Modules loaded.")
                except ImportError:
                    logger.error("Failed to import modules after installation. Please install dependencies manually.")
            else:
                logger.error(f"Dependency installation failed: {result.stderr}")
                logger.error("Please install dependencies manually using: pip install -r requirements.txt")
        except Exception as install_error:
            logger.error(f"Failed to run installer: {install_error}")
            logger.error("Please install dependencies manually using: pip install -r requirements.txt")


class KontextAssistant(scripts.Script):
    """Smart Assistant for FLUX.1 Kontext prompt generation"""
    
    # Shared analyzer across instances to avoid reloading
    _shared_analyzer = None
    _analyzer_settings = None
    
    def __init__(self):
        super().__init__()
        self.templates = None
        self.generator = None
        self.analyzer = None
        self.initialized = False
        # Store references to kontext images UI components
        self.kontext_image_components = None
        # Settings from UI
        self.force_cpu = False
        # Track image hashes for auto-clear
        self._image_hashes = [None, None, None]
        # Use limited cache for analysis results
        self._analysis_cache = ImageHashCache(max_images=9) if MODULES_AVAILABLE else None
        
    def title(self):
        return "Kontext Smart Assistant"
    
    def show(self, is_img2img):
        """Show in both txt2img and img2img tabs"""
        return scripts.AlwaysVisible
    
    def _initialize_modules(self, force_cpu=False):
        """Lazy initialization of modules"""
        if not self.initialized and MODULES_AVAILABLE:
            try:
                self.templates = PromptTemplates()
                self.generator = PromptGenerator(self.templates)
                
                # Check if we need to recreate analyzer due to settings change
                current_settings = (force_cpu,)
                if (KontextAssistant._shared_analyzer is None or 
                    KontextAssistant._analyzer_settings != current_settings):
                    
                    logger.info(f"Creating smart analyzer with settings: force_cpu={force_cpu}")
                    
                    # Create new smart analyzer
                    device = "cpu" if force_cpu else "cuda"
                    KontextAssistant._shared_analyzer = SmartAnalyzer(
                        device=device,
                        force_cpu=force_cpu
                    )
                    logger.info("Smart analyzer created (Florence-2 + JoyCaption)")
                    
                    KontextAssistant._analyzer_settings = current_settings
                
                self.analyzer = KontextAssistant._shared_analyzer
                self.initialized = True
                logger.info("Smart Assistant modules initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize modules: {e}")
                self.initialized = False
    
    def _should_clear_analysis(self, image_index: int, image: Optional[Image.Image]) -> bool:
        """Check if analysis should be cleared due to image change"""
        if image is None:
            # Image removed
            if self._image_hashes[image_index] is not None:
                logger.info(f"Image {image_index + 1} removed, clearing analysis")
                self._image_hashes[image_index] = None
                if self._analysis_cache:
                    self._analysis_cache.invalidate_image(image_index)
                return True
            return False
        
        # Calculate image hash
        import hashlib
        try:
            # Get image data consistently
            if hasattr(image, 'mode') and hasattr(image, 'size'):
                # Convert to RGB if needed for consistent hashing
                if image.mode != 'RGB':
                    image_rgb = image.convert('RGB')
                else:
                    image_rgb = image
                current_hash = hashlib.md5(image_rgb.tobytes()).hexdigest()
            else:
                # Fallback hash method
                current_hash = hashlib.md5(str(image).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating image hash: {e}")
            current_hash = str(time.time())  # Use timestamp as fallback
        
        # Log for debugging
        logger.info(f"Image {image_index + 1} hash check: old={self._image_hashes[image_index]}, new={current_hash}, image_size={getattr(image, 'size', 'unknown')}")
        
        # Check if image changed
        if self._image_hashes[image_index] != current_hash:
            logger.info(f"Image {image_index + 1} changed, clearing cached analysis")
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
            kontext_images = shared_state.get_images()
            image_count = sum(1 for img in kontext_images if img is not None)
            logger.info(f"[DEBUG] Shared state has {image_count} images")
            if any(img is not None for img in kontext_images):
                logger.debug(f"Got {shared_state.image_count} images from shared state")
                return kontext_images
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
            # Check if this is a PIL Image
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
                     show_detailed: bool, *args):
        """Analyze a kontext image with timing"""
        try:
            start_time = time.time()
            # Update settings
            self.force_cpu = force_cpu
            
            # Reinitialize if needed
            self._initialize_modules(force_cpu=force_cpu)
            
            # Get kontext images from UI args
            kontext_images = self._get_kontext_images_from_ui(*args)
            
            if image_index >= len(kontext_images):
                return f"❌ Invalid image index {image_index + 1}", {}
            
            image = kontext_images[image_index]
            if image is None:
                # Clear analysis cache for this slot
                self._analysis_cache.invalidate_image(image_index)
                return f"❌ No image in slot {image_index + 1} - please load an image in Forge FluxKontext Pro first", {}
            
            # Check if we should clear due to image change
            if self._should_clear_analysis(image_index, image):
                self._analysis_cache.invalidate_image(image_index)
                logger.info(f"Image {image_index + 1} changed, clearing cached analysis")
            
            # Check cache first
            cache_key = f"analysis_{use_florence}_{use_joycaption}"
            cached_result = self._analysis_cache.get(self._analysis_cache.get_image_key(image_index, cache_key))
            if cached_result:
                logger.info(f"Using cached analysis for image {image_index + 1}")
                return self._format_analysis_output(cached_result, show_detailed), cached_result
            
            # Skip progress for now - might be causing the hang
            # Previously had progress callback here but removed it
            
            # Use SmartAnalyzer with selected models
            if not use_florence and not use_joycaption:
                return "❌ Please select at least one analysis method", {}
                
            try:
                # Run analysis with SmartAnalyzer
                analysis = self.analyzer.analyze(image, use_florence=use_florence, use_joycaption=use_joycaption)
                
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
            
            # Cache the result
            self._analysis_cache.set(self._analysis_cache.get_image_key(image_index, cache_key), analysis)
            
            # Format output using the method
            return self._format_analysis_output(analysis, show_detailed), analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_index + 1}: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", {}
    
    def ui(self, is_img2img):
        """Create Smart Assistant UI"""
        
        if not MODULES_AVAILABLE:
            with InputAccordion(False, label="❌ " + self.title() + " (Modules Missing)"):
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
        
        with InputAccordion(False, label="🤖 " + self.title()) as enabled:
            # Image analysis section in a frame
            with gradio.Group():
                gradio.Markdown("### 📸 Context Image Analysis")
                
                # Analysis settings - above analysis fields
                with gradio.Row():
                    use_florence = gradio.Checkbox(
                        label="Use Florence-2 (Generate simple description). Fast mode",
                        value=False,
                        info="Technical analysis: object detection with coordinates, text reading"
                    )
                    use_joycaption = gradio.Checkbox(
                        label="Use JoyCaption (Generate advanced description and tags). Slow mode",
                        value=True,
                        info="Artistic analysis: detailed descriptions, Danbooru tags, style detection (8GB model)"
                    )
            
            
            # Analyze All button
            with gradio.Row():
                analyze_all_btn = gradio.Button(
                    "🔍 Analyze All Images",
                    variant="primary",
                    size="sm"
                )
                analyze_status = gradio.Markdown("", elem_id="analyze_status")
            
            analysis_displays = []
            analysis_data = []
            
            for i in range(3):
                with gradio.Row():
                    with gradio.Column(scale=4):
                        with gradio.Row():
                            analysis_text = gradio.Textbox(
                                value="",
                                label=f"Analysis {i+1}",
                                interactive=False,
                                lines=3,
                                placeholder="Click 🔍 to analyze kontext image {i+1}...",
                                elem_id=f"analysis_text_{i}"
                            )
                    with gradio.Column(scale=1, min_width=100):
                        analyze_btn = gradio.Button(
                            "🔍",
                            variant="secondary",
                            size="sm",
                            elem_id=f"analyze_btn_{i}"
                        )
                
                analysis_displays.append((analyze_btn, analysis_text))
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
                
                with gradio.Row():
                    copy_info = gradio.Markdown(
                        "*💡 Copy the generated prompt to your main prompt field for best results*",
                        elem_id="copy_info"
                    )
            
            # Advanced options
            with gradio.Accordion("⚙️ Advanced Options", open=False):
                preservation_strength = gradio.Slider(
                    label="Context Preservation Strength",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    info="How much to preserve unchanged elements"
                )
                
                use_analysis = gradio.Checkbox(
                    label="Use image analysis in prompt generation",
                    value=True
                )
                
                show_detailed = gradio.Checkbox(
                    label="Show Detailed Results",
                    value=False,
                    info="Display enhanced analysis results in JSON format"
                )
                
                # Performance settings
                with gradio.Row():
                    force_cpu = gradio.Checkbox(
                        label="Force CPU mode",
                        value=False,
                        info="Use CPU instead of GPU (slower but more compatible)"
                    )
                
                show_debug = gradio.Checkbox(
                    label="Show debug information",
                    value=False
                )
            
            # Event handlers
            def generate_prompt(task_type, user_intent, use_analysis, preservation_strength, *analysis_states):
                """Generate FLUX.1 Kontext prompt"""
                try:
                    if not user_intent:
                        return "❌ Please describe what you want to change"
                    
                    if not self.generator:
                        return "❌ Prompt generator not initialized"
                    
                    # Combine analysis data if enabled
                    combined_analysis = {}
                    if use_analysis:
                        for state in analysis_states:
                            if isinstance(state, dict) and state:
                                combined_analysis.update(state)
                    
                    # Generate prompt
                    prompt = self.generator.generate(
                        task_type=task_type,
                        user_intent=user_intent,
                        image_analysis=combined_analysis if combined_analysis else None,
                        preservation_strength=preservation_strength
                    )
                    
                    return prompt
                    
                except Exception as e:
                    logger.error(f"Error generating prompt: {e}")
                    return f"❌ Error: {str(e)}"
            
            def clear_all():
                """Clear all fields"""
                return "", "", {}
            
            def analyze_all_images(*args):
                """Analyze all kontext images with progress updates"""
                results = []
                force_cpu, use_florence, use_joycaption, show_detailed = args[:4]
                
                logger.info(f"analyze_all_images called with use_florence={use_florence}, use_joycaption={use_joycaption}")
                
                # Get current images to check for changes
                current_images = self._get_kontext_images_from_ui(*args[4:])
                
                # Count images to analyze
                images_to_analyze = sum(1 for img in current_images if img is not None)
                if images_to_analyze == 0:
                    status = "❌ No images to analyze"
                    return ["", {}, "", {}, "", {}, status]
                
                # First pass: check all images for changes and clear if needed
                for i in range(3):
                    if current_images[i] is None and self._image_hashes[i] is not None:
                        # Image was removed
                        logger.info(f"Image {i+1} was removed, clearing analysis")
                        self._image_hashes[i] = None
                        self._analysis_cache.invalidate_image(i)
                        # Clear the display
                        results.append(("", {}))
                    else:
                        # Image exists or slot was already empty
                        results.append(None)  # Placeholder
                
                # Second pass: analyze images that exist sequentially
                analyzed_count = 0
                for i in range(3):
                    if results[i] is None and current_images[i] is not None:  # Not already cleared and has image
                        try:
                            # Update status to show progress
                            analyzed_count += 1
                            logger.info(f"Analyzing image {i+1} of {images_to_analyze}...")
                            
                            # Check if image changed
                            self._should_clear_analysis(i, current_images[i])
                            
                            # Analyze with progress info
                            result_text, result_data = self.analyze_image(
                                i, force_cpu, use_florence, use_joycaption,
                                show_detailed, *args[4:]
                            )
                            
                            # Add progress info to result
                            progress_info = f"\n[{analyzed_count}/{images_to_analyze} analyzed]"
                            if analyzed_count < images_to_analyze:
                                result_text = result_text + progress_info
                            
                            results[i] = (result_text, result_data)
                            
                            # Force garbage collection after each image to free memory
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            logger.error(f"Error analyzing image {i+1}: {e}")
                            results[i] = (f"❌ Error analyzing image {i+1}: {str(e)}", {})
                    elif results[i] is None:  # Empty slot
                        results[i] = ("", {})
                
                # Return all results + status message
                status = f"✅ All {images_to_analyze} images analyzed successfully!"
                
                # Unload models after analysis to free memory for generation
                if hasattr(self, 'analyzer') and self.analyzer:
                    try:
                        logger.info("Unloading analyzer models to free GPU memory...")
                        self.analyzer.unload_models()
                        # Force cleanup
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info("Analyzer models unloaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to unload models: {e}")
                
                return [
                    results[0][0], results[0][1],  # Image 1
                    results[1][0], results[1][1],  # Image 2
                    results[2][0], results[2][1],  # Image 3
                    status
                ]
            
            def copy_description(analysis_data, idx):
                """Copy main description to clipboard"""
                if isinstance(analysis_data, dict) and 'description' in analysis_data:
                    description = analysis_data['description']
                    # Gradio doesn't support direct clipboard access, so we'll show a message
                    # User can select and copy from the textbox
                    return f"✅ Description ready to copy from Analysis {idx+1} textbox"
                return "❌ No description to copy"
            
            # Function to check for removed images and clear analysis
            def check_and_clear_removed_images(*args):
                """Check all slots and clear analysis for removed images"""
                force_cpu, use_florence, use_joycaption, show_detailed = args[:4]
                
                # Get current images
                current_images = self._get_kontext_images_from_ui(*args[4:])
                
                results = []
                for i in range(3):
                    if current_images[i] is None and self._image_hashes[i] is not None:
                        # Image was removed, clear it
                        logger.info(f"Image {i+1} removed, clearing analysis display")
                        self._image_hashes[i] = None
                        self._analysis_cache.invalidate_image(i)
                        results.extend(["", {}])  # Clear display and data
                    else:
                        # Keep existing display/data
                        results.extend([gradio.update(), gradio.update()])
                
                return results
            
            # Connect individual analyze buttons
            for i, (btn, display) in enumerate(analysis_displays):
                btn.click(
                    fn=lambda *args, idx=i: self.analyze_image(idx, args[0], args[1], args[2], args[3], *args[4:]),
                    inputs=[force_cpu, use_florence, use_joycaption, show_detailed],
                    outputs=[display, analysis_data[i]]
                ).then(
                    # After analysis, check all slots for removed images
                    fn=check_and_clear_removed_images,
                    inputs=[force_cpu, use_florence, use_joycaption, show_detailed],
                    outputs=[item for pair in [(display, data) for display, data in zip([d[1] for d in analysis_displays], analysis_data)] for item in pair]
                )
            
            # Connect analyze all button
            analyze_all_btn.click(
                fn=analyze_all_images,
                inputs=[force_cpu, use_florence, use_joycaption, show_detailed],
                outputs=[
                    analysis_displays[0][1], analysis_data[0],  # Image 1
                    analysis_displays[1][1], analysis_data[1],  # Image 2
                    analysis_displays[2][1], analysis_data[2],  # Image 3
                    analyze_status
                ]
            )
            
            # Connect generate button
            generate_btn.click(
                fn=generate_prompt,
                inputs=[task_type, user_intent, use_analysis, preservation_strength] + analysis_data,
                outputs=generated_prompt
            )
            
            # Connect clear button
            clear_btn.click(
                fn=clear_all,
                outputs=[user_intent, generated_prompt] + [analysis_data[0]]
            )
            
            # Helper function to clear analysis when images change
            def on_image_change(img_index, *args):
                """Called when an image slot changes"""
                logger.info(f"Image {img_index + 1} slot changed")
                # Return empty analysis for this slot
                return "", {}
            
            # Add load event to set initial states
            enabled.change(
                fn=lambda x: (False, True) if x else (False, True),
                inputs=[enabled],
                outputs=[use_florence, use_joycaption]
            )
            
            # If we can access kontext image components, add change handlers
            # This would need to be done in the main UI where kontext images are defined
            # For now, we rely on the analyze functions to detect changes
        
        # Return UI components for Forge
        return [enabled, task_type, user_intent, generated_prompt, 
                preservation_strength, use_analysis, show_detailed, show_debug, force_cpu]
    
    def _format_analysis_output(self, analysis: dict, show_detailed: bool = False) -> str:
        """Format analysis results into readable output"""
        output = ""
        
        # Check if this is Florence-only mode for simple formatting
        analyzers_used = analysis.get('analyzers_used', {})
        is_florence_only = analyzers_used.get('florence2', False) and not analyzers_used.get('joycaption', False)
        
        # For Florence-only, use simple format
        if is_florence_only:
            # Just show description and basic technical data
            if 'description' in analysis:
                output += f"\n📝 Description: {analysis['description']}"
            
            # Technical data
            if 'technical_data' in analysis:
                tech = analysis['technical_data']
                output += "\n\n🔬 Technical Data:"
                if 'size' in tech:
                    output += f"\n📐 Size: {tech['size']}"
                if 'objects_with_positions' in tech and tech['objects_with_positions']:
                    main_objects = tech['objects_with_positions'].get('main', [])
                    if main_objects:
                        output += f"\n🎯 Objects: {', '.join(main_objects[:5])}"
            
            # Environment info from Florence
            if 'environment' in analysis and isinstance(analysis['environment'], dict):
                env = analysis['environment']
                if env.get('setting') and env['setting'] != 'unknown':
                    output += f"\n🌍 Setting: {env['setting']}"
                if env.get('time_of_day') and env['time_of_day'] != 'unknown':
                    output += f"\n🕐 Time: {env['time_of_day']}"
            
            # Style from Florence (simple)
            if 'style' in analysis and isinstance(analysis['style'], dict):
                style = analysis['style']
                if style.get('mood') and style['mood'] != 'unknown':
                    output += f"\n💭 Mood: {style['mood']}"
            
            # Skip the rest of formatting for Florence-only
        else:
            # JoyCaption or combined mode - use full formatting
            # Categories (from JoyCaption) FIRST
            if 'categories' in analysis and isinstance(analysis['categories'], dict):
            cats = analysis['categories']
            
            # Characters
            if cats.get('characters'):
                output += f"\n👤 Characters: {', '.join(cats['characters'][:5])}"
                
            # Objects
            if cats.get('objects'):
                output += f"\n🎯 Objects: {', '.join(cats['objects'][:8])}"
                
            # Colors
            if cats.get('colors'):
                output += f"\n🎨 Colors: {', '.join(cats['colors'][:6])}"
                
            # Materials
            if cats.get('materials') and show_detailed:
                output += f"\n🧱 Materials: {', '.join(cats['materials'][:5])}"
                
            # Environment
            if cats.get('environment'):
                env = cats['environment'] if isinstance(cats['environment'], str) else ', '.join(cats['environment'])
                if env:
                    output += f"\n🌍 Environment: {env}"
                    
            # Lighting
            if cats.get('lighting'):
                lighting = cats['lighting'] if isinstance(cats['lighting'], str) else ', '.join(cats['lighting'])
                if lighting:
                    output += f"\n💡 Lighting: {lighting}"
                    
            # Pose
            if cats.get('pose') and show_detailed:
                pose = cats['pose'] if isinstance(cats['pose'], str) else ', '.join(cats['pose'])
                if pose:
                    output += f"\n🚶 Pose: {pose}"
                    
            # Style
            if cats.get('style'):
                style = cats['style'] if isinstance(cats['style'], str) else ', '.join(cats['style'][:3])
                if style:
                    output += f"\n🖼️ Style: {style}"
                    
            # Mood
            if cats.get('mood'):
                mood = cats['mood'] if isinstance(cats['mood'], str) else ', '.join(cats['mood'][:3])
                if mood:
                    output += f"\n💭 Mood: {mood}"
                    
            # Text/Symbols
            if cats.get('text') and show_detailed:
                text = cats['text'] if isinstance(cats['text'], str) else ', '.join(cats['text'])
                if text:
                    output += f"\n📝 Text/Symbols: {text}"
                    
            # Genre
            if cats.get('genre'):
                genre = cats['genre'] if isinstance(cats['genre'], str) else ', '.join(cats['genre'])
                if genre:
                    output += f"\n🎭 Genre: {genre}"
                    
            # Subject Actions
            if cats.get('subjects_actions') and show_detailed:
                actions = cats['subjects_actions'] if isinstance(cats['subjects_actions'], str) else ', '.join(cats['subjects_actions'])
                if actions:
                    output += f"\n🎬 Actions: {actions}"
        
        # Tags (from JoyCaption) - after categories
        if 'tags' in analysis and isinstance(analysis['tags'], dict):
            tags = analysis['tags']
            if tags.get('danbooru'):
                output += f"\n\n🏷️ Danbooru tags: {tags['danbooru']}"
            if tags.get('general') and show_detailed:
                output += f"\n🔖 General tags: {tags['general']}"
        
        # Descriptions - check if we have both models' descriptions
        analyzers_used = analysis.get('analyzers_used', {})
        both_models = analyzers_used.get('florence2', False) and analyzers_used.get('joycaption', False)
        
        if both_models:
            # Show both descriptions when both models are used
            if 'florence_description' in analysis:
                florence_desc = DescriptionCleaner.clean(analysis['florence_description'])
                if florence_desc:
                    output += f"\n\n📝 Florence-2 Description: {florence_desc}"
            
            if 'joycaption_description' in analysis:
                joy_desc = DescriptionCleaner.clean(analysis['joycaption_description'])
                if joy_desc:
                    output += f"\n\n📝 JoyCaption Description: {joy_desc}"
        else:
            # Single description for single model
            if 'description' in analysis:
                desc = DescriptionCleaner.clean(analysis['description'])
                if desc:
                    output += f"\n\n📝 Description: {desc}"
        
        # Technical data (from Florence)
        if 'technical_data' in analysis and isinstance(analysis['technical_data'], dict):
            tech = analysis['technical_data']
            output += f"\n\n🔬 Technical Data:"
            if tech.get('size'):
                output += f"\n📐 Size: {tech['size']}"
            if tech.get('text_detected'):
                output += f"\n📝 Text detected: {', '.join(tech['text_detected'][:3])}"
            if show_detailed and tech.get('object_counts'):
                output += f"\n📊 Object counts: {', '.join(f'{k}:{v}' for k,v in list(tech['object_counts'].items())[:5])}"
        
        # Objects (combined)
        elif 'objects' in analysis:
            objects = analysis['objects']
            if isinstance(objects, dict):
                if 'combined' in objects and objects['combined']:
                    output += f"\n🎯 Objects: {', '.join(objects['combined'][:8])}"
                elif 'florence' in objects and objects['florence']:
                    output += f"\n🔍 Florence objects: {', '.join(objects['florence'][:5])}"
                elif 'joycaption' in objects and objects['joycaption']:
                    output += f"\n🏷️ JoyCaption objects: {', '.join(objects['joycaption'][:5])}"
                elif 'main' in objects and objects['main']:
                    output += f"\n🎯 Main objects: {', '.join(objects['main'])}"
                    if 'secondary' in objects and objects['secondary'] and show_detailed:
                        output += f"\n🔸 Secondary objects: {', '.join(objects['secondary'])}"
        
        # Style (merged) - only show if not already shown in categories
        if 'style' in analysis and isinstance(analysis['style'], dict):
            style_info = analysis['style']
            
            # Check if we already showed style in categories
            showed_style_in_categories = ('categories' in analysis and 
                                        isinstance(analysis.get('categories'), dict) and 
                                        analysis['categories'].get('style'))
            
            artistic_style = style_info.get('artistic_style', 'unknown')
            if artistic_style != 'unknown' and not showed_style_in_categories:
                output += f"\n🎨 Artistic style: {artistic_style}"
                
            # Mood might not be in categories, so show it if available
            mood = style_info.get('mood', 'unknown')
            showed_mood_in_categories = ('categories' in analysis and 
                                       isinstance(analysis.get('categories'), dict) and 
                                       analysis['categories'].get('mood'))
            if mood != 'unknown' and not showed_mood_in_categories:
                output += f"\n💭 Mood: {mood}"
                
            composition = style_info.get('composition', 'unknown')
            if composition != 'unknown' and show_detailed:
                output += f"\n📸 Composition: {composition}"
        
        # Analysis time and mode
        if 'total_analysis_time' in analysis:
            output += f"\n\n⏱️ Analysis time: {analysis['total_analysis_time']:.2f}s"
        elif 'analysis_time' in analysis:
            output += f"\n\n⏱️ Analysis time: {analysis['analysis_time']:.2f}s"
            
        # Analyzers used
        if 'analyzers_used' in analysis:
            used = analysis['analyzers_used']
            modes = []
            if used.get('florence2'):
                modes.append('Florence-2')
            if used.get('joycaption'):
                modes.append('JoyCaption')
            if modes:
                output += f" | Using: {', '.join(modes)}"
        elif 'analysis_mode' in analysis:
            output += f" | Mode: {analysis['analysis_mode']}"
        
        return output

    def process(self, p, enabled, task_type, user_intent, generated_prompt,
                preservation_strength, use_analysis, show_detailed, show_debug, force_cpu):
        """Process - we don't modify the generation, just provide UI"""
        if not enabled:
            return
        
        # Log usage if debug enabled
        if show_debug and generated_prompt:
            logger.info(f"Smart Assistant used - Task: {task_type}, Generated: {generated_prompt[:50]}...")
    
    def process_before_every_sampling(self, p, *args, **kwargs):
        """Hook to get access to all script args including kontext images"""
        # This method will receive all args from all scripts
        # We can use it to find kontext images
        pass
    
    def postprocess(self, p, processed, *args):
        """Postprocess - nothing to do"""
        pass


# Register the script
kontext_assistant = KontextAssistant()