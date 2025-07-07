"""
Kontext Smart Assistant for Forge WebUI
Provides AI-powered prompt generation for FLUX.1 Kontext
"""

import gradio
import torch
import logging
from pathlib import Path
import sys
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

# Forge imports
from modules import scripts, shared
from modules.ui_components import InputAccordion

# Compatibility imports handled in modules

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
    from ka_modules.image_analyzer import ImageAnalyzer
    from ka_modules.shared_state import shared_state
    MODULES_AVAILABLE = True
    logger.info("Smart Assistant modules loaded successfully")
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.error(f"Failed to load Smart Assistant modules: {e}")


class KontextAssistant(scripts.Script):
    """Smart Assistant for FLUX.1 Kontext prompt generation"""
    
    def __init__(self):
        super().__init__()
        self.templates = None
        self.generator = None
        self.analyzer = None
        self.initialized = False
        # Store references to kontext images UI components
        self.kontext_image_components = None
        
    def title(self):
        return "Kontext Smart Assistant"
    
    def show(self, is_img2img):
        """Show in both txt2img and img2img tabs"""
        return scripts.AlwaysVisible
    
    def _initialize_modules(self):
        """Lazy initialization of modules"""
        if not self.initialized and MODULES_AVAILABLE:
            try:
                self.templates = PromptTemplates()
                self.generator = PromptGenerator(self.templates)
                self.analyzer = ImageAnalyzer()
                self.initialized = True
                logger.info("Smart Assistant modules initialized")
            except Exception as e:
                logger.error(f"Failed to initialize modules: {e}")
                self.initialized = False
    
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
        # Based on kontext.py structure: enabled, img1, img2, img3, sizing, reduce, metrics, dims_info
        # So images should be at indices 1, 2, 3
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
    
    def analyze_image(self, image_index: int, *args):
        """Analyze a kontext image"""
        try:
            logger.info(f"[DEBUG] analyze_image called for index {image_index}")
            
            # Get kontext images from UI args
            kontext_images = self._get_kontext_images_from_ui(*args)
            
            if image_index >= len(kontext_images):
                return f"❌ Invalid image index {image_index + 1}", {}
            
            image = kontext_images[image_index]
            if image is None:
                return f"❌ No image in slot {image_index + 1} - please load an image in Forge FluxKontext Pro first", {}
            
            logger.info(f"[DEBUG] Found image for analysis: {image.size}")
            
            # Analyze with our analyzer
            if self.analyzer and hasattr(self.analyzer, 'analyze'):
                analysis = self.analyzer.analyze(image)
            else:
                # Fallback analysis
                analysis = {
                    "size": f"{image.size[0]}x{image.size[1]}",
                    "mode": image.mode,
                    "description": "Basic analysis (Florence-2 not loaded)"
                }
            
            # Format output - with proper type checking
            output = f"✅ Image {image_index + 1}: {analysis.get('size', 'Unknown size')}\n"
            
            if 'description' in analysis:
                output += f"Content: {analysis['description']}"
            
            # Safe handling of objects
            if 'objects' in analysis:
                objects = analysis['objects']
                if isinstance(objects, list):
                    # If it's a list, join first 5 items
                    output += f"\nObjects: {', '.join(str(obj) for obj in objects[:5])}"
                elif isinstance(objects, str):
                    # If it's a string, use it directly
                    output += f"\nObjects: {objects}"
                else:
                    # For any other type, convert to string
                    output += f"\nObjects: {str(objects)}"
            
            if 'style' in analysis:
                output += f"\nStyle: {analysis['style']}"
            
            logger.info(f"[DEBUG] Analysis complete for image {image_index + 1}")
            return output, analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_index + 1}: {e}", exc_info=True)
            return f"❌ Error: {str(e)[:100]}", {}
    
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
        
        # Initialize modules on first UI creation
        self._initialize_modules()
        
        with InputAccordion(False, label="🤖 " + self.title()) as enabled:
            gradio.Markdown("💡 Analyzes context images and generates optimal FLUX.1 Kontext prompts")
            
            # Image analysis section
            gradio.Markdown("### 📸 Context Image Analysis")
            gradio.Markdown("*If you have kontext images loaded, click analyze to understand their content*")
            
            analysis_displays = []
            analysis_data = []
            
            for i in range(3):
                with gradio.Row():
                    with gradio.Column(scale=1):
                        analyze_btn = gradio.Button(
                            f"🔍 Analyze Image {i+1}",
                            variant="secondary"
                        )
                    with gradio.Column(scale=3):
                        analysis_text = gradio.Textbox(
                            value="",
                            label=f"Analysis {i+1}",
                            interactive=False,
                            lines=2,
                            placeholder="Click analyze to scan kontext image..."
                        )
                
                analysis_displays.append((analyze_btn, analysis_text))
                # Hidden state to store analysis data
                analysis_state = gradio.State(value={})
                analysis_data.append(analysis_state)
            
            gradio.Markdown("---")
            
            # Prompt generation section
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
                
                show_debug = gradio.Checkbox(
                    label="Show debug information",
                    value=False
                )
            
            # Event handlers
            def generate_prompt(task_type, user_intent, use_analysis, *analysis_states):
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
                        image_analysis=combined_analysis if combined_analysis else None
                    )
                    
                    return prompt
                    
                except Exception as e:
                    logger.error(f"Error generating prompt: {e}")
                    return f"❌ Error: {str(e)}"
            
            def clear_all():
                """Clear all fields"""
                return "", "", {}
            
            # Connect analyze buttons
            for i, (btn, display) in enumerate(analysis_displays):
                btn.click(
                    fn=lambda *args, idx=i: self.analyze_image(idx, *args),
                    inputs=[],  # Will receive args from Forge
                    outputs=[display, analysis_data[i]]
                )
            
            # Connect generate button
            generate_btn.click(
                fn=generate_prompt,
                inputs=[task_type, user_intent, use_analysis] + analysis_data,
                outputs=generated_prompt
            )
            
            # Connect clear button
            clear_btn.click(
                fn=clear_all,
                outputs=[user_intent, generated_prompt] + [analysis_data[0]]
            )
        
        # Return UI components for Forge
        return [enabled, task_type, user_intent, generated_prompt, 
                preservation_strength, use_analysis, show_debug]
    
    def process(self, p, enabled, task_type, user_intent, generated_prompt,
                preservation_strength, use_analysis, show_debug):
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