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

# Fix for Python 3.10 collections
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KontextAssistant")

# Setup imports for ka_modules
script_dir = Path(__file__).parent
extension_root = script_dir.parent

# Add paths for imports
for path in [extension_root, script_dir]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import our modules
try:
    from ka_modules.templates import PromptTemplates
    from ka_modules.prompt_generator import PromptGenerator
    from ka_modules.image_analyzer import ImageAnalyzer
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
    
    def _find_kontext_images(self, *args) -> List[Optional[Image.Image]]:
        """Try to find kontext images from the main kontext script"""
        kontext_images = []
        
        # Try to get from ForgeKontext class
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            
            # Add scripts directory to path if needed
            scripts_dir = Path(__file__).parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            
            # Try to import kontext module
            try:
                from kontext import ForgeKontext
                kontext_images = ForgeKontext.get_kontext_images()
                
                # Debug output
                image_count = sum(1 for img in kontext_images if img is not None)
                if image_count > 0:
                    logger.info(f"Successfully retrieved {image_count} kontext images")
                    
                return kontext_images
                
            except ImportError as e:
                logger.debug(f"Could not import ForgeKontext: {e}")
                
        except Exception as e:
            logger.debug(f"Error getting images from ForgeKontext: {e}")
        
        # If we couldn't get images, return empty list
        logger.debug("No kontext images found")
        return [None, None, None]
    
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
            def analyze_image(image_index: int, *args):
                """Analyze a kontext image"""
                try:
                    # Get kontext images
                    kontext_images = self._find_kontext_images(*args)
                    
                    # Debug output
                    logger.debug(f"Found {sum(1 for img in kontext_images if img is not None)} kontext images")
                    
                    if image_index >= len(kontext_images):
                        return f"❌ Invalid image index {image_index + 1}", {}
                    
                    image = kontext_images[image_index]
                    if image is None:
                        return f"❌ No image in slot {image_index + 1} - please load an image in Forge FluxKontext Pro first", {}
                    
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
                    
                    # Format output
                    output = f"✅ Image {image_index + 1}: {analysis.get('size', 'Unknown size')}\n"
                    if 'description' in analysis:
                        output += f"Content: {analysis['description']}"
                    
                    return output, analysis
                    
                except Exception as e:
                    logger.error(f"Error analyzing image {image_index + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"❌ Error: {str(e)}", {}
            
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
                    fn=lambda *args, idx=i: analyze_image(idx, *args),
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
    
    def postprocess(self, p, processed, *args):
        """Postprocess - nothing to do"""
        pass


# Register the script
kontext_assistant = KontextAssistant()