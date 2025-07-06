"""
FluxKontext Smart Assistant - Main Script Class
Fixed imports to avoid conflicts with Forge modules
"""

import gradio as gr
import logging
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Fix for Python 3.10+ compatibility
if sys.version_info >= (3, 10):
    import collections
    import collections.abc
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping

# Get the extension root directory (parent of scripts/)
ext_dir = Path(__file__).parent.parent
ext_modules_dir = ext_dir / "modules"

# Add our extension to Python path FIRST to prioritize our modules
if str(ext_dir) not in sys.path:
    sys.path.insert(0, str(ext_dir))

# Import from Forge WebUI
from modules import scripts
from modules.ui_components import InputAccordion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KontextAssistant")

# Now import our modules directly
try:
    # Direct import from our extension's modules folder
    import importlib.util
    
    # Load ui_components
    ui_spec = importlib.util.spec_from_file_location(
        "ka_ui_components", 
        ext_modules_dir / "ui_components.py"
    )
    ka_ui_components = importlib.util.module_from_spec(ui_spec)
    ui_spec.loader.exec_module(ka_ui_components)
    
    # Load prompt_generator
    pg_spec = importlib.util.spec_from_file_location(
        "ka_prompt_generator",
        ext_modules_dir / "prompt_generator.py"
    )
    ka_prompt_generator = importlib.util.module_from_spec(pg_spec)
    pg_spec.loader.exec_module(ka_prompt_generator)
    
    # Load templates
    tmpl_spec = importlib.util.spec_from_file_location(
        "ka_templates",
        ext_modules_dir / "templates.py"
    )
    ka_templates = importlib.util.module_from_spec(tmpl_spec)
    tmpl_spec.loader.exec_module(ka_templates)
    
    logger.info("Successfully loaded all KontextAssistant modules")
    
except Exception as e:
    logger.error(f"Error loading KontextAssistant modules: {str(e)}")
    ka_ui_components = None
    ka_prompt_generator = None
    ka_templates = None


class ForgeKontextAssistant(scripts.Script):
    """Smart assistant for FLUX.1 Kontext prompt generation."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.generator = None
        self.enhancer = None
        self.initialized = False
        
        # Get extension directory
        self.extension_dir = ext_dir
        self.config_dir = self.extension_dir / "configs"
        
        logger.info("ForgeKontextAssistant initialized")
    
    def title(self) -> str:
        """Return the title of the script"""
        return "FluxKontext Smart Assistant"
    
    def show(self, is_img2img: bool) -> scripts.AlwaysVisible:
        """Show in both txt2img and img2img"""
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img: bool):
        """Build the UI components"""
        # Use InputAccordion to create a collapsible section
        with InputAccordion(False, label="🤖 FluxKontext Smart Assistant") as enabled:
            # Check if modules loaded successfully
            if ka_ui_components is None:
                # Simplified fallback UI
                gr.Markdown("### ⚠️ Running in simplified mode")
                
                with gr.Group():
                    with gr.Row():
                        task_type = gr.Dropdown(
                            label="Task Type",
                            choices=[
                                ("Change Color", "object_color"),
                                ("Add Object", "add_object"),
                                ("Remove Object", "remove_object"),
                                ("Style Transfer", "style_artistic"),
                                ("Change Weather", "environment_weather"),
                                ("Change Time", "environment_time"),
                                ("Change Location", "environment_location")
                            ],
                            value="object_color"
                        )
                    
                    user_intent = gr.Textbox(
                        label="What do you want to do?",
                        placeholder="Examples: make the car blue, add a tree, change to sunset...",
                        lines=2
                    )
                    
                    with gr.Row():
                        generate_btn = gr.Button("✨ Generate Prompt", variant="primary")
                        clear_btn = gr.Button("🔄 Clear", variant="secondary")
                    
                    generated_prompt = gr.Textbox(
                        label="Generated Prompt (copy this to your main prompt)",
                        lines=4,
                        interactive=True
                    )
                    
                    # Simple handlers
                    def generate_simple(task, intent):
                        try:
                            # Basic template-based generation
                            templates = {
                                "object_color": f"Change the color as requested: {intent}. Maintain all other aspects unchanged.",
                                "add_object": f"Add {intent} to the scene while preserving existing composition.",
                                "remove_object": f"Remove {intent} from the image, maintaining natural appearance.",
                                "style_artistic": f"Apply {intent} style while preserving recognizable objects and composition.",
                                "environment_weather": f"Change weather to {intent} while keeping all objects unchanged.",
                                "environment_time": f"Change time to {intent} with appropriate lighting adjustments.",
                                "environment_location": f"Change background to {intent} while preserving foreground subjects."
                            }
                            
                            base = templates.get(task, f"Execute task: {intent}")
                            return base + " Keep all unmentioned elements exactly as they are."
                            
                        except Exception as e:
                            return f"Error: {str(e)}"
                    
                    def clear_fields():
                        return "", ""
                    
                    generate_btn.click(
                        fn=generate_simple,
                        inputs=[task_type, user_intent],
                        outputs=[generated_prompt]
                    )
                    
                    clear_btn.click(
                        fn=clear_fields,
                        inputs=[],
                        outputs=[user_intent, generated_prompt]
                    )
                
                return [enabled, task_type, user_intent, generate_btn, 
                       clear_btn, generated_prompt]
            
            else:
                # Full UI from ui_components module
                try:
                    components = ka_ui_components.build_assistant_ui(self, None, is_img2img)
                    return [enabled] + components
                except Exception as e:
                    logger.error(f"Error building full UI: {str(e)}")
                    gr.Markdown(f"### ❌ Error: {str(e)}")
                    return [enabled]
    
    def _lazy_load_generator(self):
        """Lazy load the prompt generator"""
        if self.generator is None and ka_prompt_generator is not None:
            logger.info("Loading prompt generator...")
            try:
                self.generator = ka_prompt_generator.PromptGenerator(self.config_dir)
                logger.info("Prompt generator loaded successfully")
            except Exception as e:
                logger.error(f"Could not load prompt generator: {e}")
                self.generator = None
    
    def run(self, p, *args):
        """Main processing - we don't modify the generation"""
        # Extract the enabled state from args
        enabled = args[0] if args else False
        
        if enabled:
            logger.info("FluxKontext Smart Assistant is enabled for this generation")
        
        # We don't modify the generation process
        pass
    
    def postprocess(self, p, processed, *args):
        """Post-processing - clean up resources"""
        # Clear any caches
        if hasattr(self, '_analysis_cache'):
            self._analysis_cache.clear()


# Create script instance - Forge looks for this
forgeKontextAssistant = ForgeKontextAssistant()