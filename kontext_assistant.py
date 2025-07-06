"""
FluxKontext Smart Assistant - Main Script Class
Integrates with Forge WebUI to provide intelligent prompt generation
for FLUX.1 Kontext model based on context image analysis.
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from modules import scripts
from modules.ui_components import InputAccordion

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KontextAssistant")


class KontextAssistant(scripts.Script):
    """
    Smart assistant for FLUX.1 Kontext prompt generation.
    Analyzes context images and generates appropriate instructional prompts.
    """
    
    def __init__(self):
        super().__init__()
        self.analyzer = None      # Lazy load Florence-2
        self.generator = None     # Lazy load prompt generator
        self.enhancer = None      # Lazy load Phi-3 (optional)
        self.initialized = False
        self.sorting_priority = 1  # Show after main Kontext
        
        # Get extension directory
        self.extension_dir = Path(__file__).parent
        self.config_dir = self.extension_dir / "configs"
        
        logger.info("KontextAssistant initialized")
    
    def title(self) -> str:
        """Return the title of the script"""
        return "Kontext Smart Assistant"
    
    def show(self, is_img2img: bool) -> scripts.AlwaysVisible:
        """Always show in both txt2img and img2img tabs"""
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img: bool):
        """Build the UI components"""
        try:
            # Import UI builder
            from .modules.ui_components import build_assistant_ui
            
            # Get reference to kontext images if available
            kontext_images = self._find_kontext_images()
            
            # Build and return UI components
            return build_assistant_ui(self, kontext_images, is_img2img)
            
        except Exception as e:
            logger.error(f"Error building UI: {str(e)}")
            # Return minimal UI on error
            with gr.Group():
                gr.Markdown("❌ Error loading Kontext Assistant UI")
                error_display = gr.Textbox(
                    value=str(e),
                    label="Error Details",
                    interactive=False
                )
            return [error_display]
    
    def _find_kontext_images(self) -> Optional[List]:
        """
        Try to find kontext image components from the main kontext script.
        This allows us to access the same images without duplicating UI.
        """
        # This is a placeholder - actual implementation would need to
        # coordinate with the main kontext.py script
        return None
    
    def _lazy_load_analyzer(self):
        """Lazy load the image analyzer"""
        if self.analyzer is None:
            logger.info("Loading Florence-2 image analyzer...")
            from .modules.image_analyzer import ImageAnalyzer
            self.analyzer = ImageAnalyzer()
            logger.info("Image analyzer loaded successfully")
    
    def _lazy_load_generator(self):
        """Lazy load the prompt generator"""
        if self.generator is None:
            logger.info("Loading prompt generator...")
            from .modules.prompt_generator import PromptGenerator
            self.generator = PromptGenerator(self.config_dir)
            logger.info("Prompt generator loaded successfully")
    
    def _lazy_load_enhancer(self):
        """Lazy load the LLM enhancer (optional)"""
        if self.enhancer is None:
            try:
                logger.info("Loading Phi-3 enhancer...")
                from .modules.llm_enhancer import Phi3Enhancer
                self.enhancer = Phi3Enhancer()
                logger.info("LLM enhancer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load LLM enhancer: {str(e)}")
                logger.warning("Continuing without LLM enhancement")
    
    def analyze_image(self, image, image_index: int) -> Dict[str, Any]:
        """
        Analyze a context image using Florence-2.
        
        Args:
            image: PIL Image to analyze
            image_index: Index of the image (0-2)
            
        Returns:
            Dictionary with structured analysis results
        """
        if image is None:
            return {"error": "No image provided"}
        
        try:
            self._lazy_load_analyzer()
            
            logger.info(f"Analyzing image {image_index + 1}...")
            analysis = self.analyzer.analyze_comprehensive(image)
            
            # Cache the analysis for later use
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}
            self._analysis_cache[image_index] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": str(e)}
    
    def generate_prompt(
        self,
        task_type: str,
        user_intent: str,
        use_enhancement: bool = False,
        image_analyses: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a FLUX.1 Kontext prompt based on task and intent.
        
        Args:
            task_type: Selected task type (e.g., "object_color")
            user_intent: User's natural language intent
            use_enhancement: Whether to use Phi-3 enhancement
            image_analyses: List of image analysis results
            
        Returns:
            Generated instructional prompt
        """
        try:
            self._lazy_load_generator()
            
            # Use cached analyses if not provided
            if image_analyses is None and hasattr(self, '_analysis_cache'):
                image_analyses = list(self._analysis_cache.values())
            
            # Generate base prompt
            logger.info(f"Generating prompt for task: {task_type}")
            base_prompt = self.generator.generate(
                task_type=task_type,
                user_intent=user_intent,
                image_analyses=image_analyses
            )
            
            # Enhance with LLM if requested
            if use_enhancement and self._should_enhance(user_intent):
                try:
                    self._lazy_load_enhancer()
                    if self.enhancer:
                        logger.info("Enhancing prompt with Phi-3...")
                        enhanced_prompt = self.enhancer.enhance_prompt(
                            base_prompt=base_prompt,
                            user_intent=user_intent,
                            context=image_analyses
                        )
                        return enhanced_prompt
                except Exception as e:
                    logger.warning(f"Enhancement failed, using base prompt: {str(e)}")
            
            return base_prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return f"Error: Could not generate prompt - {str(e)}"
    
    def _should_enhance(self, user_intent: str) -> bool:
        """
        Determine if LLM enhancement would be beneficial.
        
        Args:
            user_intent: User's natural language intent
            
        Returns:
            Boolean indicating if enhancement is recommended
        """
        # Simple heuristics for now
        enhancement_triggers = [
            "like", "style of", "similar to", "but keep", "except",
            "как", "в стиле", "похоже на", "но сохранить", "кроме"
        ]
        
        intent_lower = user_intent.lower()
        return any(trigger in intent_lower for trigger in enhancement_triggers)
    
    def format_analysis_display(self, analysis: Dict[str, Any]) -> str:
        """
        Format analysis results for display in UI.
        
        Args:
            analysis: Analysis dictionary from Florence-2
            
        Returns:
            Formatted string for display
        """
        if "error" in analysis:
            return f"❌ Error: {analysis['error']}"
        
        try:
            sections = []
            
            # Objects section
            if "objects" in analysis:
                obj_data = analysis["objects"]
                main_objs = ", ".join(obj_data.get("main", []))
                sections.append(f"🎯 **Objects**: {main_objs}")
            
            # Style section
            if "style" in analysis:
                style_data = analysis["style"]
                style_desc = f"{style_data.get('artistic', 'unknown')} - {style_data.get('mood', 'neutral')}"
                sections.append(f"🎨 **Style**: {style_desc}")
            
            # Environment section
            if "environment" in analysis:
                env_data = analysis["environment"]
                env_desc = f"{env_data.get('setting', 'unknown')} ({env_data.get('time_of_day', 'day')})"
                sections.append(f"🌍 **Environment**: {env_desc}")
            
            # Lighting section
            if "lighting" in analysis:
                light_data = analysis["lighting"]
                light_desc = f"{light_data.get('quality', 'neutral')} from {light_data.get('direction', 'unknown')}"
                sections.append(f"💡 **Lighting**: {light_desc}")
            
            return "\n".join(sections)
            
        except Exception as e:
            logger.error(f"Error formatting analysis: {str(e)}")
            return "Error formatting analysis results"
    
    def run(self, p, *args):
        """
        Main processing - we don't modify the generation process,
        just provide prompt assistance.
        """
        # This script doesn't modify the generation process
        pass
    
    def postprocess(self, p, processed, *args):
        """
        Post-processing - clean up any resources if needed.
        """
        # Clear analysis cache after generation
        if hasattr(self, '_analysis_cache'):
            self._analysis_cache.clear()


# Register the script
kontext_assistant = KontextAssistant()