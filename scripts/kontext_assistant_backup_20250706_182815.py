"""
Kontext Smart Assistant - Intelligent prompt generation for FLUX.1 Kontext
Simplified version for testing with proper imports
"""

import gradio as gr
import torch
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Forge WebUI imports
try:
    from ka_modules import scripts, shared
    from ka_modules.ui_components import InputAccordion
except ImportError:
    # Fallback for testing outside Forge
    scripts = None
    shared = None
    InputAccordion = None

# Import our modules with fixed path
from ka_modules.templates import PromptTemplates
from ka_modules.prompt_generator import PromptGenerator
from ka_modules.image_analyzer import ImageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KontextAssistant")

class KontextAssistant(scripts.Script if scripts else object):
    """Smart Assistant for FLUX.1 Kontext prompt generation"""
    
    def __init__(self):
        super().__init__()
        self.prompt_generator = PromptGenerator()
        self.image_analyzer = ImageAnalyzer()
        self.kontext_images = []
        self.analysis_results = {}
        
    def title(self):
        """Return the title of the script"""
        return "Kontext Smart Assistant"
    
    def show(self, is_img2img):
        """Determine when to show the script"""
        return scripts.AlwaysVisible if scripts else True
    
    def _find_kontext_images(self) -> List[Any]:
        """Find kontext images from the main kontext.py script"""
        try:
            # Method 1: Try to get from ForgeKontext state
            from scripts.kontext import ForgeKontext
            state = ForgeKontext.get_current_kontext_state()
            if state and hasattr(state, 'kontext_images'):
                return state.kontext_images
        except:
            pass
        
        try:
            # Method 2: Search in gradio components
            import gradio as gr
            # Look for kontext image components in the current gradio blocks
            for component in gr.context.Context.block.children:
                if hasattr(component, 'elem_id') and 'kontext' in str(component.elem_id):
                    if hasattr(component, 'value') and component.value:
                        self.kontext_images.append(component.value)
        except:
            pass
        
        logger.info(f"Found {len(self.kontext_images)} kontext images")
        return self.kontext_images
    
    def analyze_image(self, image_index: int, task_type: str) -> str:
        """Analyze a kontext image"""
        try:
            # Get kontext images
            images = self._find_kontext_images()
            
            if not images or image_index >= len(images):
                return "No image found at this index. Please load images in Kontext first."
            
            image = images[image_index]
            if image is None:
                return "Empty image slot."
            
            # Analyze image
            logger.info(f"Analyzing image {image_index + 1} for task: {task_type}")
            analysis = self.image_analyzer.analyze(image, task_type)
            
            # Store results
            self.analysis_results[image_index] = analysis
            
            # Format output
            output = f"=== Image {image_index + 1} Analysis ===\n\n"
            output += f"Task Context: {task_type}\n\n"
            
            if 'objects' in analysis:
                output += f"**Main Objects**: {', '.join(analysis['objects']['main'])}\n"
                output += f"**Secondary**: {', '.join(analysis['objects']['secondary'])}\n\n"
            
            if 'style' in analysis:
                output += f"**Style**: {analysis['style']['artistic']}\n"
                output += f"**Mood**: {analysis['style']['mood']}\n\n"
            
            if 'environment' in analysis:
                output += f"**Setting**: {analysis['environment']['setting']}\n"
                output += f"**Time**: {analysis['environment']['time_of_day']}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    
    def generate_prompt(self, task_type: str, user_intent: str, context_strength: float) -> str:
        """Generate FLUX.1 Kontext prompt"""
        try:
            # Validate inputs
            if not task_type or not user_intent:
                return "Please select a task type and describe your intent."
            
            # Get all analysis results
            if not self.analysis_results:
                # Try to find and analyze images automatically
                images = self._find_kontext_images()
                for i, img in enumerate(images[:3]):  # Max 3 images
                    if img is not None:
                        self.analyze_image(i, task_type)
            
            # Generate prompt
            prompt = self.prompt_generator.generate(
                task_type=task_type,
                user_intent=user_intent,
                image_analyses=list(self.analysis_results.values()),
                context_strength=context_strength
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return f"Error generating prompt: {str(e)}"
    
    def ui(self, is_img2img):
        """Create the UI components"""
        with gr.Group():
            gr.Markdown("### 🤖 Kontext Smart Assistant")
            gr.Markdown("*Analyzes your context images and generates proper FLUX.1 Kontext prompts*")
            
            # Analysis section
            with gr.Group():
                gr.Markdown("**Step 1: Analyze Context Images**")
                with gr.Row():
                    for i in range(3):
                        analyze_btn = gr.Button(
                            f"Analyze Image {i+1}",
                            variant="secondary",
                            scale=1
                        )
                        # Store button references
                        setattr(self, f"analyze_btn_{i}", analyze_btn)
                
                analysis_output = gr.Textbox(
                    label="Analysis Results",
                    lines=8,
                    interactive=False
                )
            
            # Prompt generation section
            with gr.Group():
                gr.Markdown("**Step 2: Generate Instructional Prompt**")
                
                task_type = gr.Dropdown(
                    label="Task Type",
                    choices=[
                        "object_color",
                        "object_state", 
                        "style_transfer",
                        "environment_change",
                        "element_combination",
                        "state_changes",
                        "outpainting"
                    ],
                    value="object_color"
                )
                
                user_intent = gr.Textbox(
                    label="Your Intent",
                    placeholder="Describe what you want to change (e.g., 'make the car blue')",
                    lines=2
                )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    context_strength = gr.Slider(
                        label="Context Preservation Strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05
                    )
                    
                    use_llm = gr.Checkbox(
                        label="Use Phi-3 Enhancement (requires more VRAM)",
                        value=False
                    )
                
                generate_btn = gr.Button(
                    "Generate Prompt",
                    variant="primary"
                )
                
                generated_prompt = gr.Textbox(
                    label="Generated Prompt",
                    lines=6,
                    interactive=True
                )
                
                with gr.Row():
                    copy_btn = gr.Button("📋 Copy to Prompt", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear", variant="stop")
            
            # Event handlers
            def analyze_wrapper(idx):
                def _analyze(task_type):
                    return self.analyze_image(idx, task_type)
                return _analyze
            
            # Connect analyze buttons
            for i in range(3):
                btn = getattr(self, f"analyze_btn_{i}")
                btn.click(
                    fn=analyze_wrapper(i),
                    inputs=[task_type],
                    outputs=[analysis_output]
                )
            
            # Generate prompt
            generate_btn.click(
                fn=self.generate_prompt,
                inputs=[task_type, user_intent, context_strength],
                outputs=[generated_prompt]
            )
            
            # Clear button
            clear_btn.click(
                fn=lambda: ("", "", {}),
                outputs=[analysis_output, generated_prompt]
            ).then(
                fn=lambda: setattr(self, 'analysis_results', {}),
                outputs=[]
            )
            
            # Return components for script processing
            return [task_type, user_intent, context_strength, use_llm]
    
    def process(self, p, *args):
        """Process is called by Forge but we don't modify generation"""
        # This assistant only helps with prompt generation
        # It doesn't interfere with the actual image generation process
        pass

# For testing outside Forge
if __name__ == "__main__":
    assistant = KontextAssistant()
    print("Kontext Smart Assistant initialized")