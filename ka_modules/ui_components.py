"""
UI components for FluxKontext Smart Assistant.

This module handles all Gradio UI building and event handling for the
assistant interface, following Forge WebUI patterns.
"""

import gradio as gr
import logging
from typing import List, Optional, Tuple, Dict, Any, Callable
from pathlib import Path

logger = logging.getLogger("KontextAssistant.UI")


def build_assistant_ui(script_instance, kontext_images: Optional[List] = None, 
                      is_img2img: bool = False) -> List:
    """
    Build the complete UI for Kontext Smart Assistant.
    
    Args:
        script_instance: The KontextAssistant instance
        kontext_images: Optional reference to kontext image components
        is_img2img: Whether in img2img mode
        
    Returns:
        List of gradio components for the script
    """
    components = []
    
    with gr.Group(elem_id="kontext_assistant_group"):
        gr.Markdown("### 🤖 Kontext Smart Assistant")
        gr.Markdown(
            "Analyze your context images and generate perfect prompts for FLUX.1 Kontext. "
            "Upload images in the main Kontext section above, then use this assistant."
        )
        
        # Analysis section for each image
        with gr.Group():
            gr.Markdown("#### 📊 Image Analysis")
            
            analyze_buttons = []
            analysis_outputs = []
            analysis_accordions = []
            
            # Create analysis UI for up to 3 images
            for i in range(3):
                with gr.Row():
                    with gr.Column(scale=1):
                        btn = gr.Button(
                            f"🔍 Analyze Image {i+1}",
                            variant="secondary",
                            size="sm",
                            elem_id=f"analyze_btn_{i}"
                        )
                        analyze_buttons.append(btn)
                    
                    with gr.Column(scale=3):
                        # Collapsible analysis results
                        with gr.Accordion(
                            f"Analysis Results {i+1}",
                            open=False,
                            elem_id=f"analysis_accordion_{i}"
                        ) as accordion:
                            analysis_accordions.append(accordion)
                            
                            output = gr.Markdown(
                                value="*No analysis yet*",
                                elem_id=f"analysis_output_{i}"
                            )
                            analysis_outputs.append(output)
        
        # Task selection and prompt generation
        with gr.Group():
            gr.Markdown("#### 🎯 Prompt Generation")
            
            with gr.Row():
                task_type = gr.Dropdown(
                    label="Task Type",
                    choices=[
                        ("Change Object Color", "object_color"),
                        ("Change Object State", "object_state"),
                        ("Add Elements", "add_object"),
                        ("Remove Elements", "remove_object"),
                        ("Artistic Style Transfer", "style_artistic"),
                        ("Time Period Style", "style_temporal"),
                        ("Change Weather", "environment_weather"),
                        ("Change Time of Day", "environment_time"),
                        ("Change Location", "environment_location"),
                        ("Extend Image (Outpainting)", "outpainting")
                    ],
                    value="object_color",
                    elem_id="task_type_dropdown"
                )
                
                # Task-specific help
                task_help = gr.Markdown(
                    value=_get_task_help("object_color"),
                    elem_id="task_help_text"
                )
            
            user_intent = gr.Textbox(
                label="What do you want to do?",
                placeholder="Example: make the car blue, add rain, change to sunset...",
                lines=2,
                elem_id="user_intent_input"
            )
            
            # Generate button
            generate_btn = gr.Button(
                "✨ Generate Prompt",
                variant="primary",
                elem_id="generate_prompt_btn"
            )
            
            # Generated prompt output
            generated_prompt = gr.Textbox(
                label="Generated Prompt",
                lines=4,
                interactive=True,
                elem_id="generated_prompt_output"
            )
            
            # Action buttons
            with gr.Row():
                copy_btn = gr.Button(
                    "📋 Copy to Prompt",
                    size="sm",
                    elem_id="copy_prompt_btn"
                )
                
                reset_btn = gr.Button(
                    "🔄 Reset",
                    size="sm",
                    elem_id="reset_btn"
                )
        
        # Advanced options (collapsible)
        with gr.Accordion("⚙️ Advanced Options", open=False):
            use_enhancement = gr.Checkbox(
                label="Use Phi-3 Enhancement (Better for complex/creative requests)",
                value=False,
                elem_id="use_enhancement_checkbox"
            )
            
            preservation_strength = gr.Slider(
                label="Context Preservation Strength",
                minimum=0.5,
                maximum=1.0,
                value=0.8,
                step=0.05,
                elem_id="preservation_slider"
            )
            
            show_details = gr.Checkbox(
                label="Show detailed analysis",
                value=False,
                elem_id="show_details_checkbox"
            )
            
            # Debug info
            with gr.Accordion("🐛 Debug Info", open=False):
                debug_output = gr.Textbox(
                    label="Debug Information",
                    lines=5,
                    interactive=False,
                    elem_id="debug_output"
                )
    
    # Store components for reference
    components = [
        task_type,
        user_intent,
        generated_prompt,
        use_enhancement,
        preservation_strength,
        show_details,
        debug_output
    ]
    
    # Add buttons and outputs to components
    components.extend(analyze_buttons)
    components.extend(analysis_outputs)
    
    # Set up event handlers
    _setup_event_handlers(
        script_instance=script_instance,
        analyze_buttons=analyze_buttons,
        analysis_outputs=analysis_outputs,
        analysis_accordions=analysis_accordions,
        task_type=task_type,
        task_help=task_help,
        user_intent=user_intent,
        generate_btn=generate_btn,
        generated_prompt=generated_prompt,
        use_enhancement=use_enhancement,
        preservation_strength=preservation_strength,
        copy_btn=copy_btn,
        reset_btn=reset_btn,
        debug_output=debug_output,
        show_details=show_details
    )
    
    return components


def _setup_event_handlers(
    script_instance,
    analyze_buttons: List[gr.Button],
    analysis_outputs: List[gr.Markdown],
    analysis_accordions: List[gr.Accordion],
    task_type: gr.Dropdown,
    task_help: gr.Markdown,
    user_intent: gr.Textbox,
    generate_btn: gr.Button,
    generated_prompt: gr.Textbox,
    use_enhancement: gr.Checkbox,
    preservation_strength: gr.Slider,
    copy_btn: gr.Button,
    reset_btn: gr.Button,
    debug_output: gr.Textbox,
    show_details: gr.Checkbox
):
    """Set up all event handlers for the UI components"""
    
    # Analysis button handlers
    for i, (btn, output, accordion) in enumerate(zip(
        analyze_buttons, analysis_outputs, analysis_accordions
    )):
        def create_analyze_handler(idx):
            def analyze_handler():
                try:
                    # This is a placeholder - actual implementation would get the image
                    # from kontext_images or find it in the UI
                    logger.info(f"Analyzing image {idx + 1}")
                    
                    # For now, return a mock analysis
                    analysis_text = f"""
🎯 **Objects**: car (red), trees, road
🎨 **Style**: photorealistic, bright daylight
🌍 **Environment**: urban street, clear weather
💡 **Lighting**: natural sunlight from top-right
📐 **Composition**: centered subject, eye-level view
                    """.strip()
                    
                    return {
                        output: analysis_text,
                        accordion: gr.update(open=True)
                    }
                    
                except Exception as e:
                    logger.error(f"Error in analysis: {str(e)}")
                    return {
                        output: f"❌ Error: {str(e)}",
                        accordion: gr.update(open=True)
                    }
            
            return analyze_handler
        
        btn.click(
            fn=create_analyze_handler(i),
            inputs=[],
            outputs=[output, accordion]
        )
    
    # Task type change handler
    def on_task_change(task):
        help_text = _get_task_help(task)
        return help_text
    
    task_type.change(
        fn=on_task_change,
        inputs=[task_type],
        outputs=[task_help]
    )
    
    # Generate prompt handler
    def generate_prompt_handler(task, intent, use_enh, preservation, show_det):
        try:
            logger.info(f"Generating prompt - Task: {task}, Intent: {intent}")
            
            # Get cached analyses if available
            analyses = getattr(script_instance, '_analysis_cache', {})
            
            # Generate prompt
            prompt = script_instance.generate_prompt(
                task_type=task,
                user_intent=intent,
                use_enhancement=use_enh,
                image_analyses=list(analyses.values()) if analyses else None
            )
            
            # Debug info
            debug_info = f"Task: {task}\nIntent: {intent}\nEnhancement: {use_enh}\n"
            debug_info += f"Analyses available: {len(analyses)}\n"
            debug_info += f"Generated length: {len(prompt)} chars"
            
            return prompt, debug_info
            
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return f"Error: {str(e)}", f"Error occurred: {str(e)}"
    
    generate_btn.click(
        fn=generate_prompt_handler,
        inputs=[task_type, user_intent, use_enhancement, preservation_strength, show_details],
        outputs=[generated_prompt, debug_output]
    )
    
    # Copy prompt handler (placeholder - actual implementation would interact with main prompt field)
    def copy_prompt_handler(prompt):
        logger.info("Copy prompt clicked")
        return gr.update(value="✅ Copied!")
    
    copy_btn.click(
        fn=copy_prompt_handler,
        inputs=[generated_prompt],
        outputs=[copy_btn]
    ).then(
        lambda: gr.update(value="📋 Copy to Prompt"),
        inputs=[],
        outputs=[copy_btn],
        _js="setTimeout(() => {}, 2000)"  # Reset button text after 2 seconds
    )
    
    # Reset handler
    def reset_handler():
        return (
            "",  # user_intent
            "",  # generated_prompt
            False,  # use_enhancement
            0.8,  # preservation_strength
            ""  # debug_output
        )
    
    reset_btn.click(
        fn=reset_handler,
        inputs=[],
        outputs=[user_intent, generated_prompt, use_enhancement, 
                preservation_strength, debug_output]
    )


def _get_task_help(task_type: str) -> str:
    """Get help text for a specific task type"""
    
    help_texts = {
        "object_color": "**Object Color Change** - Changes the color of specific objects. Example: 'red car to blue'",
        "object_state": "**Object State Change** - Transforms objects between states. Example: 'open the door'",
        "add_object": "**Add Elements** - Adds new objects to the scene. Example: 'add a cat on the sofa'",
        "remove_object": "**Remove Elements** - Removes objects cleanly. Example: 'remove the person'",
        "style_artistic": "**Artistic Style** - Applies art styles. Example: 'impressionist style'",
        "style_temporal": "**Time Period Style** - Changes era aesthetic. Example: 'vintage 1950s'",
        "environment_weather": "**Weather Change** - Modifies weather conditions. Example: 'make it rainy'",
        "environment_time": "**Time of Day** - Changes lighting and time. Example: 'sunset lighting'",
        "environment_location": "**Location Change** - Replaces background. Example: 'beach background'",
        "outpainting": "**Extend Image** - Expands beyond borders. Example: 'extend left with more trees'"
    }
    
    return help_texts.get(task_type, "Select a task type to see description")


def create_quick_access_buttons() -> List[gr.Button]:
    """Create quick access buttons for common tasks"""
    
    common_tasks = [
        ("🎨 Change Color", "object_color"),
        ("🌦️ Change Weather", "environment_weather"),
        ("🎭 Apply Style", "style_artistic"),
        ("➕ Add Object", "add_object")
    ]
    
    buttons = []
    with gr.Row():
        for label, task in common_tasks:
            btn = gr.Button(label, size="sm")
            buttons.append((btn, task))
    
    return buttons


# Standalone function for testing UI
if __name__ == "__main__":
    # Test UI building
    with gr.Blocks() as demo:
        gr.Markdown("# FluxKontext Smart Assistant Test UI")
        
        # Mock script instance
        class MockScript:
            def generate_prompt(self, **kwargs):
                return "Generated test prompt based on your input..."
            
            def analyze_image(self, image, index):
                return {"objects": {"main": ["test object"]}}
        
        mock_script = MockScript()
        components = build_assistant_ui(mock_script)
        
    print("UI components created successfully!")
    print(f"Total components: {len(components)}")